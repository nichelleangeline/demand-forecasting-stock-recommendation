# app/services/forecast_service.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import text

from app.db import engine
from app.services.model_service import get_all_model_runs


# =========================================================
# Helper: ambil model_run aktif
# =========================================================

def _get_active_lgbm_run():
    """
    Ambil 1 row model_run yang active_flag = 1
    dan model_type LGBM (sesuai admin_forecast_train: 'lgbm').
    """
    runs = get_all_model_runs()
    if not runs:
        return None

    for r in runs:
        mtype = (r.get("model_type") or "").lower()
        # UPDATED: sekarang pakai model_type 'lgbm' (atau turunan)
        if r.get("active_flag") == 1 and mtype.startswith("lgbm"):
            return r
    return None


def _load_cluster_models(run_dir: Path) -> dict:
    """
    Load semua model cluster dari folder models di run_dir.
    Format file (sesuai training admin_forecast_train.py):
        lgbm_cluster_{cid}.txt
    """
    model_dir = run_dir / "models"
    models = {}

    if not model_dir.exists():
        raise FileNotFoundError(f"Folder models tidak ditemukan: {model_dir}")

    # UPDATED: pola nama file disamakan dengan training: lgbm_cluster_*.txt
    for p in model_dir.glob("lgbm_cluster_*.txt"):
        # nama: lgbm_cluster_{cid}.txt
        try:
            cid = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        booster = lgb.Booster(model_file=str(p))
        models[cid] = booster

    if not models:
        raise ValueError(f"Tidak ada file model cluster di {model_dir}")

    return models


def _get_exog_map():
    """
    Ambil external_data ke dict:
        key: (cabang, periode_month)
        val: dict(event_flag, holiday_count, rainfall)
    """
    sql = """
        SELECT cabang, periode, event_flag, holiday_count, rainfall
        FROM external_data
    """
    with engine.begin() as conn:
        exog = pd.read_sql(sql, conn, parse_dates=["periode"])

    exog["event_flag"] = (
        pd.to_numeric(exog["event_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    exog["holiday_count"] = (
        pd.to_numeric(exog["holiday_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    exog["rainfall"] = (
        pd.to_numeric(exog["rainfall"], errors="coerce")
        .fillna(0.0)
    )

    exog_map = {}
    for _, row in exog.iterrows():
        key = (row["cabang"], row["periode"].to_period("M"))
        exog_map[key] = {
            "event_flag": int(row["event_flag"]),
            "holiday_count": int(row["holiday_count"]),
            "rainfall": float(row["rainfall"]),
        }
    return exog_map


def _next_month(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Geser ke bulan berikutnya (tanggal 1).
    """
    year = ts.year
    month = ts.month + 1
    if month == 13:
        month = 1
        year += 1
    return pd.Timestamp(year=year, month=month, day=1)


# =========================================================
# Forecast future per SKU
# =========================================================

def _build_future_rows_for_sku(
    hist_row: pd.Series,
    feature_cols: list[str],
    model: lgb.Booster,
    horizon: int,
    exog_map: dict,
) -> list[dict]:
    """
    Forecast ke depan untuk 1 cabang-sku, secara iteratif (multi-step).
    - hist_row: baris terakhir untuk SKU tsb (periode terakhir history)
    - feature_cols: list fitur yang dipakai model
    - model: booster cluster yang sesuai
    - horizon: H bulan ke depan
    - exog_map: dict (cabang, periode_month) -> event/holiday/rainfall
    """
    cabang = hist_row["cabang"]
    sku = hist_row["sku"]

    last_periode = pd.Timestamp(hist_row["periode"])
    last_qty = float(hist_row["qty"])

    # ambil qty_lag1..12 awal dari history terakhir
    lag_names = [f"qty_lag{i}" for i in range(1, 13)]
    lags = {}
    for i, name in enumerate(lag_names, start=1):
        lags[i] = float(hist_row.get(name, np.nan))

    results = []
    current_row_base = hist_row.copy()

    for h in range(1, horizon + 1):
        per_fore = last_periode.to_period("M") + h
        per_fore_ts = per_fore.to_timestamp()

        # exog future: current & previous month
        key_this = (cabang, per_fore)
        key_prev = (cabang, per_fore - 1)

        ex_this = exog_map.get(
            key_this,
            {"event_flag": 0, "holiday_count": 0, "rainfall": 0.0},
        )
        ex_prev = exog_map.get(
            key_prev,
            {"event_flag": 0, "holiday_count": 0, "rainfall": 0.0},
        )

        event_flag = int(ex_this["event_flag"])
        holiday_count = int(ex_this["holiday_count"])
        event_flag_lag1 = int(ex_prev["event_flag"])
        holiday_count_lag1 = int(ex_prev["holiday_count"])

        if cabang == "16C":
            rainfall_lag1 = float(ex_prev["rainfall"])
        else:
            rainfall_lag1 = 0.0

        # update lags qty (lag1 = last_qty, lag2..12 geser)
        new_lags = {}
        new_lags[1] = last_qty
        for i in range(2, 13):
            new_lags[i] = lags.get(i - 1, np.nan)

        # rolling features dari lags
        def _roll_stats(max_w):
            vals = [
                new_lags[i]
                for i in range(1, max_w + 1)
                if not np.isnan(new_lags[i])
            ]
            if len(vals) == 0:
                return np.nan, np.nan
            mean = float(np.mean(vals))
            if len(vals) >= 2:
                std = float(np.std(vals, ddof=1))
            else:
                std = 0.0
            return mean, std

        roll_features = {}
        for w in (3, 6, 12):
            m, s = _roll_stats(w)
            roll_features[f"qty_rollmean_{w}"] = m
            roll_features[f"qty_rollstd_{w}"] = s

        # row baru: copy base lalu override field yang berubah
        row = current_row_base.copy()

        row["periode"] = per_fore_ts
        row["qty"] = np.nan  # future, tidak ada actual

        row["event_flag"] = event_flag
        row["event_flag_lag1"] = event_flag_lag1
        row["holiday_count"] = holiday_count
        row["holiday_count_lag1"] = holiday_count_lag1
        row["rainfall_lag1"] = rainfall_lag1

        row["is_train"] = 0
        row["is_test"] = 0
        row["imputed"] = 0
        row["spike_flag"] = 0
        row["sample_weight"] = 1.0

        row["month"] = per_fore_ts.month
        row["year"] = per_fore_ts.year
        row["qtr"] = per_fore_ts.quarter

        # assign lag qty & rolling
        for i in range(1, 13):
            row[f"qty_lag{i}"] = new_lags[i]
        for name, val in roll_features.items():
            row[name] = val

        # pastikan semua feature_cols ada
        for c in feature_cols:
            if c not in row.index:
                row[c] = np.nan

        # prediksi
        X = row[feature_cols].values.reshape(1, -1)
        pred_log = model.predict(X)
        pred_qty = float(np.expm1(pred_log[0]))

        row["pred_qty"] = pred_qty
        row["is_future"] = 1
        row["horizon"] = h

        results.append(row.to_dict())

        # update state untuk step berikutnya
        last_qty = pred_qty
        lags = new_lags
        current_row_base = row

    return results


# =========================================================
# Forecast future semua SKU (tanpa simpan)
# =========================================================

def forecast_future_all_sku(
    horizon_months: int = 6,
) -> pd.DataFrame:
    """
    Forecast future untuk SEMUA cabang-SKU yang ada di panel training aktif.
    - Pakai model_run aktif (LGBM clusters)
    - Pakai panel_with_predictions sebagai history base
    - Forecast H bulan ke depan per SKU secara iteratif
    Return:
        DataFrame dengan history + future:
          - cabang, sku, periode
          - qty (history saja yang terisi)
          - pred_qty (history & future)
          - is_train, is_test, is_future, horizon
    """
    active_run = _get_active_lgbm_run()
    if not active_run:
        raise RuntimeError("Tidak ada model_run aktif untuk LGBM.")

    params = json.loads(active_run["params_json"])
    feature_cols = json.loads(active_run["feature_cols_json"])

    run_dir = Path(params["run_dir"])
    panel_path = Path(params["panel_path"])

    if not panel_path.exists():
        raise FileNotFoundError(
            f"panel_with_predictions.csv tidak ditemukan: {panel_path}"
        )

    # history panel + pred train/test dari run aktif
    df_hist = pd.read_csv(panel_path, parse_dates=["periode"])

    # cek kolom fitur
    missing = [c for c in feature_cols if c not in df_hist.columns]
    if missing:
        raise ValueError(f"Kolom fitur hilang di panel history: {missing}")

    # load semua model cluster
    models = _load_cluster_models(run_dir)

    # exog
    exog_map = _get_exog_map()

    # row terakhir per cabang-sku
    df_last = (
        df_hist.sort_values(["cabang", "sku", "periode"])
        .groupby(["cabang", "sku"], as_index=False)
        .tail(1)
    )

    if "cluster" not in df_last.columns:
        raise ValueError("Kolom 'cluster' tidak ada di panel history.")

    df_last = df_last[df_last["cluster"].notna()].copy()
    df_last["cluster"] = df_last["cluster"].astype(int)

    future_rows = []

    for cid, g in df_last.groupby("cluster", sort=False):
        if cid not in models:
            # cluster yang tidak ada modelnya, skip
            continue

        model = models[cid]
        for _, row in g.iterrows():
            rows_future = _build_future_rows_for_sku(
                hist_row=row,
                feature_cols=feature_cols,
                model=model,
                horizon=horizon_months,
                exog_map=exog_map,
            )
            future_rows.extend(rows_future)

    if not future_rows:
        raise RuntimeError(
            "Tidak ada baris future yang berhasil dibuat. Cek cluster & model."
        )

    df_future = pd.DataFrame(future_rows)

    # tandai history
    df_hist2 = df_hist.copy()
    df_hist2["is_future"] = 0
    df_hist2["horizon"] = 0
    if "pred_qty" not in df_hist2.columns:
        df_hist2["pred_qty"] = np.nan

    df_all = (
        pd.concat([df_hist2, df_future], ignore_index=True)
        .sort_values(["cabang", "sku", "periode"])
        .reset_index(drop=True)
    )

    return df_all


# =========================================================
# Generate + SAVE ke DB
# =========================================================

def generate_and_store_forecast(
    horizon_months: int = 6,
) -> int:
    """
    Hitung forecast future semua SKU untuk model_run aktif,
    lalu simpan ke tabel forecast_monthly.
    - Hapus dulu data forecast lama untuk model_run aktif.
    Return: jumlah baris yang disimpan.
    """
    active_run = _get_active_lgbm_run()
    if not active_run:
        raise RuntimeError("Tidak ada model_run aktif untuk LGBM.")

    model_run_id = int(active_run["id"])

    # 1) hitung forecast full (historical + future)
    df_all = forecast_future_all_sku(horizon_months=horizon_months)

    if df_all.empty:
        return 0

    # pastikan kolom area ada
    if "area" not in df_all.columns:
        df_all["area"] = ""

    # 2) pilih dan rename kolom untuk disimpan
    df_save = df_all[
        [
            "area",
            "cabang",
            "sku",
            "periode",
            "qty",        # → qty_actual
            "pred_qty",
            "is_train",
            "is_test",
            "is_future",
            "horizon",
        ]
    ].copy()

    df_save.rename(columns={"qty": "qty_actual"}, inplace=True)

    # 3) tipe kolom dasar
    df_save["area"] = df_save["area"].astype(str)
    df_save["cabang"] = df_save["cabang"].astype(str)
    df_save["sku"] = df_save["sku"].astype(str)

    df_save["qty_actual"] = df_save["qty_actual"].astype(float)

    # pred_qty: paksa float dan finite
    df_save["pred_qty"] = df_save["pred_qty"].astype(float)
    df_save.loc[~np.isfinite(df_save["pred_qty"]), "pred_qty"] = 0.0

    # flag & horizon: jangan dibiarkan NaN
    for col in ["is_train", "is_test", "is_future", "horizon"]:
        if col in df_save.columns:
            df_save[col] = df_save[col].fillna(0).astype(int)

    # tambahkan model_run_id
    df_save["model_run_id"] = model_run_id

    # 4) convert ke list of dict
    records = df_save.to_dict(orient="records")

    # 5) bersihkan NaN / inf di level record
    cleaned_records = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    if k == "pred_qty":
                        v = 0.0
                    else:
                        v = None
            clean[k] = v
        cleaned_records.append(clean)

    if not cleaned_records:
        return 0

    insert_sql = """
        INSERT INTO forecast_monthly (
            model_run_id,
            area,
            cabang,
            sku,
            periode,
            qty_actual,
            pred_qty,
            is_train,
            is_test,
            is_future,
            horizon
        ) VALUES (
            :model_run_id,
            :area,
            :cabang,
            :sku,
            :periode,
            :qty_actual,
            :pred_qty,
            :is_train,
            :is_test,
            :is_future,
            :horizon
        )
    """

    with engine.begin() as conn:
        # hapus forecast lama untuk model_run ini dulu
        conn.execute(
            text("DELETE FROM forecast_monthly WHERE model_run_id = :mid"),
            {"mid": model_run_id},
        )
        conn.execute(text(insert_sql), cleaned_records)

    return len(cleaned_records)

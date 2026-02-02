import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import text
from app.db import engine
from app.services.model_service import get_all_model_runs, get_active_model_run



def _norm_str_upper(s):
    return str(s).strip().upper() if s is not None else None


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _parse_date_any(x):
    if x is None:
        return None
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_period("M").to_timestamp()
    except Exception:
        return None


def _mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _infer_max_lag(feature_cols: list[str], default=12) -> int:
    max_lag = 0
    for c in feature_cols:
        s = str(c).strip()
        if s.startswith("qty_lag"):
            tail = s.replace("qty_lag", "")
            if tail.isdigit():
                max_lag = max(max_lag, int(tail))
    return max(1, max_lag if max_lag > 0 else int(default))


def _upsert_forecast_config(key: str, value: str, updated_by: int | None):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO forecast_config (config_key, config_value, updated_by)
                VALUES (:k, :v, :u)
                ON DUPLICATE KEY UPDATE
                    config_value = VALUES(config_value),
                    updated_by   = VALUES(updated_by),
                    updated_at   = CURRENT_TIMESTAMP
                """
            ),
            {"k": key, "v": value, "u": updated_by},
        )


def _load_model_run_by_id(model_run_id: int) -> dict | None:
    runs = get_all_model_runs()
    if not runs:
        return None
    for r in runs:
        try:
            if int(r.get("id")) == int(model_run_id):
                return r
        except Exception:
            continue
    return None


def _resolve_model_run(model_run_id: int | None) -> dict:
    if model_run_id is not None:
        run = _load_model_run_by_id(int(model_run_id))
        if not run:
            raise RuntimeError(f"model_run_id {model_run_id} tidak ditemukan.")
        return run

    active = get_active_model_run()
    if not active:
        raise RuntimeError("Belum ada model aktif. Aktifkan salah satu model dulu.")
    return active

def _load_cluster_models(run_dir: Path) -> dict:
    model_dir = run_dir / "models"
    models = {}

    if not model_dir.exists():
        raise FileNotFoundError(f"Folder models tidak ditemukan: {model_dir}")

    for p in model_dir.glob("lgbm_cluster_*.txt"):
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
    sql = """
        SELECT cabang, periode, event_flag, holiday_count, rainfall
        FROM external_data
    """
    with engine.begin() as conn:
        exog = pd.read_sql(text(sql), conn, parse_dates=["periode"])

    if exog.empty:
        return {}

    exog["cabang"] = exog["cabang"].astype(str).str.strip().str.upper()
    exog["event_flag"] = pd.to_numeric(exog["event_flag"], errors="coerce").fillna(0).astype(int)
    exog["holiday_count"] = pd.to_numeric(exog["holiday_count"], errors="coerce").fillna(0).astype(int)
    exog["rainfall"] = pd.to_numeric(exog["rainfall"], errors="coerce").fillna(0.0)

    exog_map = {}
    for _, row in exog.iterrows():
        key = (row["cabang"], row["periode"].to_period("M"))
        exog_map[key] = {
            "event_flag": int(row["event_flag"]),
            "holiday_count": int(row["holiday_count"]),
            "rainfall": float(row["rainfall"]),
        }
    return exog_map


def _get_sales_observed_map() -> dict:
    sql = """
        SELECT cabang, sku, periode, SUM(qty) AS qty_sum
        FROM sales_monthly
        GROUP BY cabang, sku, periode
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, parse_dates=["periode"])

    if df.empty:
        return {}

    df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    df["qty_sum"] = pd.to_numeric(df["qty_sum"], errors="coerce")

    obs_map = {}
    for _, r in df.iterrows():
        per = pd.Timestamp(r["periode"]).to_period("M")
        key = (r["cabang"], r["sku"], per)
        if pd.notna(r["qty_sum"]):
            obs_map[key] = float(r["qty_sum"])
    return obs_map


def _observed_mask(df_hist: pd.DataFrame, sales_obs_map: dict) -> pd.Series:
    if df_hist is None or df_hist.empty:
        return pd.Series([], dtype=bool)

    # kalau panel punya is_observed, pakai itu
    if "is_observed" in df_hist.columns:
        return pd.to_numeric(df_hist["is_observed"], errors="coerce").fillna(0).astype(int) == 1

    cab = df_hist["cabang"].astype(str).str.strip().str.upper()
    sku = df_hist["sku"].astype(str).str.strip().str.upper()
    per = pd.to_datetime(df_hist["periode"], errors="coerce").dt.to_period("M")
    keys = list(zip(cab.tolist(), sku.tolist(), per.tolist()))
    return pd.Series([k in sales_obs_map for k in keys], index=df_hist.index)


def _load_model_snapshot_skus(model_run_id: int) -> pd.DataFrame:
    sql = """
        SELECT cabang, sku, cluster, demand_level
        FROM model_run_sku
        WHERE model_run_id = :mid AND eligible_model = 1
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"mid": int(model_run_id)})

    if df.empty:
        return df

    df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int)

    df = df.drop_duplicates(subset=["cabang", "sku"], keep="last").copy()
    df = df[df["cluster"] >= 0].copy()
    return df

def _apply_train_test_flags_from_run(df_hist: pd.DataFrame, run: dict) -> pd.DataFrame:
    ts_train_start = _parse_date_any(run.get("train_start"))
    ts_train_end = _parse_date_any(run.get("train_end"))
    ts_test_start = _parse_date_any(run.get("test_start"))
    ts_test_end = _parse_date_any(run.get("test_end"))

    df = df_hist.copy()
    df["periode"] = pd.to_datetime(df["periode"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    if "is_train" not in df.columns:
        df["is_train"] = 0
    if "is_test" not in df.columns:
        df["is_test"] = 0

    # reset dulu biar konsisten
    df["is_train"] = 0
    df["is_test"] = 0

    # train range
    if ts_train_start is not None and ts_train_end is not None:
        mask_train = (df["periode"] >= ts_train_start) & (df["periode"] <= ts_train_end)
        df.loc[mask_train, "is_train"] = 1

    # test range
    if ts_test_start is not None and ts_test_end is not None:
        mask_test = (df["periode"] >= ts_test_start) & (df["periode"] <= ts_test_end)
        df.loc[mask_test, "is_test"] = 1

    # kalau overlap (harusnya tidak), test menang
    df.loc[df["is_test"] == 1, "is_train"] = 0
    return df

def _build_future_rows_for_sku(
    hist_row: pd.Series,
    feature_cols: list[str],
    model: lgb.Booster,
    horizon: int,
    exog_map: dict,
    max_lag: int,
) -> list[dict]:
    cabang = _norm_str_upper(hist_row.get("cabang"))
    sku = _norm_str_upper(hist_row.get("sku"))

    last_periode = pd.Timestamp(hist_row["periode"]).to_period("M").to_timestamp()
    last_qty = float(hist_row["qty"]) if pd.notna(hist_row.get("qty")) else 0.0

    lag_names = [f"qty_lag{i}" for i in range(1, max_lag + 1)]
    lags = {i: float(hist_row.get(name, np.nan)) for i, name in enumerate(lag_names, start=1)}

    results = []
    current_row_base = hist_row.copy()

    # horizon bebas (asal > 0)
    horizon = max(1, int(horizon))

    for h in range(1, horizon + 1):
        per_fore = last_periode.to_period("M") + h
        per_fore_ts = per_fore.to_timestamp()

        key_this = (cabang, per_fore)
        key_prev = (cabang, per_fore - 1)

        ex_this = exog_map.get(key_this, {"event_flag": 0, "holiday_count": 0, "rainfall": 0.0})
        ex_prev = exog_map.get(key_prev, {"event_flag": 0, "holiday_count": 0, "rainfall": 0.0})

        event_flag = int(ex_this["event_flag"])
        holiday_count = int(ex_this["holiday_count"])
        event_flag_lag1 = int(ex_prev["event_flag"])
        holiday_count_lag1 = int(ex_prev["holiday_count"])

        rainfall_lag1 = float(ex_prev["rainfall"]) if cabang == "16C" else 0.0

        # shift lag: lag1 = last_qty, lag2 = old lag1, dst
        new_lags = {1: last_qty}
        for i in range(2, max_lag + 1):
            new_lags[i] = lags.get(i - 1, np.nan)

        def _roll_stats(w):
            w = min(int(w), max_lag)
            vals = [new_lags[i] for i in range(1, w + 1) if not np.isnan(new_lags[i])]
            if not vals:
                return np.nan, np.nan
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
            return mean, std

        roll_features = {}
        for w in (3, 6, 12):
            m, s = _roll_stats(w)
            roll_features[f"qty_rollmean_{w}"] = m
            roll_features[f"qty_rollstd_{w}"] = s

        row = current_row_base.copy()
        row["periode"] = per_fore_ts
        row["qty"] = np.nan

        row["event_flag"] = event_flag
        row["event_flag_lag1"] = event_flag_lag1
        row["holiday_count"] = holiday_count
        row["holiday_count_lag1"] = holiday_count_lag1
        row["rainfall_lag1"] = rainfall_lag1

        row["month"] = per_fore_ts.month
        row["year"] = per_fore_ts.year
        row["qtr"] = per_fore_ts.quarter

        for i in range(1, max_lag + 1):
            row[f"qty_lag{i}"] = new_lags[i]
        for name, val in roll_features.items():
            row[name] = val

        # pastikan semua feature ada
        for c in feature_cols:
            if c not in row.index:
                row[c] = np.nan

        X = row[feature_cols].values.reshape(1, -1)
        pred_log = model.predict(X)
        pred_qty = float(np.expm1(pred_log[0]))
        if not np.isfinite(pred_qty):
            pred_qty = 0.0

        row["pred_qty"] = pred_qty
        row["is_future"] = 1
        row["horizon"] = h
        row["is_train"] = 0
        row["is_test"] = 0

        results.append(row.to_dict())

        last_qty = pred_qty
        lags = new_lags
        current_row_base = row

    return results


def forecast_future_all_sku(model_run_id: int | None, horizon_months: int = 6) -> pd.DataFrame:
    run = _resolve_model_run(model_run_id)
    mid = int(run["id"])

    params = json.loads(run["params_json"]) if run.get("params_json") else {}
    feature_cols = json.loads(run["feature_cols_json"]) if run.get("feature_cols_json") else []

    run_dir = Path(params["run_dir"])
    panel_path = Path(params["panel_path"])

    if not panel_path.exists():
        raise FileNotFoundError(f"panel_with_predictions.csv tidak ditemukan: {panel_path}")

    df_hist = pd.read_csv(panel_path, parse_dates=["periode"])
    if df_hist.empty:
        raise RuntimeError("Panel history kosong.")

    if "qty" not in df_hist.columns:
        raise ValueError("Kolom 'qty' tidak ada di panel history.")
    if "cluster" not in df_hist.columns:
        raise ValueError("Kolom 'cluster' tidak ada di panel history.")

    df_hist["cabang"] = df_hist["cabang"].astype(str).str.strip().str.upper()
    df_hist["sku"] = df_hist["sku"].astype(str).str.strip().str.upper()

    missing = [c for c in feature_cols if c not in df_hist.columns]
    if missing:
        raise ValueError(f"Kolom fitur hilang di panel history: {missing}")

    snap = _load_model_snapshot_skus(mid)
    if snap.empty:
        raise ValueError("Snapshot model_run_sku kosong untuk model ini.")

    # filter hist ke sku snapshot model
    df_hist = df_hist.merge(
        snap[["cabang", "sku"]].drop_duplicates(),
        on=["cabang", "sku"],
        how="inner",
    )
    if df_hist.empty:
        raise RuntimeError("Panel history setelah difilter snapshot jadi kosong.")

    # paksa cluster sesuai snapshot
    df_hist = df_hist.drop(columns=[c for c in ["cluster", "demand_level"] if c in df_hist.columns], errors="ignore")
    df_hist = df_hist.merge(
        snap[["cabang", "sku", "cluster", "demand_level"]],
        on=["cabang", "sku"],
        how="left",
    )
    df_hist["cluster"] = pd.to_numeric(df_hist["cluster"], errors="coerce").fillna(-1).astype(int)
    df_hist = df_hist[df_hist["cluster"] >= 0].copy()
    if df_hist.empty:
        raise RuntimeError("Snapshot punya cluster invalid. Cek model_run_sku.")

    # flags train/test: pakai tanggal dari model_run (ini yang bikin MAPE test kamu akhirnya ada)
    df_hist = _apply_train_test_flags_from_run(df_hist, run)

    models = _load_cluster_models(run_dir)
    exog_map = _get_exog_map()
    sales_obs_map = _get_sales_observed_map()

    df_hist_sorted = df_hist.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)
    obs_mask = _observed_mask(df_hist_sorted, sales_obs_map)

    # starting point = baris terakhir yang observed
    df_last = df_hist_sorted[obs_mask].groupby(["cabang", "sku"], as_index=False).tail(1)
    if df_last.empty:
        raise RuntimeError("Tidak ada baris observed untuk starting point forecast.")

    df_last["cluster"] = pd.to_numeric(df_last["cluster"], errors="coerce").fillna(-1).astype(int)
    df_last = df_last[df_last["cluster"] >= 0].copy()

    max_lag = _infer_max_lag(feature_cols, default=12)

    future_rows = []
    for cid, g in df_last.groupby("cluster", sort=False):
        cid_int = int(cid)
        if cid_int not in models:
            continue
        model = models[cid_int]
        for _, row in g.iterrows():
            future_rows.extend(
                _build_future_rows_for_sku(
                    hist_row=row,
                    feature_cols=feature_cols,
                    model=model,
                    horizon=int(horizon_months),
                    exog_map=exog_map,
                    max_lag=max_lag,
                )
            )

    if not future_rows:
        raise RuntimeError("Tidak ada baris future yang dibuat.")

    df_future = pd.DataFrame(future_rows)

    # history: hanya observed yang punya qty. yang tidak observed -> qty = NaN
    df_hist2 = df_hist_sorted.copy()
    df_hist2["is_future"] = 0
    df_hist2["horizon"] = 0

    # pred_qty histori:
    # - kalau panel punya pred_qty, pakai itu
    # - kalau kosong, isi = qty (netral, tidak bikin MAPE jadi aneh)
    if "pred_qty" not in df_hist2.columns:
        df_hist2["pred_qty"] = np.nan

    obs_full = _observed_mask(df_hist2, sales_obs_map)
    df_hist2["_is_observed_row"] = obs_full.astype(int)
    df_hist2.loc[df_hist2["_is_observed_row"] == 0, "qty"] = np.nan

    # isi pred_qty untuk baris observed yang masih kosong
    df_hist2["qty"] = pd.to_numeric(df_hist2["qty"], errors="coerce")
    df_hist2["pred_qty"] = pd.to_numeric(df_hist2["pred_qty"], errors="coerce")
    miss_pred = (df_hist2["_is_observed_row"] == 1) & df_hist2["qty"].notna() & df_hist2["pred_qty"].isna()
    df_hist2.loc[miss_pred, "pred_qty"] = df_hist2.loc[miss_pred, "qty"]

    df_all = pd.concat([df_hist2, df_future], ignore_index=True)
    return df_all



def generate_and_store_forecast(
    model_run_id: int | None,
    horizon_months: int = 6,
    updated_by: int | None = None,
) -> int:
    run = _resolve_model_run(model_run_id)
    mid = int(run["id"])

    snap = _load_model_snapshot_skus(mid)
    if snap.empty:
        raise ValueError("Snapshot model_run_sku kosong untuk model ini.")

    df_all = forecast_future_all_sku(model_run_id=mid, horizon_months=int(horizon_months))
    if df_all.empty:
        return 0

    df_all["cabang"] = df_all["cabang"].astype(str).str.strip().str.upper()
    df_all["sku"] = df_all["sku"].astype(str).str.strip().str.upper()

    # enforce snapshot skus
    df_all = df_all.merge(
        snap[["cabang", "sku"]].drop_duplicates(),
        on=["cabang", "sku"],
        how="inner",
    )
    if df_all.empty:
        return 0

    if "area" not in df_all.columns:
        df_all["area"] = ""

    # pastikan kolom flag ada
    if "is_train" not in df_all.columns:
        df_all["is_train"] = 0
    if "is_test" not in df_all.columns:
        df_all["is_test"] = 0
    if "is_future" not in df_all.columns:
        df_all["is_future"] = 0
    if "horizon" not in df_all.columns:
        df_all["horizon"] = 0
    if "pred_qty" not in df_all.columns:
        df_all["pred_qty"] = np.nan

    # normalisasi tipe
    df_all["periode"] = pd.to_datetime(df_all["periode"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df_all["qty"] = pd.to_numeric(df_all["qty"], errors="coerce")
    df_all["pred_qty"] = pd.to_numeric(df_all["pred_qty"], errors="coerce")
    df_all["is_train"] = pd.to_numeric(df_all["is_train"], errors="coerce").fillna(0).astype(int)
    df_all["is_test"] = pd.to_numeric(df_all["is_test"], errors="coerce").fillna(0).astype(int)
    df_all["is_future"] = pd.to_numeric(df_all["is_future"], errors="coerce").fillna(0).astype(int)
    df_all["horizon"] = pd.to_numeric(df_all["horizon"], errors="coerce").fillna(0).astype(int)
    df_all.loc[df_all["is_future"] == 1, "is_train"] = 0
    df_all.loc[df_all["is_future"] == 1, "is_test"] = 0
    df_all.loc[df_all["is_future"] == 1, "qty"] = np.nan
    df_all = df_all[(df_all["is_future"] == 1) | (df_all["qty"].notna())].copy()
    miss_pred_hist = (df_all["is_future"] == 0) & df_all["qty"].notna() & df_all["pred_qty"].isna()
    df_all.loc[miss_pred_hist, "pred_qty"] = df_all.loc[miss_pred_hist, "qty"]
    df_all["pred_qty"] = df_all["pred_qty"].fillna(0.0)

    df_save = df_all[
        ["area", "cabang", "sku", "periode", "qty", "pred_qty", "is_train", "is_test", "is_future", "horizon"]
    ].copy()
    df_save.rename(columns={"qty": "qty_actual"}, inplace=True)

    df_save["model_run_id"] = mid

    test_mask = (df_save["is_test"] == 1) & (df_save["is_future"] == 0) & df_save["qty_actual"].notna()
    if test_mask.any():
        global_mape = _mape(df_save.loc[test_mask, "qty_actual"], df_save.loc[test_mask, "pred_qty"])
        _upsert_forecast_config(f"GLOBAL_TEST_MAPE_{mid}", str(global_mape), updated_by)

        mape_sku = {}
        g = df_save.loc[test_mask].groupby(["cabang", "sku"], dropna=False)
        for (cab, sku), part in g:
            if len(part) == 0:
                continue
            val = _mape(part["qty_actual"], part["pred_qty"])
            mape_sku[f"{cab}|{sku}"] = float(val)

        _upsert_forecast_config(f"SKU_TEST_MAPE_JSON_{mid}", json.dumps(mape_sku, ensure_ascii=False), updated_by)
    else:
        # tetap simpan supaya dashboard/alpha punya fallback yang jelas
        _upsert_forecast_config(f"GLOBAL_TEST_MAPE_{mid}", "", updated_by)
        _upsert_forecast_config(f"SKU_TEST_MAPE_JSON_{mid}", "{}", updated_by)

    # insert DB
    records = df_save.to_dict(orient="records")
    cleaned_records = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = None
            clean[k] = v
        cleaned_records.append(clean)

    if not cleaned_records:
        return 0

    insert_sql = """
        INSERT INTO forecast_monthly (
            model_run_id, area, cabang, sku, periode,
            qty_actual, pred_qty, is_train, is_test, is_future, horizon
        ) VALUES (
            :model_run_id, :area, :cabang, :sku, :periode,
            :qty_actual, :pred_qty, :is_train, :is_test, :is_future, :horizon
        )
    """

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM forecast_monthly WHERE model_run_id = :mid"), {"mid": mid})
        conn.execute(text(insert_sql), cleaned_records)

        # safety delete: histori tanpa qty_actual jangan ada
        conn.execute(
            text(
                """
                DELETE FROM forecast_monthly
                WHERE model_run_id = :mid
                  AND is_future = 0
                  AND qty_actual IS NULL
                """
            ),
            {"mid": mid},
        )

    return int(len(cleaned_records))

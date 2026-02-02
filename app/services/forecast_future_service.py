import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import text
from app.db import engine
from app.inference.predict_cluster_pipeline import load_cluster_models
from app.services.model_service import get_all_model_runs


def _get_active_model_run() -> Dict:
    runs = get_all_model_runs()
    if not runs:
        return None

    active = [r for r in runs if r.get("active_flag") == 1]
    if active:
        active_sorted = sorted(active, key=lambda r: r.get("trained_at") or "")
        return active_sorted[-1]

    runs_sorted = sorted(runs, key=lambda r: r.get("trained_at") or "")
    return runs_sorted[-1]


def _safe_load_json(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return None

def _build_future_exog(cabang: str, periode: pd.Timestamp) -> Dict:

    sql = """
        SELECT event_flag, holiday_count, rainfall
        FROM external_data
        WHERE cabang = :cabang
          AND periode = :periode
        LIMIT 1
    """
    with engine.begin() as conn:
        row = conn.execute(
            text(sql),
            {"cabang": cabang, "periode": periode.date()},
        ).fetchone()

    if row:
        return {
            "event_flag": row[0],
            "holiday_count": row[1],
            "rainfall": row[2],
        }

    return {
        "event_flag": 0,
        "holiday_count": 0,
        "rainfall": 0.0,
    }


def _split_dynamic_static_features(feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    dynamic_keywords = [
        "qty_lag",
        "qty_rollmean",
        "qty_rollstd",
        "month",
        "year",
        "qtr",
        "event_flag",
        "holiday_count",
        "rainfall",
    ]

    dynamic_cols = []
    static_cols = []

    for c in feature_cols:
        if any(kw in c for kw in dynamic_keywords):
            dynamic_cols.append(c)
        else:
            static_cols.append(c)

    return dynamic_cols, static_cols

def _forecast_series_future(
    hist: pd.DataFrame,
    model,
    feature_cols: List[str],
    horizon: int,
) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame()

    # sort by periode
    hist = hist.sort_values("periode").reset_index(drop=True)

    # qty_wins fallback kalau tidak ada
    if "qty_wins" in hist.columns:
        qty_hist = hist["qty_wins"].astype(float).tolist()
    else:
        qty_hist = hist["qty"].astype(float).tolist()

    # exog history (kalau ada)
    event_hist = hist["event_flag"].astype(float).tolist() if "event_flag" in hist.columns else [0.0] * len(hist)
    hol_hist = hist["holiday_count"].astype(float).tolist() if "holiday_count" in hist.columns else [0.0] * len(hist)
    rain_hist = hist["rainfall"].astype(float).tolist() if "rainfall" in hist.columns else [0.0] * len(hist)

    # dynamic vs static features
    dynamic_cols, static_cols = _split_dynamic_static_features(feature_cols)

    last_row = hist.iloc[-1]

    # static features diambil dari last_row
    static_vals = {}
    for c in static_cols:
        if c in hist.columns:
            static_vals[c] = last_row[c]
        else:
            static_vals[c] = 0  # default aman

    area_val = last_row.get("area", None)
    cabang_val = last_row["cabang"]
    sku_val = last_row["sku"]

    future_rows = []

    for step in range(horizon):
        last_date = hist["periode"].max()
        next_period = (last_date + pd.offsets.MonthBegin(1)).normalize()

        # ambil exog future
        exg = _build_future_exog(cabang_val, next_period)
        event_future = exg["event_flag"]
        hol_future = exg["holiday_count"]
        rain_future = exg["rainfall"]

        # build feature dict
        feat = {}

        # static features
        for c in static_cols:
            feat[c] = static_vals.get(c, 0)

        # time features
        if "month" in dynamic_cols:
            feat["month"] = next_period.month
        if "year" in dynamic_cols:
            feat["year"] = next_period.year
        if "qtr" in dynamic_cols:
            feat["qtr"] = next_period.quarter

        # exog + lags
        # lag1 dari history (pakai nilai terakhir)
        last_event = event_hist[-1] if event_hist else 0
        last_hol = hol_hist[-1] if hol_hist else 0
        last_rain = rain_hist[-1] if rain_hist else 0.0

        if "event_flag" in dynamic_cols:
            feat["event_flag"] = event_future
        if "event_flag_lag1" in dynamic_cols:
            feat["event_flag_lag1"] = last_event

        if "holiday_count" in dynamic_cols:
            feat["holiday_count"] = hol_future
        if "holiday_count_lag1" in dynamic_cols:
            feat["holiday_count_lag1"] = last_hol

        if "rainfall" in dynamic_cols:
            feat["rainfall"] = rain_future
        if "rainfall_lag1" in dynamic_cols:
            feat["rainfall_lag1"] = last_rain

        # qty lags
        for i in range(1, 13):
            col = f"qty_lag{i}"
            if col in dynamic_cols:
                if len(qty_hist) >= i:
                    feat[col] = qty_hist[-i]
                else:
                    feat[col] = 0.0

        # rolling windows (3)
        if any(col.startswith("qty_roll") for col in dynamic_cols):
            arr = np.array(qty_hist, dtype=float)

            def _last_k(arr_, k):
                if len(arr_) == 0:
                    return np.array([])
                if len(arr_) <= k:
                    return arr_
                return arr_[-k:]

            if "qty_rollmean_3" in dynamic_cols:
                last3 = _last_k(arr, 3)
                feat["qty_rollmean_3"] = float(last3.mean()) if len(last3) > 0 else 0.0
            if "qty_rollstd_3" in dynamic_cols:
                last3 = _last_k(arr, 3)
                feat["qty_rollstd_3"] = float(last3.std(ddof=0)) if len(last3) > 0 else 0.0

            if "qty_rollmean_6" in dynamic_cols:
                last6 = _last_k(arr, 6)
                feat["qty_rollmean_6"] = float(last6.mean()) if len(last6) > 0 else 0.0
            if "qty_rollstd_6" in dynamic_cols:
                last6 = _last_k(arr, 6)
                feat["qty_rollstd_6"] = float(last6.std(ddof=0)) if len(last6) > 0 else 0.0

        for c in feature_cols:
            if c not in feat:
                if c in hist.columns:
                    feat[c] = last_row[c]
                else:
                    feat[c] = 0

        X = pd.DataFrame([feat], columns=feature_cols)

        # model training pakai log1p(qty_wins) → balik ke level qty
        pred_log = model.predict(X)
        pred_qty = float(np.expm1(pred_log[0]))

        # update histories
        qty_hist.append(pred_qty)
        event_hist.append(event_future)
        hol_hist.append(hol_future)
        rain_hist.append(rain_future)

        # simpan row untuk output
        future_rows.append(
            {
                "area": area_val,
                "cabang": cabang_val,
                "sku": sku_val,
                "periode": next_period,
                "pred_qty": pred_qty,
            }
        )

        # tambahkan ke hist untuk next loop (supaya tanggal max naik)
        hist = pd.concat(
            [
                hist,
                pd.DataFrame(
                    {
                        "periode": [next_period],
                        "cabang": [cabang_val],
                        "sku": [sku_val],
                    }
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(future_rows)

def _save_future_to_db(df_future: pd.DataFrame, model_run_id: int):
    if df_future.empty:
        return

    # hapus dulu forecast lama untuk model_run_id ini (biar tidak double)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM forecast_future WHERE model_run_id = :mid"),
            {"mid": model_run_id},
        )

        sql = """
            INSERT INTO forecast_future (
                model_run_id,
                area,
                cabang,
                sku,
                periode,
                pred_qty
            ) VALUES (
                :model_run_id,
                :area,
                :cabang,
                :sku,
                :periode,
                :pred_qty
            )
        """

        rows = df_future.to_dict(orient="records")
        for r in rows:
            conn.execute(
                text(sql),
                {
                    "model_run_id": model_run_id,
                    "area": r.get("area", ""),
                    "cabang": r["cabang"],
                    "sku": r["sku"],
                    "periode": r["periode"].date(),
                    "pred_qty": float(r["pred_qty"]),
                },
            )


def build_future_panel_and_forecast_all(
    engine,
    train_end: pd.Timestamp,
    horizon: int,
    run_dir: Path,
    feature_cols: List[str],
) -> pd.DataFrame:

    active = _get_active_model_run()
    if active is None:
        raise RuntimeError("Tidak ada model_run di DB.")

    model_run_id = active["id"]

    params = _safe_load_json(active.get("params_json"))
    panel_path = None

    if isinstance(params, dict):
        if params.get("panel_path"):
            panel_path = Path(params["panel_path"])
        elif params.get("run_dir"):
            run_dir = Path(params["run_dir"])
            panel_path = run_dir / "panel_with_predictions.csv"

    if panel_path is None:
        panel_path = Path(run_dir) / "panel_with_predictions.csv"

    if not panel_path.exists():
        raise FileNotFoundError(
            f"panel_with_predictions.csv tidak ditemukan di {panel_path}"
        )

    df_panel = pd.read_csv(panel_path, parse_dates=["periode"])

    required_cols = ["cabang", "sku", "periode", "qty", "pred_qty", "cluster"]
    missing = [c for c in required_cols if c not in df_panel.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib hilang di panel_with_predictions.csv: {missing}"
        )

    # Ambil hanya SKU yang ikut training window (≤ train_end)
    df_hist = df_panel[df_panel["periode"] <= train_end].copy()
    if df_hist.empty:
        raise ValueError(
            "Tidak ada history ≤ train_end di panel_with_predictions.csv."
        )

    # List unik series (eligible) dari panel
    series_list = (
        df_hist[["area", "cabang", "sku", "cluster"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Load model per cluster dari folder models/
    model_dir = Path(run_dir) / "models"
    cluster_ids = sorted(series_list["cluster"].dropna().unique())
    model_paths = {}
    for cid in cluster_ids:
        path = model_dir / f"lgbm_full_cluster_{int(cid)}.txt"
        if path.exists():
            model_paths[int(cid)] = path

    if not model_paths:
        raise RuntimeError(
            f"Tidak menemukan file model di {model_dir}. Cek hasil training."
        )

    models = load_cluster_models(model_paths)

    # Forecast untuk semua series
    future_rows_all = []

    for _, row in series_list.iterrows():
        area = row.get("area")
        cabang = row["cabang"]
        sku = row["sku"]
        cid = int(row["cluster"])

        model = models.get(cid)
        if model is None:
            # skip kalau model cluster tidak ada
            continue

        hist_cs = df_hist[
            (df_hist["cabang"] == cabang) & (df_hist["sku"] == sku)
        ].copy()

        if hist_cs.empty:
            continue

        try:
            df_future_cs = _forecast_series_future(
                hist=hist_cs,
                model=model,
                feature_cols=feature_cols,
                horizon=horizon,
            )
        except Exception as e:
            # supaya 1 SKU error tidak menjatuhkan semuanya
            print(f"[WARN] Gagal forecast future untuk {cabang}-{sku}: {e}")
            continue

        if df_future_cs.empty:
            continue

        # isi area kalau None
        if area is not None:
            df_future_cs["area"] = area
        else:
            df_future_cs["area"] = hist_cs["area"].iloc[0]

        future_rows_all.append(df_future_cs)

    if not future_rows_all:
        return pd.DataFrame(columns=["area", "cabang", "sku", "periode", "pred_qty"])

    df_future = pd.concat(future_rows_all, ignore_index=True)
    df_future = df_future.sort_values(["periode", "cabang", "sku"])

    # Simpan ke DB
    _save_future_to_db(df_future, model_run_id=model_run_id)

    return df_future

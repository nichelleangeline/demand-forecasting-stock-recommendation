# app/services/forecast_engine_future.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import text

from app.features.hierarchy_features import add_hierarchy_features
from app.features.stabilizer_features import add_stabilizer_features
from app.features.outlier_handler import winsorize_outliers
from app.features.time_features import add_all_lags_and_rollings

from app.services.history_service import load_history
from app.services.model_service import get_active_model_run
from app.inference.predict_cluster_pipeline import load_cluster_models

# ============================================
# HELPER: ambil cluster dari panel training
# ============================================

def _get_cluster_for_sku_from_panel(
    run: dict,
    cabang: str,
    sku: str,
) -> int:
    """
    Ambil cluster SKU dari panel training (panel_with_predictions.csv)
    supaya konsisten dengan training. Tidak pakai clustering ulang.
    """

    params_raw = run.get("params_json")
    if isinstance(params_raw, str):
        try:
            params = json.loads(params_raw)
        except Exception:
            params = {}
    elif isinstance(params_raw, dict):
        params = params_raw
    else:
        params = {}

    panel_path: Optional[Path] = None

    if "panel_path" in params:
        panel_path = Path(params["panel_path"])
    elif "run_dir" in params:
        panel_path = Path(params["run_dir"]) / "panel_with_predictions.csv"
    else:
        raise ValueError(
            "params_json pada model_run tidak punya panel_path / run_dir. "
            "Pastikan Tab 3 training versi terbaru yang menyimpan panel."
        )

    if not panel_path.exists():
        raise FileNotFoundError(
            f"panel_with_predictions.csv tidak ditemukan di {panel_path}"
        )

    df_panel = pd.read_csv(panel_path, usecols=["cabang", "sku", "cluster"])

    mask = (df_panel["cabang"] == cabang) & (df_panel["sku"] == sku)
    sub = df_panel.loc[mask]

    if sub.empty:
        raise ValueError(
            f"SKU {cabang}-{sku} tidak ada di panel training. "
            "Kemungkinan SKU ini tidak eligible saat training atau benar-benar baru."
        )

    cid = sub["cluster"].iloc[0]
    if pd.isna(cid):
        raise ValueError(
            f"Cluster untuk {cabang}-{sku} kosong di panel training."
        )

    return int(cid)


# ============================================
# HELPER: FE seperti training
# ============================================
def _apply_full_FE(df: pd.DataFrame) -> pd.DataFrame:
    """
    FE harus sama seperti Tab 3 training.

    Di sini df sudah punya:
      - area, cabang, sku, periode, qty, event_flag, holiday_count, rainfall
      - is_train (untuk stabilizer_features)
      - cluster (supaya ikut dipakai sebagai fitur kalau ada)
    """

    df = df.copy()

    # 0) pastikan urut
    if {"cabang", "sku", "periode"}.issubset(df.columns):
        df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

    # 1) GLOBAL STATS PER (cabang, sku)
    #    qty_mean_cs, qty_std_cs, qty_cnt_cs, qty_cv_cs
    #    ini yang dibutuhkan stabilizer_features
    need_stats = any(
        c not in df.columns
        for c in ["qty_mean_cs", "qty_std_cs", "qty_cnt_cs", "qty_cv_cs"]
    )

    if need_stats and "qty" in df.columns:
        grp = df.groupby(["cabang", "sku"])["qty"]
        df["qty_mean_cs"] = grp.transform("mean")
        df["qty_std_cs"] = grp.transform("std").fillna(0.0)
        df["qty_cnt_cs"] = grp.transform("count")
        df["qty_cv_cs"] = df["qty_std_cs"] / (df["qty_mean_cs"] + 1e-9)

    # 2) hierarchy features (area/cabang/sku/family)
    df = add_hierarchy_features(df)

    # 3) family_idx kalau ada
    if "family" in df.columns:
        fam_map = {
            fam: idx
            for idx, fam in enumerate(sorted(df["family"].astype(str).unique()))
        }
        df["family_idx"] = df["family"].astype(str).map(fam_map).astype("int16")

    # 4) stabilizer flags (butuh is_train + qty_*_cs)
    df = add_stabilizer_features(df)

    # 5) winsorize qty → qty_wins
    df = winsorize_outliers(df)

    # 6) log features (seperti training)
    df["log_qty"] = np.log1p(df["qty"])
    df["log_qty_wins"] = np.log1p(df["qty_wins"])

    # 7) time-based lags & rolling
    df = add_all_lags_and_rollings(df)

    return df


def _infer_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_cols = [
        "area",
        "cabang",
        "sku",
        "periode",
        "qty",
        "qty_wins",
        "log_qty",
        "log_qty_wins",
        "is_train",
        "is_test",
        "sample_weight",
        "family",
    ]
    return [c for c in df.columns if c not in drop_cols]


# ============================================
# FORECAST 1 SKU (multi-step)
# ============================================

def forecast_future_by_sku(
    engine,
    cabang: str,
    sku: str,
    horizon: int = 6,
    feature_cols: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Forecast ke depan untuk 1 (cabang, sku) selama 'horizon' bulan.

    kwargs dipakai untuk kompatibilitas dengan pemanggilan lama
    (misal: models=..., run_dir=..., dsb) dan diabaikan di sini.
    """

    # 1. ambil run aktif + lokasi model
    run = get_active_model_run()
    if run is None:
        raise ValueError("Tidak ada model_run aktif.")

    # cluster untuk SKU ini dari panel training
    cid = _get_cluster_for_sku_from_panel(run, cabang, sku)

    # ambil params_json → run_dir
    params_raw = run.get("params_json")
    if isinstance(params_raw, str):
        params = json.loads(params_raw)
    elif isinstance(params_raw, dict):
        params = params_raw
    else:
        params = {}

    if "run_dir" in params:
        run_dir = Path(params["run_dir"])
    elif "panel_path" in params:
        run_dir = Path(params["panel_path"]).parent
    else:
        raise ValueError(
            "params_json tidak punya run_dir/panel_path, tidak tahu lokasi model."
        )

    model_dir = run_dir / "models"
    model_path = model_dir / f"lgbm_full_cluster_{cid}.txt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model cluster {cid} tidak ditemukan: {model_path}"
        )

    # load model cluster
    models = load_cluster_models({cid: model_path})
    model: lgb.Booster = models[cid]

    # 2. load history (qty + exog) dari DB
    hist = load_history(engine, cabang, sku)
    if hist.empty:
        raise ValueError(f"Tidak ada history untuk {cabang}-{sku}.")

    # sort & flag train
    hist = hist.sort_values("periode").reset_index(drop=True)
    hist["is_train"] = 1
    hist["cluster"] = cid

    # 3. FE + lags/rolling pada seluruh history
    df_hist = _apply_full_FE(hist)

    # feature cols: pakai dari training kalau dikasih, kalau tidak infer
    if feature_cols is None:
        feature_cols = _infer_feature_cols(df_hist)

    # 4. loop multi-step
    df_loop = df_hist.copy()
    preds = []

    for step in range(horizon):
        last_period = df_loop["periode"].max()
        next_period = (last_period + pd.offsets.MonthBegin(1)).normalize()

        # ambil exog future
        sql_exog = text(
            """
            SELECT event_flag, holiday_count, rainfall
            FROM external_data
            WHERE cabang = :cabang
              AND periode = :periode
            LIMIT 1
            """
        )
        with engine.begin() as conn:
            row = conn.execute(
                sql_exog,
                {"cabang": cabang, "periode": next_period},
            ).fetchone()

        if row:
            event_flag, holiday_count, rainfall = row
        else:
            event_flag, holiday_count, rainfall = 0, 0, 0.0

        # row future mentah
        f = pd.DataFrame(
            [
                {
                    "area": df_loop["area"].iloc[0],
                    "cabang": cabang,
                    "sku": sku,
                    "periode": next_period,
                    "qty": 0.0,
                    "event_flag": event_flag,
                    "holiday_count": holiday_count,
                    "rainfall": rainfall,
                    "is_train": 0,
                    "cluster": cid,
                }
            ]
        )

        # gabung ke df_loop, FE + lags ulang (biar lag pakai pred step sebelumnya)
        tmp = pd.concat([df_loop, f], ignore_index=True)
        tmp = tmp.sort_values("periode").reset_index(drop=True)

        tmp = _apply_full_FE(tmp)

        # ambil baris periode next_period setelah FE
        f2 = tmp[tmp["periode"] == next_period].copy()
        X = f2[feature_cols]

        pred_log = model.predict(X)
        pred_qty = np.expm1(pred_log)[0]

        f2["pred_qty"] = float(pred_qty)
        preds.append(f2)

        # update df_loop: isi qty untuk next_period pakai pred agar lag next step pakai angka ini
        tmp.loc[tmp["periode"] == next_period, "qty"] = pred_qty
        df_loop = tmp.copy()

    df_future = pd.concat(preds, ignore_index=True)
    return df_future


# ============================================
# SIMPAN KE DB (opsional)
# ============================================

def save_future_to_db(engine, df_future: pd.DataFrame, model_run_id: int):
    """
    Simpan hasil forecast future ke tabel forecast_future.
    """

    sql = text(
        """
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
    )

    rows = df_future[["area", "cabang", "sku", "periode", "pred_qty"]].copy()

    with engine.begin() as conn:
        for _, r in rows.iterrows():
            conn.execute(
                sql,
                {
                    "model_run_id": model_run_id,
                    "area": r["area"],
                    "cabang": r["cabang"],
                    "sku": r["sku"],
                    "periode": r["periode"].date(),
                    "pred_qty": float(r["pred_qty"]),
                },
            )

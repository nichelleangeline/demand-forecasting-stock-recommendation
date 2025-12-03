# app/pipeline/lgbm_global_pipeline.py

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import lightgbm as lgb

from sqlalchemy import text

from app.db import engine

# profiling & clustering
from app.profiling.sku_profiler import build_sku_profile
from app.profiling.clustering import run_sku_clustering

# feature modules
from app.features.hierarchy_features import add_hierarchy_features
from app.features.stabilizer_features import add_stabilizer_features
from app.features.outlier_handler import winsorize_outliers

# trainer cluster LGBM (tweedie, optuna, log1p)
from app.modeling.lgbm_trainer_cluster import train_lgbm_per_cluster


# =========================================
# METRIC FUNCTIONS
# =========================================
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0


# =========================================
# PATH MODEL OUTPUT (UNTUK SISTEM)
# =========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT_ROOT   = PROJECT_ROOT / "outputs" / "lgbm_global_clusters_tweedie_noleak"
MODEL_DIR  = OUT_ROOT / "models"
METRIC_DIR = OUT_ROOT / "metrics"

for d in [OUT_ROOT, MODEL_DIR, METRIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================
# STEP 1: LOAD PANEL GLOBAL DARI DB
# =========================================

def load_panel_from_db() -> pd.DataFrame:
    """
    Ambil data bulanan dari DB dan join ke external_data.

    Asumsi tabel:
    - sales_monthly(area, cabang, sku, periode, qty, ...)
    - external_data(area, cabang, periode, event_flag, holiday_count, rainfall, ...)
    """
    sql = """
    SELECT
        s.area,
        s.cabang,
        s.sku,
        s.periode,
        s.qty,
        COALESCE(e.event_flag, 0)     AS event_flag,
        COALESCE(e.holiday_count, 0)  AS holiday_count,
        COALESCE(e.rainfall, 0)       AS rainfall
    FROM sales_monthly s
    LEFT JOIN external_data e
        ON s.area   = e.area
       AND s.cabang = e.cabang
       AND s.periode = e.periode
    """
    df = pd.read_sql(text(sql), engine, parse_dates=["periode"])
    df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

    # jaga-jaga tipe
    df["qty"] = df["qty"].astype(float)
    df["event_flag"] = df["event_flag"].astype(int)
    df["holiday_count"] = df["holiday_count"].astype(int)
    df["rainfall"] = df["rainfall"].astype(float)

    return df


# =========================================
# STEP 2: BUAT FLAG TRAIN / TEST DARI CONFIG
# =========================================

def add_train_test_flags(df: pd.DataFrame, test_months: int, min_train_months: int = 24) -> pd.DataFrame:
    """
    train = semua bulan selain N bulan terakhir
    test  = N bulan terakhir (global, semua cabang/sku)
    """
    df = df.copy()
    df["month_key"] = df["periode"].dt.to_period("M")

    months = sorted(df["month_key"].unique())
    if len(months) < test_months + min_train_months:
        raise ValueError(
            f"Data terlalu pendek. Punya {len(months)} bulan, "
            f"butuh minimal {min_train_months} + {test_months}"
        )

    test_months_set = set(months[-test_months:])

    df["is_test"] = df["month_key"].isin(test_months_set).astype(int)
    df["is_train"] = 1 - df["is_test"]

    df = df.drop(columns=["month_key"])
    return df


# =========================================
# STEP 3: BASE FEATURE ENGINEERING (GLOBAL)
# =========================================

def add_base_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah:
    - lag qty 1..12
    - rolling mean/std 3, 6, 12
    - lag exog (event_flag_lag1, holiday_count_lag1, rainfall_lag1)
    - calendar (month, year, qtr)
    - dummy imputed, spike_flag, sample_weight
      (kalau nanti ada logic benerannya, tinggal ganti di sini)
    """
    df = df.copy()
    df = df.sort_values(["cabang", "sku", "periode"])

    keys = ["cabang", "sku"]

    # calendar
    df["month"] = df["periode"].dt.month.astype("int16")
    df["year"] = df["periode"].dt.year.astype("int16")
    df["qtr"] = df["periode"].dt.quarter.astype("int16")

    # exog lag
    df["event_flag_lag1"] = (
        df.groupby(keys)["event_flag"].shift(1).fillna(0).astype(int)
    )
    df["holiday_count_lag1"] = (
        df.groupby(keys)["holiday_count"].shift(1).fillna(0).astype(int)
    )
    df["rainfall_lag1"] = (
        df.groupby(keys)["rainfall"].shift(1).fillna(0.0).astype(float)
    )

    # qty lags
    for l in range(1, 13):
        df[f"qty_lag{l}"] = (
            df.groupby(keys)["qty"].shift(l).astype(float)
        )

    # rolling stats
    df["qty_rollmean_3"] = (
        df.groupby(keys)["qty"].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    )
    df["qty_rollstd_3"] = (
        df.groupby(keys)["qty"].rolling(3, min_periods=1).std().reset_index(level=[0,1], drop=True)
    )
    df["qty_rollmean_6"] = (
        df.groupby(keys)["qty"].rolling(6, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    )
    df["qty_rollstd_6"] = (
        df.groupby(keys)["qty"].rolling(6, min_periods=1).std().reset_index(level=[0,1], drop=True)
    )
    df["qty_rollmean_12"] = (
        df.groupby(keys)["qty"].rolling(12, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    )
    df["qty_rollstd_12"] = (
        df.groupby(keys)["qty"].rolling(12, min_periods=1).std().reset_index(level=[0,1], drop=True)
    )

    # fill std NaN saat window kecil
    for c in ["qty_rollstd_3", "qty_rollstd_6", "qty_rollstd_12"]:
        df[c] = df[c].fillna(0.0)

    # placeholder flags
    df["imputed"] = 0
    df["spike_flag"] = 0
    df["sample_weight"] = 1.0

    return df


# =========================================
# STEP 4: FULL PIPELINE TRAINING GLOBAL
# =========================================

def run_global_lgbm_training(
    test_months: int = 4,
    min_train_months: int = 24,
) -> Dict[int, str]:
    """
    Pipeline training global LGBM dari DB.

    Return dict: {cluster_id: path_model}
    """
    print("====================================")
    print("RUN GLOBAL LGBM TRAINING (NO LEAK, FROM DB)")
    print("====================================")

    # 1) Load panel dari DB
    print("[STEP 1] Load panel dari DB ...")
    df = load_panel_from_db()
    print("Rows:", len(df))

    # 2) Tambah train / test flags
    print(f"[STEP 2] Tambah flag is_train/is_test (test {test_months} bulan terakhir)...")
    df = add_train_test_flags(df, test_months=test_months, min_train_months=min_train_months)

    # 3) Base features
    print("[STEP 3] Base TS features (lag, rolling, calendar, exog lag)...")
    df = add_base_ts_features(df)

    # 4) Build profile dari TRAIN
    print("[STEP 4] Build SKU profile dari TRAIN...")
    df_train = df[df["is_train"] == 1].copy()
    profile = build_sku_profile(df_train)

    profile_path = OUT_ROOT / "cluster_profiles_raw_train_only.csv"
    profile.to_csv(profile_path, index=False)
    print("Saved raw train profile to:", profile_path)

    # 5) Clustering (TRAIN only)
    print("[STEP 5] Clustering SKU (TRAIN only, 4 cluster)...")
    profile_clustered = run_sku_clustering(profile, n_clusters=4)

    profile_cluster_path = OUT_ROOT / "cluster_profiles_full_train_only.csv"
    profile_clustered.to_csv(profile_cluster_path, index=False)
    print("Saved clustered profile to:", profile_cluster_path)

    print("Cluster summary:")
    print(
        profile_clustered.groupby("cluster")[["qty_mean", "cv", "zero_ratio", "total_qty"]]
                         .mean()
                         .round(2)
                         .to_string()
    )

    # 6) Merge cluster + demand_level ke panel
    print("[STEP 6] Merge cluster & demand_level ke panel (train+test)...")
    df = df.merge(
        profile_clustered[["cabang", "sku", "cluster", "demand_level"]],
        on=["cabang", "sku"],
        how="left",
    )
    df["cluster"] = df["cluster"].fillna(-1).astype(int)

    # 7) Hierarchy features (family)
    print("[STEP 7] Tambah hierarchy features (family)...")
    df = add_hierarchy_features(df)

    if "family" in df.columns:
        fam_map = {fam: idx for idx, fam in enumerate(sorted(df["family"].astype(str).unique()))}
        df["family_idx"] = df["family"].astype(str).map(fam_map).astype("int16")
        print("Family mapping:", fam_map)

    # 8) Stabilizer features (no leak, pakai stats TRAIN)
    print("[STEP 8] Tambah stabilizer features (qty_mean_cs, volatility, seasonal_ratio_12)...")
    df = add_stabilizer_features(df)

    # 9) Outlier treatment per SKU (no leak)
    print("[STEP 9] Winsorize outliers per SKU...")
    df = winsorize_outliers(df)

    # backup log1p
    df["log_qty"] = np.log1p(df["qty"])
    df["log_qty_wins"] = np.log1p(df["qty_wins"])

    df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

    # 10) Feature columns
    print("[STEP 10] Tentukan feature_cols...")
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
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print("Num features:", len(feature_cols))
    print("Contoh fitur:", feature_cols[:20])

    obj_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print("WARNING: Masih ada kolom object di feature_cols:", obj_cols)

    # 11) Training per cluster
    print("[STEP 11] Training LGBM per cluster (Tweedie, no leak, log1p)...")
    cluster_ids = sorted(df["cluster"].dropna().unique())
    models: Dict[int, lgb.Booster] = {}
    model_paths: Dict[int, str] = {}

    for cid in cluster_ids:
        if cid == -1:
            print(f"Cluster {cid} == -1 (unknown), skip.")
            continue

        print("\n====================================")
        print(f"TRAINING CLUSTER {cid}")
        print("====================================")

        model = train_lgbm_per_cluster(
            df=df,
            cluster_id=int(cid),
            feature_cols=feature_cols,
            log_target=True,
            n_trials=40,
        )

        if model is None:
            print(f"Cluster {cid}: model None, skip.")
            continue

        models[cid] = model
        model_path = MODEL_DIR / f"lgbm_global_cluster_{cid}.txt"
        model.save_model(str(model_path))
        model_paths[cid] = str(model_path)
        print(f"Cluster {cid}: model saved to {model_path}")

    if not models:
        raise RuntimeError("Tidak ada model yang berhasil dilatih.")

    # 12) Global metrics train/test
    print("[STEP 12] Global metrics (train/test)...")
    df_pred_list = []

    for cid, model in models.items():
        df_c = df[df["cluster"] == cid].copy()
        if df_c.empty:
            continue

        X_c = df_c[feature_cols]
        pred_log = model.predict(X_c)
        pred_qty = np.expm1(pred_log)

        df_c["pred_qty"] = pred_qty
        df_pred_list.append(df_c)

    df_pred = pd.concat(df_pred_list, axis=0).sort_index()

    metrics_global = []
    for split_name, mask in [
        ("train", df_pred["is_train"] == 1),
        ("test", df_pred["is_test"] == 1),
    ]:
        if not mask.any():
            continue

        yt = df_pred.loc[mask, "qty"].values
        yp = df_pred.loc[mask, "pred_qty"].values

        metrics_global.append({
            "split": split_name,
            "n_obs": int(len(yt)),
            "MSE": mse(yt, yp),
            "RMSE": rmse(yt, yp),
            "MAE": mae(yt, yp),
            "MAPE": mape(yt, yp),
            "sMAPE": smape(yt, yp),
        })

    global_df = pd.DataFrame(metrics_global)
    global_metric_path = METRIC_DIR / "global_metrics_clusters_tweedie_global_noleak.csv"
    global_df.to_csv(global_metric_path, index=False)
    print("Saved global metrics to:", global_metric_path)
    print(global_df.to_string(index=False))

    # 13) Metrics per series
    print("[STEP 13] Metrics per cabang–SKU...")
    rows = []

    for (cab, sku), g in df_pred.groupby(["cabang", "sku"], sort=False):
        g_tr = g[g["is_train"] == 1]
        g_te = g[g["is_test"] == 1]

        row = {
            "cabang": cab,
            "sku": sku,
            "cluster": g["cluster"].iloc[0],
            "n_train": int(len(g_tr)),
            "n_test": int(len(g_te)),
        }

        if len(g_tr) > 0:
            yt_tr = g_tr["qty"].values
            yp_tr = g_tr["pred_qty"].values
            row.update({
                "train_mae": mae(yt_tr, yp_tr),
                "train_mse": mse(yt_tr, yp_tr),
                "train_rmse": rmse(yt_tr, yp_tr),
                "train_mape": mape(yt_tr, yp_tr),
                "train_smape": smape(yt_tr, yp_tr),
            })
        else:
            row.update({
                "train_mae": np.nan,
                "train_mse": np.nan,
                "train_rmse": np.nan,
                "train_mape": np.nan,
                "train_smape": np.nan,
            })

        if len(g_te) > 0:
            yt_te = g_te["qty"].values
            yp_te = g_te["pred_qty"].values
            row.update({
                "test_mae": mae(yt_te, yp_te),
                "test_mse": mse(yt_te, yp_te),
                "test_rmse": rmse(yt_te, yp_te),
                "test_mape": mape(yt_te, yp_te),
                "test_smape": smape(yt_te, yp_te),
            })
        else:
            row.update({
                "test_mae": np.nan,
                "test_mse": np.nan,
                "test_rmse": np.nan,
                "test_mape": np.nan,
                "test_smape": np.nan,
            })

        rows.append(row)

    metrics_series = pd.DataFrame(rows)
    metrics_series["gap_RMSE"] = metrics_series["test_rmse"] - metrics_series["train_rmse"]
    metrics_series["ratio_RMSE"] = metrics_series["test_rmse"] / metrics_series["train_rmse"]

    series_metric_path = METRIC_DIR / "metrics_by_series_clusters_tweedie_global_noleak.csv"
    metrics_series.to_csv(series_metric_path, index=False)
    print("Saved metrics per series to:", series_metric_path)

    print("\nSELESAI: Global LGBM training dari DB selesai, model per cluster tersimpan.")
    return model_paths


if __name__ == "__main__":
    # contoh manual run
    run_global_lgbm_training(test_months=4, min_train_months=24)

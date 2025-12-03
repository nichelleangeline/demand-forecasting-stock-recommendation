# app/services/global_panel_service.py

from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import engine

from app.features.hierarchy_features import add_hierarchy_features
from app.features.stabilizer_features import add_stabilizer_features
from app.features.outlier_handler import winsorize_outliers
from app.profiling.sku_profiler import build_sku_profile
from app.profiling.clustering import run_sku_clustering


# =========================================
# 1. Ambil data bulanan dari DB
# =========================================

def fetch_monthly_panel_from_db(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    cabang_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ambil data dari:
    - sales_monthly
    - external_data
    dan join ke 1 panel: (area, cabang, sku, periode, qty, event_flag, holiday_count, rainfall).
    """

    query = """
        SELECT
            s.area,
            s.cabang,
            s.sku,
            s.periode,
            s.qty,
            COALESCE(e.event_flag, 0)      AS event_flag,
            COALESCE(e.holiday_count, 0)   AS holiday_count,
            COALESCE(e.rainfall, 0)        AS rainfall
        FROM sales_monthly s
        LEFT JOIN external_data e
          ON s.area   = e.area
         AND s.cabang = e.cabang
         AND s.periode = e.periode
        WHERE 1 = 1
    """

    params = {}

    if start_date is not None:
        query += " AND s.periode >= :start_date"
        params["start_date"] = start_date

    if end_date is not None:
        query += " AND s.periode <= :end_date"
        params["end_date"] = end_date

    if cabang_list:
        # bikin list param :c0, :c1, dst
        placeholders = []
        for i, cab in enumerate(cabang_list):
            key = f"c{i}"
            placeholders.append(f":{key}")
            params[key] = cab
        query += f" AND s.cabang IN ({', '.join(placeholders)})"

    with engine.connect() as conn:
        df = pd.read_sql(
            text(query),
            conn,
            params=params,
            parse_dates=["periode"],
        )

    if df.empty:
        return df

    # Normalisasi ke awal bulan (kalau user iseng isi tanggal 15)
    df["periode"] = pd.to_datetime(df["periode"])
    df["periode"] = df["periode"].dt.to_period("M").dt.to_timestamp()

    df["qty"] = df["qty"].astype(float)

    # Pastikan exog non-null
    for col in ["event_flag", "holiday_count", "rainfall"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    return df


# =========================================
# 2. Lengkapi grid bulanan per (area, cabang, sku)
# =========================================

def build_monthly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lengkapi semua bulan dari min(periode) sampai max(periode) per (area, cabang, sku).
    Missing qty diisi 0, dan tandai imputed.
    """

    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values(["area", "cabang", "sku", "periode"])

    global_start = df["periode"].min()
    global_end = df["periode"].max()

    all_months = pd.date_range(global_start, global_end, freq="MS")

    panels = []

    for (area, cabang, sku), g in df.groupby(["area", "cabang", "sku"], sort=False):
        base = pd.DataFrame({
            "area": area,
            "cabang": cabang,
            "sku": sku,
            "periode": all_months,
        })

        merged = base.merge(
            g,
            on=["area", "cabang", "sku", "periode"],
            how="left",
            suffixes=("", "_orig"),
        )

        # flag imputed kalau qty asli NaN
        merged["imputed"] = merged["qty"].isna().astype("int8")

        # qty kosong diisi 0
        merged["qty"] = merged["qty"].fillna(0.0)

        # exog kosong diisi 0
        for col in ["event_flag", "holiday_count", "rainfall"]:
            if col not in merged.columns:
                merged[col] = 0
            merged[col] = merged[col].fillna(0)

        # spike_flag awal 0, sample_weight awal 1
        merged["spike_flag"] = 0
        merged["sample_weight"] = 1.0

        panels.append(merged)

    out = pd.concat(panels, axis=0, ignore_index=True)
    out = out.sort_values(["area", "cabang", "sku", "periode"]).reset_index(drop=True)
    return out


# =========================================
# 3. Fitur dasar time series (lag, rolling, kalender)
# =========================================

def add_base_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah fitur:
    - event_flag_lag1, holiday_count_lag1, rainfall_lag1
    - qty_lag1..qty_lag12
    - qty_rollmean_3/6/12, qty_rollstd_3/6/12
    - month, year, qtr
    Semua rolling causal (hanya lihat masa lalu).
    """

    df = df.copy()
    df = df.sort_values(["cabang", "sku", "periode"])

    g = df.groupby(["cabang", "sku"], sort=False)

    # LAG EXOG
    df["event_flag_lag1"] = g["event_flag"].shift(1).fillna(0).astype(int)
    df["holiday_count_lag1"] = g["holiday_count"].shift(1).fillna(0).astype(int)
    df["rainfall_lag1"] = g["rainfall"].shift(1).fillna(0.0)

    # LAG QTY
    for lag in range(1, 13):
        df[f"qty_lag{lag}"] = g["qty"].shift(lag)

    # ROLLING
    df["qty_rollmean_3"] = g["qty"].rolling(window=3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df["qty_rollstd_3"] = g["qty"].rolling(window=3, min_periods=1).std().reset_index(level=[0, 1], drop=True)

    df["qty_rollmean_6"] = g["qty"].rolling(window=6, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df["qty_rollstd_6"] = g["qty"].rolling(window=6, min_periods=1).std().reset_index(level=[0, 1], drop=True)

    df["qty_rollmean_12"] = g["qty"].rolling(window=12, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df["qty_rollstd_12"] = g["qty"].rolling(window=12, min_periods=1).std().reset_index(level=[0, 1], drop=True)

    # Kalender
    df["month"] = df["periode"].dt.month.astype("int8")
    df["year"] = df["periode"].dt.year.astype("int16")
    df["qtr"] = df["periode"].dt.quarter.astype("int8")

    return df


# =========================================
# 4. Flag train / test
# =========================================

def add_train_test_flags(
    df: pd.DataFrame,
    test_months: int = 4,
) -> pd.DataFrame:
    """
    Bagi panel jadi train + test:
    - test = N bulan terakhir (test_months)
    - train = sisanya sebelum itu
    """

    df = df.copy()
    df = df.sort_values(["cabang", "sku", "periode"])

    max_per = df["periode"].max()
    # contoh: test_months=4, ambil 4 bulan terakhir
    cutoff_start = (max_per.to_period("M") - (test_months - 1)).to_timestamp()

    df["is_test"] = (df["periode"] >= cutoff_start).astype("int8")
    df["is_train"] = (df["periode"] < cutoff_start).astype("int8")

    return df


# =========================================
# 5. Pipeline global: DB -> panel siap LGBM
# =========================================

def build_lgbm_ready_panel_from_db(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    cabang_list: Optional[List[str]] = None,
    test_months: int = 4,
) -> pd.DataFrame:
    """
    Full pipeline global (tanpa train LGBM):
    1) Tarik data dari DB
    2) Lengkapi grid bulanan per SKU
    3) Tambah fitur dasar TS
    4) Flag train/test
    5) Profil + clustering (A)
    6) Stabilizer features (B)
    7) Outlier winsorization (C)
    8) Hierarchy / family (E)
    Return: panel final siap dikasih ke trainer LGBM.
    """

    # 1) tarik dari DB
    df = fetch_monthly_panel_from_db(
        start_date=start_date,
        end_date=end_date,
        cabang_list=cabang_list,
    )

    if df.empty:
        print("Panel dari DB kosong. Cek data.")
        return df

    # 2) lengkapi grid bulanan
    df = build_monthly_grid(df)

    # 3) fitur TS dasar
    df = add_base_ts_features(df)

    # 4) flag train / test
    df = add_train_test_flags(df, test_months=test_months)

    # 5) profiling + clustering pakai TRAIN saja
    df_train = df[df["is_train"] == 1].copy()
    profile = build_sku_profile(df_train)
    profile_clustered = run_sku_clustering(profile, n_clusters=4)

    # merge cluster + demand_level
    df = df.merge(
        profile_clustered[["cabang", "sku", "cluster", "demand_level"]],
        on=["cabang", "sku"],
        how="left",
    )
    df["cluster"] = df["cluster"].fillna(-1).astype(int)

    # 6) stabilizer features (no leak, pakai fungsi yang sudah ada)
    df = add_stabilizer_features(df)

    # 7) winsorize qty per SKU (no leak)
    df = winsorize_outliers(df)

    # backup log qty kalau mau dipakai
    df["log_qty"] = np.log1p(df["qty"])
    df["log_qty_wins"] = np.log1p(df["qty_wins"])

    # 8) hierarchy / family
    df = add_hierarchy_features(df)

    # encode family ke index numeric
    if "family" in df.columns:
        family_map = {
            fam: idx for idx, fam in enumerate(sorted(df["family"].astype(str).unique()))
        }
        df["family_idx"] = df["family"].astype(str).map(family_map).astype("int16")

    df = df.sort_values(["area", "cabang", "sku", "periode"]).reset_index(drop=True)
    return df

# app/profiling/sku_profiler.py

import pandas as pd
import numpy as np
from sqlalchemy import text

from app.db import engine


WIN_START_DEFAULT = pd.Timestamp("2021-01-01")


def build_sku_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versi yang dipakai offline LGBM:
    - groupby cabang, sku
    - hitung summary qty + zero_ratio + cv + demand_level
    """
    df = df.copy()
    df["qty"] = df["qty"].astype(float)

    profile = (
        df.groupby(["cabang", "sku"])
          .agg(
              n_months=("periode", "nunique"),
              qty_mean=("qty", "mean"),
              qty_std=("qty", "std"),
              qty_max=("qty", "max"),
              qty_min=("qty", "min"),
              total_qty=("qty", "sum"),
              zero_months=("qty", lambda x: (x == 0).sum()),
          )
          .reset_index()
    )

    profile["zero_ratio"] = profile["zero_months"] / profile["n_months"]
    profile["cv"] = profile["qty_std"] / profile["qty_mean"].replace(0, np.nan)

    # demand_level: quartile dari qty_mean (0..3)
    profile["demand_level"] = pd.qcut(
        profile["qty_mean"],
        q=4,
        labels=[0, 1, 2, 3]
    ).astype(int)

    return profile


def build_and_store_sku_profile(
    train_end: pd.Timestamp,
    win_start: pd.Timestamp | None = None,
) -> int:
    """
    Ambil data dari sales_monthly (window waktu),
    build profil SKU, lalu simpan ke tabel sku_profile.

    Dipakai Tab 1 (Build / Refresh SKU Profile).

    Return:
        jumlah baris yang ditulis ke sku_profile.
    """
    if win_start is None:
        win_start = WIN_START_DEFAULT

    # Pastikan tipe Timestamp
    train_end = pd.Timestamp(train_end)
    win_start = pd.Timestamp(win_start)

    # -------------------------------
    # 1. Ambil data dari sales_monthly
    # -------------------------------
    sql = text(
        """
        SELECT
            area,
            cabang,
            sku,
            periode,
            qty
        FROM sales_monthly
        WHERE periode >= :win_start
          AND periode <= :train_end
        ORDER BY cabang, sku, periode
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {
                "win_start": win_start,
                "train_end": train_end,
            },
        ).mappings().all()

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["periode"] = pd.to_datetime(df["periode"])
    df["qty"] = df["qty"].astype(float)

    # -------------------------------
    # 2. Build profil SKU (sama offline)
    # -------------------------------
    profile = build_sku_profile(df)

    # Tambah rule default eligible_model
    profile["nonzero_months"] = profile["n_months"] - profile["zero_months"]
    profile["eligible_model"] = (
        (profile["n_months"] >= 36)
        & (profile["nonzero_months"] >= 10)
        & (profile["total_qty"] >= 30)
        & (profile["zero_ratio"] <= 0.7)
    ).astype(int)

    # cluster default -1 (belum ada clustering)
    profile["cluster"] = -1

    # -------------------------------
    # 3. Simpan ke tabel sku_profile (full refresh)
    # -------------------------------
    insert_sql = text(
        """
        INSERT INTO sku_profile (
            cabang,
            sku,
            n_months,
            qty_mean,
            qty_std,
            qty_max,
            qty_min,
            total_qty,
            zero_months,
            zero_ratio,
            cv,
            demand_level,
            cluster,
            eligible_model
        )
        VALUES (
            :cabang,
            :sku,
            :n_months,
            :qty_mean,
            :qty_std,
            :qty_max,
            :qty_min,
            :total_qty,
            :zero_months,
            :zero_ratio,
            :cv,
            :demand_level,
            :cluster,
            :eligible_model
        )
        """
    )

    with engine.begin() as conn:
        # Full rebuild
        conn.execute(text("TRUNCATE TABLE sku_profile"))

        for _, row in profile.iterrows():
            conn.execute(
                insert_sql,
                {
                    "cabang": row["cabang"],
                    "sku": row["sku"],
                    "n_months": int(row["n_months"]),
                    "qty_mean": float(row["qty_mean"]),
                    "qty_std": float(row["qty_std"]) if not pd.isna(row["qty_std"]) else 0.0,
                    "qty_max": float(row["qty_max"]),
                    "qty_min": float(row["qty_min"]),
                    "total_qty": float(row["total_qty"]),
                    "zero_months": int(row["zero_months"]),
                    "zero_ratio": float(row["zero_ratio"]),
                    "cv": float(row["cv"]) if not pd.isna(row["cv"]) else 0.0,
                    "demand_level": int(row["demand_level"]),
                    "cluster": int(row["cluster"]),
                    "eligible_model": int(row["eligible_model"]),
                },
            )

    return len(profile)


def upsert_sku_profile_from_profile(profile: pd.DataFrame) -> int:
    """
    Sync hasil profile + clustering dari pipeline LGBM (Tab 3)
    ke tabel sku_profile.

    - Upsert berdasarkan (cabang, sku)
    - Update semua statistik + demand_level + cluster
    - TIDAK mengubah eligible_model (biar edit manual admin aman)

    Return:
        jumlah baris yang di-upsert.
    """
    if profile.empty:
        return 0

    required_cols = [
        "cabang",
        "sku",
        "n_months",
        "qty_mean",
        "qty_std",
        "qty_max",
        "qty_min",
        "total_qty",
        "zero_months",
        "zero_ratio",
        "cv",
        "demand_level",
        "cluster",
    ]
    missing = [c for c in required_cols if c not in profile.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ada di profile untuk upsert sku_profile: {missing}")

    # Bersihkan tipe
    prof = profile.copy()

    prof["n_months"] = prof["n_months"].fillna(0).astype(int)
    prof["qty_mean"] = prof["qty_mean"].astype(float)
    prof["qty_std"] = prof["qty_std"].fillna(0.0).astype(float)
    prof["qty_max"] = prof["qty_max"].astype(float)
    prof["qty_min"] = prof["qty_min"].astype(float)
    prof["total_qty"] = prof["total_qty"].astype(float)
    prof["zero_months"] = prof["zero_months"].fillna(0).astype(int)
    prof["zero_ratio"] = prof["zero_ratio"].fillna(0.0).astype(float)
    prof["cv"] = prof["cv"].fillna(0.0).astype(float)
    prof["demand_level"] = prof["demand_level"].astype(int)
    prof["cluster"] = prof["cluster"].fillna(-1).astype(int)

    insert_sql = text(
        """
        INSERT INTO sku_profile (
            cabang,
            sku,
            n_months,
            qty_mean,
            qty_std,
            qty_max,
            qty_min,
            total_qty,
            zero_months,
            zero_ratio,
            cv,
            demand_level,
            cluster,
            eligible_model
        )
        VALUES (
            :cabang,
            :sku,
            :n_months,
            :qty_mean,
            :qty_std,
            :qty_max,
            :qty_min,
            :total_qty,
            :zero_months,
            :zero_ratio,
            :cv,
            :demand_level,
            :cluster,
            :eligible_model
        )
        ON DUPLICATE KEY UPDATE
            n_months     = VALUES(n_months),
            qty_mean     = VALUES(qty_mean),
            qty_std      = VALUES(qty_std),
            qty_max      = VALUES(qty_max),
            qty_min      = VALUES(qty_min),
            total_qty    = VALUES(total_qty),
            zero_months  = VALUES(zero_months),
            zero_ratio   = VALUES(zero_ratio),
            cv           = VALUES(cv),
            demand_level = VALUES(demand_level),
            cluster      = VALUES(cluster)
        """
    )

    with engine.begin() as conn:
        for _, row in prof.iterrows():
            conn.execute(
                insert_sql,
                {
                    "cabang": row["cabang"],
                    "sku": row["sku"],
                    "n_months": int(row["n_months"]),
                    "qty_mean": float(row["qty_mean"]),
                    "qty_std": float(row["qty_std"]),
                    "qty_max": float(row["qty_max"]),
                    "qty_min": float(row["qty_min"]),
                    "total_qty": float(row["total_qty"]),
                    "zero_months": int(row["zero_months"]),
                    "zero_ratio": float(row["zero_ratio"]),
                    "cv": float(row["cv"]),
                    "demand_level": int(row["demand_level"]),
                    "cluster": int(row["cluster"]),
                    # untuk row baru, default eligible_model = 0
                    "eligible_model": 0,
                },
            )

    return len(prof)

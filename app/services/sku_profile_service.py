# app/services/sku_profile_service.py

import pandas as pd
from sqlalchemy import text

from app.db import engine
from app.profiling.sku_profiler import build_sku_profile
from app.profiling.clustering import run_sku_clustering


def load_panel_from_db() -> pd.DataFrame:
    """
    Build panel sederhana dari sales_monthly.
    Minimal kolom:
        cabang, sku, periode, qty
    Area tidak terlalu penting untuk profiling, tapi kita ambil juga.
    """

    sql = """
        SELECT
            area,
            cabang,
            sku,
            periode,
            qty
        FROM sales_monthly
        ORDER BY cabang, sku, periode
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()

    if not rows:
        return pd.DataFrame(columns=["area", "cabang", "sku", "periode", "qty"])

    df = pd.DataFrame(rows)
    df["periode"] = pd.to_datetime(df["periode"])

    return df


def save_sku_profile_to_db(profile_df: pd.DataFrame):
    """
    Upsert ke tabel sku_profile pakai UNIQUE (cabang, sku).
    """

    if profile_df.empty:
        return 0

    sql = """
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
            n_months      = VALUES(n_months),
            qty_mean      = VALUES(qty_mean),
            qty_std       = VALUES(qty_std),
            qty_max       = VALUES(qty_max),
            qty_min       = VALUES(qty_min),
            total_qty     = VALUES(total_qty),
            zero_months   = VALUES(zero_months),
            zero_ratio    = VALUES(zero_ratio),
            cv            = VALUES(cv),
            demand_level  = VALUES(demand_level),
            cluster       = VALUES(cluster),
            eligible_model= VALUES(eligible_model),
            last_updated  = CURRENT_TIMESTAMP
    """

    records = profile_df.to_dict(orient="records")

    with engine.begin() as conn:
        for rec in records:
            conn.execute(text(sql), rec)

    return len(records)


def build_and_store_sku_profile(n_clusters: int = 4) -> int:
    """
    Main function:
    1) Load panel dari DB
    2) Build profile per (cabang, sku)
    3) Clustering
    4) Upsert ke sku_profile
    Return: jumlah baris yang di-upsert
    """

    panel = load_panel_from_db()
    if panel.empty:
        raise RuntimeError("Tidak ada data di sales_monthly, isi dulu datanya.")

    profile = build_sku_profile(panel)
    profile_clustered = run_sku_clustering(profile, n_clusters=n_clusters)

    n = save_sku_profile_to_db(profile_clustered)
    return n

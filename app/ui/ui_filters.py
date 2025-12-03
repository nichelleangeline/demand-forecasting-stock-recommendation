# app/ui_filters.py
import pandas as pd
import streamlit as st
from sqlalchemy import text
from app.db import engine


def load_master_keys():
    """
    Ambil distinct cabang & sku dari panel_global_monthly
    (bukan dari raw penjualan, supaya sinkron sama model & dashboard).
    """
    sql = text("""
        SELECT DISTINCT cabang, sku
        FROM panel_global_monthly
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    # safety
    df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip()

    return df


def sidebar_filters():
    """
    Sidebar filter global:
    - multiselect cabang (searchable, bisa pilih >1, bisa 'All')
    - multiselect sku   (searchable, bisa pilih >1, bisa 'All')
    Return: dict {cabang_list or None, sku_list or None}
    """
    st.sidebar.markdown("### Filter Data")

    df_keys = load_master_keys()
    if df_keys.empty:
        st.sidebar.warning("Belum ada data di panel_global_monthly.")
        return {"cabang": None, "sku": None}

    cabang_all = sorted(df_keys["cabang"].unique().tolist())
    sku_all    = sorted(df_keys["sku"].unique().tolist())

    ALL_CABANG = "[Semua Cabang]"
    ALL_SKU    = "[Semua SKU]"

    # --------- CABANG ---------
    cabang_options = [ALL_CABANG] + cabang_all
    cabang_selected = st.sidebar.multiselect(
        "Cabang",
        options=cabang_options,
        default=[ALL_CABANG],
        help="Bisa pilih lebih dari satu. Ketik untuk search."
    )

    if (ALL_CABANG in cabang_selected) or (len(cabang_selected) == 0):
        cabang_filter = None   # artinya: semua cabang
    else:
        cabang_filter = [c for c in cabang_selected if c != ALL_CABANG]

    # --------- SKU (auto filter by cabang kalau mau) ---------
    df_for_sku = df_keys.copy()
    if cabang_filter is not None:
        df_for_sku = df_for_sku[df_for_sku["cabang"].isin(cabang_filter)]

    sku_all_filtered = sorted(df_for_sku["sku"].unique().tolist())
    sku_options = [ALL_SKU] + sku_all_filtered

    sku_selected = st.sidebar.multiselect(
        "SKU Produk",
        options=sku_options,
        default=[ALL_SKU],
        help="Bisa pilih lebih dari satu. Ketik untuk search."
    )

    if (ALL_SKU in sku_selected) or (len(sku_selected) == 0):
        sku_filter = None
    else:
        sku_filter = [s for s in sku_selected if s != ALL_SKU]

    return {
        "cabang": cabang_filter,
        "sku": sku_filter,
    }

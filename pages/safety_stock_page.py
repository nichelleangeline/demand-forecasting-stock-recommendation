# pages/safety_stock_page.py

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import text

from app.db import engine
from app.ui.theme import inject_global_theme, render_sidebar_user_and_logout
from app.loading_utils import init_loading_css, action_with_loader
from app.services.stok_policy_service import (
    build_stok_policy_from_sales_monthly,
    recompute_stok_policy,
)
from app.services.auth_guard import require_login


# ======================================================================
# BAGIAN 1: HELPER FUNCTIONS (DATABASE & LOGIC)
# ======================================================================

def load_stok_policy_with_latest_stock(cabang_filter=None):
    """
    Ambil stok_policy + latest_stock, optional filter per cabang.
    """
    base_sql = """
        SELECT
            ls.area,
            sp.cabang,
            sp.sku,
            sp.avg_qty,
            sp.max_lama,
            sp.index_lt,
            sp.proyeksi_max_baru,
            sp.growth,
            sp.max_baru,
            ls.last_txn_date,
            ls.last_stock
        FROM stok_policy sp
        LEFT JOIN latest_stock ls
          ON BINARY sp.cabang = BINARY ls.cabang
         AND BINARY sp.sku    = BINARY ls.sku
        WHERE 1=1
    """
    params = {}
    if cabang_filter:
        base_sql += " AND sp.cabang = :cabang"
        params["cabang"] = cabang_filter

    base_sql += " ORDER BY sp.cabang, sp.sku"

    with engine.connect() as conn:
        df = pd.read_sql(
            text(base_sql),
            conn,
            params=params,
            parse_dates=["last_txn_date"],
        )
    return df


def ensure_latest_stock_table():
    """
    Pastikan tabel latest_stock ada.
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS latest_stock (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        area VARCHAR(10) NOT NULL,
        cabang VARCHAR(10) NOT NULL,
        sku VARCHAR(100) NOT NULL,
        last_txn_date DATE NOT NULL,
        last_stock INT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_latest_stock_acs (area, cabang, sku)
    ) ENGINE=InnoDB;
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))


UPSERT_LATEST_STOCK_SQL = """
    INSERT INTO latest_stock (
        area,
        cabang,
        sku,
        last_txn_date,
        last_stock
    )
    VALUES (
        :area,
        :cabang,
        :sku,
        :last_txn_date,
        :last_stock
    )
    ON DUPLICATE KEY UPDATE
        last_txn_date = VALUES(last_txn_date),
        last_stock    = VALUES(last_stock),
        area          = VALUES(area)
"""


def upsert_latest_stock_records(records):
    """
    records: list dict dengan kunci:
      - cabang
      - sku
      - last_txn_date
      - last_stock
    Fungsi ini akan isi kolom area dari sales_monthly.
    """
    if not records:
        return

    ensure_latest_stock_table()

    # Ambil mapping cabang -> area
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT cabang, area FROM sales_monthly")
        ).fetchall()

    cabang_to_area = {row.cabang: row.area for row in rows}

    enriched = []
    for rec in records:
        cab = rec["cabang"]
        new_rec = dict(rec)
        new_rec["area"] = cabang_to_area.get(cab, "UNKNOWN")
        enriched.append(new_rec)

    with engine.begin() as conn:
        conn.execute(text(UPSERT_LATEST_STOCK_SQL), enriched)


def ensure_stock_history_table():
    """
    Pastikan tabel stock_history ada.
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS stock_history (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        cabang VARCHAR(10) NOT NULL,
        sku VARCHAR(100) NOT NULL,
        periode DATE NOT NULL,
        stok_akhir INT NOT NULL,
        source_filename VARCHAR(255),
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_stock_hist (cabang, sku, periode)
    ) ENGINE=InnoDB;
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))


UPSERT_STOCK_HISTORY_SQL = """
    INSERT INTO stock_history (
        cabang,
        sku,
        periode,
        stok_akhir,
        source_filename
    )
    VALUES (
        :cabang,
        :sku,
        :periode,
        :stok_akhir,
        :source_filename
    )
    ON DUPLICATE KEY UPDATE
        stok_akhir      = VALUES(stok_akhir),
        source_filename = VALUES(source_filename)
"""


def save_uploaded_stock(df_upload, filename=None):
    """
    Simpan hasil upload stok ke DB.

    Mode 1 (history per periode):
      - Simpan semua baris ke stock_history
      - Ambil periode terakhir per (cabang, sku) untuk latest_stock

    Mode 2 (snapshot stok terakhir):
      - Langsung isi latest_stock saja
    """
    if df_upload.empty:
        return

    cols = {c.lower().strip(): c for c in df_upload.columns}

    col_sku = cols.get("sku")
    col_cabang = cols.get("cabang")
    col_loc = (
        cols.get("location code")
        or cols.get("location_code")
        or cols.get("location")
    )

    if not col_sku:
        raise ValueError("File harus punya kolom: SKU.")
    if not (col_cabang or col_loc):
        raise ValueError("File harus punya kolom: cabang atau Location Code.")

    # Mode 1: history per periode
    col_periode = (
        cols.get("periode")
        or cols.get("posting date")
        or cols.get("tanggal")
        or cols.get("date")
    )
    col_stock_hist = (
        cols.get("stok_akhir")
        or cols.get("stock_akhir")
        or cols.get("stock")
        or cols.get("qty_stok")
        or cols.get("quantity")
    )

    # Mode 2: snapshot
    col_stock_snapshot = (
        cols.get("last_stock")
        or cols.get("stok_akhir")
        or cols.get("stock")
    )
    col_tanggal_snapshot = (
        cols.get("last_txn_date")
        or cols.get("posting date")
        or cols.get("tanggal")
        or cols.get("date")
    )

    records_latest = []

    # ---------------- Mode 1: history ----------------
    if col_periode and col_stock_hist:
        ensure_stock_history_table()

        source_cabang_col = col_cabang if col_cabang else col_loc
        df = df_upload[
            [source_cabang_col, col_sku, col_periode, col_stock_hist]
        ].copy()
        df.columns = ["cabang_src", "sku", "periode", "stok_akhir"]

        df["cabang_src"] = df["cabang_src"].astype(str).str.strip()
        if col_cabang:
            df["cabang"] = df["cabang_src"]
        else:
            df["cabang"] = df["cabang_src"].str[:3]

        df["sku"] = df["sku"].astype(str).str.strip()
        df["stok_akhir"] = (
            pd.to_numeric(df["stok_akhir"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["periode"] = pd.to_datetime(df["periode"]).dt.date

        df = df[["cabang", "sku", "periode", "stok_akhir"]]

        records_hist = []
        for _, row in df.iterrows():
            records_hist.append(
                {
                    "cabang": row["cabang"],
                    "sku": row["sku"],
                    "periode": row["periode"],
                    "stok_akhir": int(row["stok_akhir"]),
                    "source_filename": filename,
                }
            )

        if records_hist:
            with engine.begin() as conn:
                conn.execute(text(UPSERT_STOCK_HISTORY_SQL), records_hist)

        # ambil periode terakhir per (cabang, sku) → latest_stock
        df_sorted = df.sort_values(["cabang", "sku", "periode"])
        idx_last = (
            df_sorted.groupby(["cabang", "sku"])["periode"].transform("max")
            == df_sorted["periode"]
        )
        df_last = df_sorted[idx_last].drop_duplicates(
            subset=["cabang", "sku"], keep="last"
        )

        for _, row in df_last.iterrows():
            records_latest.append(
                {
                    "cabang": row["cabang"],
                    "sku": row["sku"],
                    "last_stock": int(row["stok_akhir"]),
                    "last_txn_date": row["periode"],
                }
            )

    # ---------------- Mode 2: snapshot ----------------
    elif col_stock_snapshot and col_tanggal_snapshot:
        source_cabang_col = col_cabang if col_cabang else col_loc
        df = df_upload[
            [source_cabang_col, col_sku, col_stock_snapshot, col_tanggal_snapshot]
        ].copy()
        df.columns = ["cabang_src", "sku", "last_stock", "last_txn_date"]

        df["cabang_src"] = df["cabang_src"].astype(str).str.strip()
        if col_cabang:
            df["cabang"] = df["cabang_src"]
        else:
            df["cabang"] = df["cabang_src"].str[:3]

        df["sku"] = df["sku"].astype(str).str.strip()
        df["last_stock"] = (
            pd.to_numeric(df["last_stock"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["last_txn_date"] = pd.to_datetime(df["last_txn_date"]).dt.date

        for _, row in df.iterrows():
            records_latest.append(
                {
                    "cabang": row["cabang"],
                    "sku": row["sku"],
                    "last_stock": int(row["last_stock"]),
                    "last_txn_date": row["last_txn_date"],
                }
            )
    else:
        raise ValueError(
            "Format kolom tidak cocok.\n"
            "- Mode history: butuh SKU, periode/Posting Date, stok_akhir/Stock, dan cabang/Location Code\n"
            "- Mode snapshot: butuh SKU, last_stock/stok_akhir, last_txn_date/Posting Date, dan cabang/Location Code"
        )

    if records_latest:
        upsert_latest_stock_records(records_latest)


def load_latest_stock_for_cabang(cabang_filter=None):
    """
    Ambil latest_stock per cabang (atau semua).
    """
    ensure_latest_stock_table()

    base_sql = """
        SELECT
            cabang,
            sku,
            last_txn_date,
            last_stock
        FROM latest_stock
        WHERE 1=1
    """
    params = {}
    if cabang_filter:
        base_sql += " AND cabang = :cabang"
        params["cabang"] = cabang_filter
    base_sql += " ORDER BY cabang, sku"

    with engine.connect() as conn:
        df = pd.read_sql(
            text(base_sql),
            conn,
            params=params,
            parse_dates=["last_txn_date"],
        )
    return df


def save_stock_from_editor(df_editor):
    """
    Simpan hasil edit manual dari data_editor ke latest_stock.
    """
    if df_editor.empty:
        return

    df = df_editor.copy()
    df["cabang"] = df["cabang"].astype(str).str.strip()
    df["sku"] = df["sku"].astype(str).str.strip()

    df["last_stock"] = pd.to_numeric(df["last_stock"], errors="coerce")
    df["last_stock"] = df["last_stock"].fillna(0).astype(int)
    df["last_txn_date"] = pd.to_datetime(df["last_txn_date"]).dt.date

    records = [
        {
            "cabang": row["cabang"],
            "sku": row["sku"],
            "last_stock": int(row["last_stock"]),
            "last_txn_date": row["last_txn_date"],
        }
        for _, row in df.iterrows()
        if row["cabang"] and row["sku"] and row["last_txn_date"]
    ]

    upsert_latest_stock_records(records)


def load_forecast_per_horizon(cabang_filter=None):
    """
    Ambil forecast future agregat per (cabang, sku, horizon).
    Dipivot jadi kolom forecast_h1, forecast_h2, dst.
    """
    sql = """
        SELECT
            cabang,
            sku,
            horizon,
            SUM(pred_qty) AS forecast_qty
        FROM forecast_monthly
        WHERE is_future = 1
          AND horizon IS NOT NULL
          AND horizon > 0
    """
    params = {}
    if cabang_filter:
        sql += " AND cabang = :cabang"
        params["cabang"] = cabang_filter

    sql += " GROUP BY cabang, sku, horizon"

    with engine.connect() as conn:
        df_fc = pd.read_sql(text(sql), conn, params=params)

    if df_fc.empty:
        return pd.DataFrame(columns=["cabang", "sku"])


    df_pivot = df_fc.pivot(
        index=["cabang", "sku"],
        columns="horizon",
        values="forecast_qty",
    ).reset_index()

    # rename kolom 1,2,3 -> forecast_h1, forecast_h2, ...
    rename_map = {}
    for c in df_pivot.columns:
        if isinstance(c, (int, float)):
            rename_map[c] = f"forecast_h{int(c)}"

    df_pivot = df_pivot.rename(columns=rename_map)
    return df_pivot


def get_mape_global_and_per_sku():
    """
    Hitung:
      - MAPE global (semua cabang+SKU di data test)
      - MAPE per (cabang, sku) di data test
    Mapping MAPE -> alpha dilakukan nanti per baris.
    """
    sql = """
        SELECT cabang, sku, qty_actual, pred_qty
        FROM forecast_monthly
        WHERE is_test = 1
          AND qty_actual IS NOT NULL
          AND pred_qty IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    if df.empty:
        # kalau nggak ada data test sama sekali
        return None, None

    # konversi numeric
    df["qty_actual"] = pd.to_numeric(df["qty_actual"], errors="coerce")
    df["pred_qty"] = pd.to_numeric(df["pred_qty"], errors="coerce")

    # GLOBAL MAPE
    y_true = df["qty_actual"].to_numpy(float)
    y_pred = df["pred_qty"].to_numpy(float)
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    mape_global = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

    # MAPE per (cabang, sku)
    def _mape_grp(g):
        y_t = g["qty_actual"].to_numpy(float)
        y_p = g["pred_qty"].to_numpy(float)
        denom = np.maximum(np.abs(y_t), eps)
        return np.mean(np.abs(y_t - y_p) / denom) * 100.0

    df_grp = (
        df.groupby(["cabang", "sku"], as_index=False)
        .apply(lambda g: pd.Series({"mape_sku": _mape_grp(g)}))
    )

    return mape_global, df_grp


def map_mape_to_alpha(mape_value: float) -> float:
    """
    Mapping MAPE -> alpha (0-1).
    Semakin jelek MAPE, semakin kecil alpha.
    """
    if mape_value is None or np.isnan(mape_value):
        return 0.6  # default konservatif

    if mape_value < 20:
        return 0.9
    elif mape_value < 30:
        return 0.7
    elif mape_value < 40:
        return 0.6
    else:
        return 0.4


# ======================================================================
# BAGIAN 2: UI STREAMLIT
# ======================================================================

st.set_page_config(
    page_title="Safety Stock & Policy",
    layout="wide",
    initial_sidebar_state="collapsed",
)

require_login()
inject_global_theme()
render_sidebar_user_and_logout()
init_loading_css()

# Guard login
if "user" not in st.session_state:
    st.error("Silakan login dulu.")
    st.stop()

user = st.session_state["user"]
role = str(user.get("role", "user")).lower()


st.markdown("## Manajemen Stok & Rekomendasi Order")
st.markdown(
    "Halaman ini membantu bagian logistik melihat stok sekarang dan saran order per bulan."
)

tab_analisa, tab_admin = st.tabs(["Analisa & Rekomendasi", "Manajemen Data (Admin)"])


# ======================================================================
# TAB 1: ANALISA & REKOMENDASI
# ======================================================================
with tab_analisa:
    # ---------------- Filter Section ----------------
    with st.container(border=True):
        st.markdown("**Filter Data**")
        f1, f2, f3, f4 = st.columns(4)

        # Load cabang
        with engine.begin() as conn:
            rows_cabang = conn.execute(
                text("SELECT DISTINCT cabang FROM stok_policy ORDER BY cabang")
            ).fetchall()
        cabang_list = [r[0] for r in rows_cabang] if rows_cabang else []

        with f1:
            selected_cabang = st.selectbox(
                "Pilih Cabang",
                ["ALL"] + cabang_list,
                index=0,
                help="Pilih cabang tertentu atau ALL untuk semua cabang.",
            )

        filter_val = selected_cabang if selected_cabang != "ALL" else None

        # Ambil forecast untuk tahu horizon yang tersedia
        df_fc_check = load_forecast_per_horizon(filter_val)
        fc_cols_check = [
            c for c in df_fc_check.columns if c.startswith("forecast_h")
        ]
        horizons = (
            sorted(int(c.replace("forecast_h", "")) for c in fc_cols_check)
            if fc_cols_check
            else [1]
        )

        with f2:
            selected_horizon = st.selectbox(
                "Periode Order",
                options=horizons,
                index=0,
                format_func=lambda x: f"Bulan ke-{x}",
                help="Bulan ke-1 = bulan depan, bulan ke-2 = dua bulan lagi.",
            )

        with f3:
            status_filter = st.selectbox(
                "Status Stok",
                ["Semua", "Risiko Kekurangan", "Potensi Kelebihan", "Aman"],
                index=0,
                help="Pilih status stok yang ingin dilihat.",
            )

        # Dropdown SKU (bisa search)
        with f4:
            with engine.begin() as conn:
                if filter_val:
                    rows_sku = conn.execute(
                        text(
                            "SELECT DISTINCT sku FROM stok_policy "
                            "WHERE cabang = :cabang ORDER BY sku"
                        ),
                        {"cabang": filter_val},
                    ).fetchall()
                else:
                    rows_sku = conn.execute(
                        text("SELECT DISTINCT sku FROM stok_policy ORDER BY sku")
                    ).fetchall()

            sku_options = ["Semua SKU"] + [r[0] for r in rows_sku]
            selected_sku = st.selectbox(
                "Pilih SKU",
                options=sku_options,
                index=0,
                help="Ketik untuk cari SKU tertentu, atau pilih 'Semua SKU'.",
            )

        # Hitung MAPE (dipakai di belakang layar, tidak ditampilkan)
        mape_global, df_mape_sku = get_mape_global_and_per_sku()

    # ---------------- Load Data Policy + Stok ----------------
    df_raw = load_stok_policy_with_latest_stock(filter_val)
    if df_raw.empty:
        st.warning(
            "Data stok_policy / latest_stock kosong untuk cabang ini. "
            "Silakan jalankan Recompute Policy dan upload stok terlebih dahulu."
        )
        st.stop()

    # Load forecast per horizon & merge
    df_fc = load_forecast_per_horizon(filter_val)
    if not df_fc.empty:
        df = df_raw.merge(df_fc, how="left", on=["cabang", "sku"])
    else:
        df = df_raw.copy()

    # Hanya pakai SKU yang punya forecast, supaya tidak ada baris kosong
    forecast_cols = [c for c in df.columns if c.startswith("forecast_h")]
    if forecast_cols:
        df[forecast_cols] = df[forecast_cols].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        )
        df["forecast_sum"] = df[forecast_cols].fillna(0).sum(axis=1)
        df = df[df["forecast_sum"] > 0].drop(columns=["forecast_sum"])

    if df.empty:
        st.info(
            "Tidak ada SKU yang punya forecast future untuk cabang/filter ini. "
            "Silakan jalankan generate forecast di halaman Dashboard Perencanaan Stok."
        )
        st.stop()

    # ---------------- Join MAPE per cabang+SKU ----------------
    if df_mape_sku is not None and not df_mape_sku.empty:
        df = df.merge(df_mape_sku, how="left", on=["cabang", "sku"])
        # kalau tidak ada mape_sku → pakai global
        if mape_global is not None:
            df["mape_used"] = df["mape_sku"].fillna(mape_global)
        else:
            df["mape_used"] = df["mape_sku"]
    else:
        # tidak ada info MAPE per SKU
        df["mape_sku"] = np.nan
        df["mape_used"] = mape_global if mape_global is not None else np.nan

    # Hitung alpha per baris (dipakai di belakang layar saja)
    df["alpha"] = df["mape_used"].apply(map_mape_to_alpha)

    # ---------------- Konversi angka dasar ----------------
    for col in ["avg_qty", "max_baru", "last_stock"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------- Hitung Safety Stock ----------------
    # Safety stock = 80% x max_baru (max_baru tidak ditampilkan di tabel)
    df["safety_stock"] = (0.8 * df["max_baru"]).round()

    # ---------------- Gunakan forecast horizon terpilih ----------------
    col_fc_target = f"forecast_h{selected_horizon}"
    col_rec_target = "rec_order_view"

    if col_fc_target in df.columns:
        df[col_fc_target] = pd.to_numeric(df[col_fc_target], errors="coerce").fillna(0)

        # Target stock = max(SafetyStock, alpha_row * Forecast_horizon)
        df["target_stock"] = np.maximum(
            df["safety_stock"],
            df["alpha"] * df[col_fc_target],
        )

        # Rekomendasi order = max(target_stock - last_stock, 0)
        df[col_rec_target] = np.where(
            df["last_stock"].notna(),
            np.maximum(df["target_stock"] - df["last_stock"], 0),
            df["target_stock"],
        )
        df[col_rec_target] = df[col_rec_target].round().astype(int)
    else:
        df[col_fc_target] = 0
        df["target_stock"] = df["safety_stock"]
        df[col_rec_target] = np.where(
            df["last_stock"].notna(),
            np.maximum(df["target_stock"] - df["last_stock"], 0),
            df["target_stock"],
        ).round().astype(int)

    # ---------------- Status stok ----------------
    cond_short = df["last_stock"].notna() & (df["last_stock"] < df["safety_stock"])
    # Kelebihan: stok di atas target_stock
    cond_excess = df["last_stock"].notna() & (df["last_stock"] > df["target_stock"])

    df["Status"] = "Aman"
    df.loc[cond_short, "Status"] = "Risiko Kekurangan"
    df.loc[cond_excess, "Status"] = "Potensi Kelebihan"
    df.loc[df["last_stock"].isna(), "Status"] = "Data Stok Kosong"

    # ---------------- Filter tambahan (status + SKU dropdown) ----------------
    df_view = df.copy()

    if status_filter != "Semua":
        df_view = df_view[df_view["Status"] == status_filter]

    if selected_sku != "Semua SKU":
        df_view = df_view[df_view["sku"] == selected_sku]

    if df_view.empty:
        st.info("Tidak ada data untuk kombinasi filter ini.")
    else:
        # ---------------- KPI ringkas ----------------
        total_order_qty = float(df_view[col_rec_target].sum())
        n_short = int((df_view["Status"] == "Risiko Kekurangan").sum())
        n_excess = int((df_view["Status"] == "Potensi Kelebihan").sum())

        # ---------------- Peringatan otomatis ----------------
        if n_short > 0 and n_excess > 0:
            st.warning(
                f"Terdapat {n_short} SKU dengan risiko kekurangan stok dan "
                f"{n_excess} SKU dengan potensi kelebihan stok pada filter ini."
            )
        elif n_short > 0:
            st.error(
                f"Terdapat {n_short} SKU dengan risiko kekurangan stok pada filter ini."
            )
        elif n_excess > 0:
            st.warning(
                f"Terdapat {n_excess} SKU dengan potensi kelebihan stok pada filter ini."
            )
        else:
            st.info(
                "Tidak ada SKU dengan risiko kekurangan maupun potensi kelebihan stok pada filter ini."
            )

        st.markdown(f"#### Ringkasan Saran Order · Bulan ke-{selected_horizon}")

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric(
                label="Total Saran Order",
                value=f"{total_order_qty:,.0f}",
                help=f"Total qty yang disarankan untuk dipesan di bulan ke-{selected_horizon}.",
            )
        with k2:
            st.metric(
                label="SKU Risiko Kekurangan",
                value=f"{n_short}",
                help="Jumlah SKU dengan stok di bawah batas aman.",
            )
        with k3:
            st.metric(
                label="SKU Potensi Kelebihan",
                value=f"{n_excess}",
                help="Jumlah SKU dengan stok di atas stok yang disarankan.",
            )

        st.divider()

        # ---------------- Tabel detail ----------------
        st.markdown("#### Detail Stok & Saran Order per SKU")

        # Kolom yang ditampilkan untuk user logistik (tanpa MAPE & Alpha)
        show_cols = [
            "area",
            "cabang",
            "sku",
            "Status",
            "last_stock",
            "safety_stock",
            col_fc_target,
            col_rec_target,
        ]
        existing_cols = [c for c in show_cols if c in df_view.columns]
        df_display = df_view[existing_cols].copy()

        df_display = df_display.rename(
            columns={
                "area": "Area",
                "cabang": "Cabang",
                "sku": "SKU",
                "Status": "Status Stok",
                "last_stock": "Stok Saat Ini",
                "safety_stock": "Safety Stock",
                col_fc_target: f"Forecast Bulan ke-{selected_horizon}",
                col_rec_target: "Saran Order",
            }
        )

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Area": st.column_config.TextColumn(
                    "Area",
                    help="Kode area cabang.",
                ),
                "Cabang": st.column_config.TextColumn(
                    "Cabang",
                    help="Kode cabang.",
                ),
                "SKU": st.column_config.TextColumn(
                    "SKU",
                    help="Kode barang (SKU).",
                ),
                "Status Stok": st.column_config.TextColumn(
                    "Status Stok",
                    help="Posisi stok: aman, kurang, atau kelebihan.",
                ),
                "Stok Saat Ini": st.column_config.NumberColumn(
                    "Stok Saat Ini",
                    format="%.0f",
                    help="Qty stok terakhir yang tercatat.",
                ),
                "Safety Stock": st.column_config.NumberColumn(
                    "Safety Stock",
                    format="%.0f",
                    help="Batas minimal stok yang sebaiknya ada di gudang.",
                ),
                f"Forecast Bulan ke-{selected_horizon}": st.column_config.NumberColumn(
                    f"Forecast Bulan ke-{selected_horizon}",
                    format="%.0f",
                    help=f"Perkiraan kebutuhan barang di bulan ke-{selected_horizon}.",
                ),
                "Saran Order": st.column_config.NumberColumn(
                    "Saran Order",
                    format="%.0f",
                    help="Qty yang disarankan untuk dipesan.",
                ),
            },
        )


# ======================================================================
# TAB 2: MANAJEMEN DATA (ADMIN)
# ======================================================================
with tab_admin:
    c1, c2 = st.columns(2)

    # ------------ Upload stok ------------
    with c1:
        with st.container(border=True):
            st.markdown("**1. Upload Data Stok**")
            st.caption(
                "File bisa CSV / Excel.\n"
                "- Mode history: SKU, Cabang/Location, tanggal (Posting Date / Periode), stok_akhir/Stock\n"
                "- Mode snapshot: SKU, Cabang/Location, tanggal terakhir, last_stock/stok_akhir"
            )

            uploaded_file = st.file_uploader(
                "Pilih file stok",
                type=["xlsx", "xls", "csv"],
                key="upload_stok",
            )

            if uploaded_file is not None:
                if st.button("Proses Upload", type="primary", key="btn_upload_stok"):
                    try:
                        fname = uploaded_file.name
                        lower = fname.lower()
                        if lower.endswith(".csv"):
                            df_up = pd.read_csv(uploaded_file)
                        else:
                            df_up = pd.read_excel(uploaded_file)

                        save_uploaded_stock(df_up, filename=fname)
                        st.success("Data stok berhasil diupload dan disimpan.")
                    except Exception as e:
                        st.error(f"Gagal memproses file: {e}")

    # ------------ Recompute policy ------------
    with c2:
        with st.container(border=True):
            st.markdown("**2. Hitung Ulang Policy (MAX & Safety Stock)**")
            st.caption(
                "Jalankan jika ada data penjualan bulanan baru, "
                "supaya nilai MAX dan Safety Stock ikut diperbarui."
            )

            def _do_recompute():
                build_stok_policy_from_sales_monthly(cabang=None)
                recompute_stok_policy(cabang=None)

            action_with_loader(
                key="btn_recompute_policy",
                button_label="Jalankan Recompute Policy",
                message="Sedang menghitung ulang policy di database...",
                fn=_do_recompute,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------ Edit stok manual ------------
    with st.container(border=True):
        st.markdown("**3. Edit Stok Manual (latest_stock)**")

        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT cabang FROM stok_policy ORDER BY cabang")
            ).fetchall()
        cab_opts = [r[0] for r in rows] if rows else []

        cab_edit = st.selectbox(
            "Pilih cabang untuk edit stok",
            ["ALL"] + cab_opts,
            key="cab_edit_manual",
            help="Pilih cabang yang stoknya mau diubah manual.",
        )
        cab_filter = cab_edit if cab_edit != "ALL" else None

        df_edit = load_latest_stock_for_cabang(cab_filter)
        if df_edit.empty:
            st.info("Belum ada data latest stok untuk cabang ini.")
            df_edit = pd.DataFrame(
                columns=["cabang", "sku", "last_txn_date", "last_stock"]
            )

        edited = st.data_editor(
            df_edit,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "cabang": st.column_config.TextColumn(
                    "Cabang",
                    required=True,
                    help="Kode cabang.",
                ),
                "sku": st.column_config.TextColumn(
                    "SKU",
                    required=True,
                    help="Kode barang.",
                ),
                "last_txn_date": st.column_config.DateColumn(
                    "Tanggal Terakhir",
                    help="Tanggal stok terakhir tercatat.",
                ),
                "last_stock": st.column_config.NumberColumn(
                    "Stok Terakhir",
                    step=1,
                    help="Qty stok pada tanggal terakhir.",
                ),
            },
            key="editor_latest_stock",
        )

        if st.button("Simpan Perubahan Stok Manual", key="btn_save_manual"):
            try:
                save_stock_from_editor(edited)
                st.success("Perubahan stok manual berhasil disimpan.")
                try:
                    st.toast("Database updated.", icon="✅")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Gagal menyimpan stok manual: {e}")

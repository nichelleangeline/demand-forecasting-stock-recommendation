# pages/admin_forecast_train.py

import datetime as dt
from datetime import date
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.db import engine
from app.ui.theme import inject_global_theme, render_sidebar_user_and_logout
from app.profiling.sku_profiler import build_and_store_sku_profile, build_sku_profile
from app.profiling.clustering import run_sku_clustering
from app.services.model_service import get_all_model_runs, activate_model
from app.features.hierarchy_features import add_hierarchy_features
from app.features.stabilizer_features import add_stabilizer_features
from app.features.outlier_handler import winsorize_outliers
from app.modeling.lgbm_trainer_cluster import train_lgbm_per_cluster
from app.services.panel_builder import build_lgbm_full_fullfeat_from_db
from app.loading_utils import init_loading_css, action_with_loader
from app.services.auth_guard import require_login

# =========================================================
# STREAMLIT SETUP
# =========================================================

st.set_page_config(
    page_title="Admin · Training Forecast",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_theme()
init_loading_css()

# Guard login
require_login()
render_sidebar_user_and_logout()

# =========================================================
# PATH OUTPUT MODEL
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_OUT_DIR = PROJECT_ROOT / "outputs" / "Light_Gradient_Boosting_Machine"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = MODEL_OUT_DIR / "models"
METRIC_DIR = MODEL_OUT_DIR / "metrics"
PLOT_DIR = MODEL_OUT_DIR / "plots_per_series"
DIAG_DIR = MODEL_OUT_DIR / "diagnostics"

for d in [MODEL_DIR, METRIC_DIR, PLOT_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# METRIC FUNCTIONS
# =========================================================

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


# =========================================================
# Helper DB: forecast_config
# =========================================================

def _get_forecast_config_from_db():
    keys = ["TRAIN_END", "TEST_START", "TEST_END"]
    placeholders = ",".join([f":k{i}" for i in range(len(keys))])
    params = {f"k{i}": key for i, key in enumerate(keys)}

    sql = f"""
        SELECT config_key, config_value
        FROM forecast_config
        WHERE config_key IN ({placeholders})
    """

    mapping = {}
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).fetchall()
        for row in rows:
            mapping[row.config_key] = row.config_value

    def parse_date(val):
        if not val:
            return None
        return dt.date.fromisoformat(val)

    train_end = parse_date(mapping.get("TRAIN_END"))
    test_start = parse_date(mapping.get("TEST_START"))
    test_end = parse_date(mapping.get("TEST_END"))

    return train_end, test_start, test_end


def _upsert_forecast_config(configs, user_id=None):
    if not configs:
        return

    with engine.begin() as conn:
        for key, value in configs.items():
            sql = """
                INSERT INTO forecast_config (config_key, config_value, updated_by)
                VALUES (:k, :v, :u)
                ON DUPLICATE KEY UPDATE
                    config_value = VALUES(config_value),
                    updated_by   = VALUES(updated_by),
                    updated_at   = CURRENT_TIMESTAMP
            """
            conn.execute(
                text(sql),
                {
                    "k": key,
                    "v": value,
                    "u": user_id,
                },
            )


def _get_sales_date_range():
    sql = """
        SELECT MIN(periode) AS min_periode,
               MAX(periode) AS max_periode
        FROM sales_monthly
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql)).fetchone()

    if not row or row.min_periode is None:
        return None, None

    min_date = row.min_periode
    max_date = row.max_periode
    if isinstance(min_date, dt.datetime):
        min_date = min_date.date()
    if isinstance(max_date, dt.datetime):
        max_date = max_date.date()
    return min_date, max_date


# =========================================================
# Helper: default train_end untuk SKU profiling
# =========================================================

def _get_default_train_end_for_profile():
    with engine.connect() as conn:
        row = (
            conn.execute(
                text("SELECT MAX(periode) AS max_periode FROM sales_monthly")
            )
            .mappings()
            .fetchone()
        )

    if row and row["max_periode"] is not None:
        max_per = row["max_periode"]
        if isinstance(max_per, dt.datetime):
            return max_per.date()
        return max_per

    today = date.today()
    return date(today.year, today.month, 1)


# =========================================================
# Helper FE kecil
# =========================================================

def _ensure_basic_flags(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    if "imputed" not in panel.columns:
        panel["imputed"] = 0
    if "spike_flag" not in panel.columns:
        panel["spike_flag"] = 0
    if "sample_weight" not in panel.columns:
        panel["sample_weight"] = 1.0
    return panel


def _ensure_rainfall_lag1(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["cabang", "sku", "periode"]).copy()

    if "rainfall_lag1" in panel.columns:
        return panel

    if "rainfall" in panel.columns:
        panel["rainfall_lag1"] = (
            panel.groupby(["cabang", "sku"])["rainfall"].shift(1)
        )
        panel = panel.drop(columns=["rainfall"])
    else:
        panel["rainfall_lag1"] = 0.0

    if "cabang" in panel.columns:
        panel.loc[panel["cabang"] != "16C", "rainfall_lag1"] = 0.0

    panel["rainfall_lag1"] = panel["rainfall_lag1"].fillna(0.0)
    return panel


# =========================================================
# Helper untuk sku_profile
# =========================================================

def _load_sku_profile() -> pd.DataFrame:
    sql = """
        SELECT
            id,
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
            eligible_model,
            last_updated
        FROM sku_profile
        ORDER BY cabang, sku
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def _update_eligible_flags(updates: pd.DataFrame) -> int:
    if updates.empty:
        return 0

    sql = """
        UPDATE sku_profile
        SET eligible_model = :eligible_model
        WHERE id = :id
    """
    with engine.begin() as conn:
        for _, row in updates.iterrows():
            conn.execute(
                text(sql),
                {
                    "id": int(row["id"]),
                    "eligible_model": int(row["eligible_model"]),
                },
            )
    return len(updates)


# =========================================================
# CSS KECIL KHUSUS HALAMAN INI
# =========================================================

st.markdown(
    """
    <style>
    /* HEADER AREA */
    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 0.25rem 0.25rem 0.75rem 0.25rem;
    }
    .page-title {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
    }
    .page-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin-top: 4px;
    }
    .page-badge {
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 999px;
        background: #f3f4ff;
        color: #4f46e5;
        border: 1px solid #e5e7ff;
    }
    
    /* METRIC CARDS */
    .metric-card {
        background-color: #ffffff;
        padding: 14px 18px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(15,23,42,0.04);
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9ca3af;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 600;
        color: #111827;
    }
    .metric-sub {
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
    }

    /* SECTION */
    .section-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .section-title span.tag {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
    }
    .section-caption {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 0.75rem;
    }

    .filter-row {
        display: flex;
        gap: 12px;
        margin-bottom: 0.75rem;
        flex-wrap: wrap;
    }
    .filter-label {
        font-size: 12px;
        font-weight: 500;
        color: #6b7280;
    }

    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        font-size: 12px;
    }

    /* PRIMARY BUTTON (GLOBAL FEEL) */
    .stButton > button[kind="primary"] {
        border-radius: 999px;
        padding: 0.45rem 1.4rem;
        font-weight: 600;
    }
    .info-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 999px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        color: #6b7280;
        font-size: 11px;
        margin-left: 6px;
        cursor: help;
    }
    .info-icon:hover {
        background: #111827;
        color: #ffffff;
    }
    /* ==== STYLE TABEL PROFIL SKU ==== */
    .sku-table [data-testid="stDataFrame"] div[data-testid="cell"] {
        font-size: 12px;
        padding: 4px 6px;
    }
    .sku-table [data-testid="stDataFrame"] thead div[data-testid="columnHeaderCell"] {
        font-size: 11px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# GUARD LOGIN & ROLE
# =========================================================

if "user" not in st.session_state:
    st.error("Silakan login dulu.")
    st.stop()

user = st.session_state["user"]
user_id = user.get("user_id")
role = user.get("role", "user")

if role != "admin":
    st.error("Halaman ini hanya bisa diakses Admin.")
    st.stop()

# =========================================================
# HEADER & STYLES
# =========================================================

st.markdown(
    """
    <style>
    /* Font Global */
    [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
    }

    /* --- Page Title Area --- */
    .header-container {
        margin-bottom: 25px;
    }
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #111827; 
        margin-bottom: 5px;
        display: flex;
        align-items: center;
    }
    .sub-title {
        font-size: 14px;
        color: #6B7280;
    }
    
    .badge-pill {
        background-color: #333333;
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 12px;
        vertical-align: middle;
        border: 1px solid #444444;
    }

    .card-base {
        height: 160px;
        border-radius: 24px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
    }

    .card-black {
        background-color: #000000;
        border: 1px solid #333333;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        color: #FFFFFF;
    }

    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #FFFFFF;
        opacity: 0.6;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        line-height: 1.2;
        color: #FFFFFF;
    }
    
    .info-tooltip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        font-size: 10px;
        font-weight: bold;
        cursor: help;
        margin-left: 8px;
        transition: all 0.2s;
        background: #222222;
        color: #FFFFFF;
        border: 1px solid #444444;
    }
    .info-tooltip:hover {
        transform: scale(1.1);
        background: #FFFFFF;
        color: #000000;
    }

    .status-text {
        font-size: 11px;
        margin-top: 8px;
        color: #FFFFFF;
        opacity: 0.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- LOGIC DATA ---
with engine.connect() as conn:
    n_sku_profile = conn.execute(
        text("SELECT COUNT(*) AS n_sku FROM sku_profile")
    ).mappings().fetchone()
    n_sku_profile = n_sku_profile["n_sku"] if n_sku_profile else 0

min_periode, max_periode = _get_sales_date_range()
train_end_cfg, test_start_cfg, test_end_cfg = _get_forecast_config_from_db()

models_for_header = get_all_model_runs()
active_model = None
if models_for_header:
    for m in models_for_header:
        if m.get("active_flag") == 1:
            active_model = m
            break

# --- RENDER HEADER ---
st.markdown(
    """
    <div class="header-container">
        <div class="main-title">
            Panel Training Forecast
            <span class="badge-pill">Admin</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- METRIC CARDS ---
col_h1, col_h2, col_h3 = st.columns(3)

with col_h1:
    tooltip_text = "Jumlah kombinasi Produk dan Cabang yang datanya sudah siap untuk dianalisa."
    st.markdown(
        f"""
        <div class="card-base card-black">
            <div class="metric-header">
                <span class="metric-label">Total Produk Terdata</span>
                <span class="info-tooltip" title="{tooltip_text}">?</span>
            </div>
            <div class="metric-value">{n_sku_profile:,}</div>
            <div class="status-text">Kombinasi Cabang & Produk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_h2:
    tooltip_text = "Rentang waktu data penjualan yang tersimpan di dalam sistem saat ini."
    period_text = "-"
    if min_periode and max_periode:
        p_start = min_periode.strftime('%b %Y')
        p_end = max_periode.strftime('%b %Y')
        period_text = f"{p_start} — {p_end}"
    
    st.markdown(
        f"""
        <div class="card-base card-black">
            <div class="metric-header">
                <span class="metric-label">Rentang Waktu Data</span>
                <span class="info-tooltip" title="{tooltip_text}">?</span>
            </div>
            <div class="metric-value" style="font-size: 20px;">{period_text}</div>
            <div class="status-text">Data historis yang tersedia</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_h3:
    tooltip_text = "Nama model yang saat ini aktif digunakan untuk memprediksi stok di Dashboard utama."
    if active_model:
        model_name = active_model.get("model_type", "Unknown").upper()
        model_id = active_model.get("id")
        display_name = (model_name[:18] + '..') if len(model_name) > 18 else model_name
        label_val = f"{display_name} <span style='color:#FFFFFF; opacity:0.6;'>#{model_id}</span>"
        sub_val = "Status: Sedang Aktif Digunakan"
    else:
        label_val = "Belum Ada"
        sub_val = "Belum ada model yang dipilih"

    st.markdown(
        f"""
        <div class="card-base card-black">
            <div class="metric-header">
                <span class="metric-label">Model Prediksi Aktif</span>
                <span class="info-tooltip" title="{tooltip_text}">?</span>
            </div>
            <div class="metric-value" style="font-size: 20px;">{label_val}</div>
            <div class="status-text">{sub_val}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Profil Data",
        "2. Aturan Seleksi",
        "3. Latih Model (Training)",
        "4. Riwayat Model",
    ]
)

# =========================================================
# PROFIL DATA
# =========================================================

with tab1:
    st.markdown(
        """
        <style>
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border-radius: 16px !important;
            border: 1px solid #E2E8F0 !important;
            padding: 20px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stDateInput"] div[data-baseweb="input"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            color: #1E293B !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
        [data-testid="stDateInput"] div[data-baseweb="input"] > div:focus-within {
            border-color: #6D5DF8 !important;
            box-shadow: 0 0 0 1px #6D5DF8 !important;
        }
        [data-testid="stMetricValue"] {
            color: #1F2933 !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-size: 14px !important;
        }
        .help-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            background-color: #E5E7EB;
            color: #475569;
            font-size: 11px;
            font-weight: 600;
            cursor: help;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 10px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <h3 style="margin:0;">Profil &amp; Kesehatan Data SKU</h3>
                <span class="help-icon"
                      title="Ringkasan histori penjualan per produk.">
                    ?
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_profile = _load_sku_profile()

    if not df_profile.empty:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric("Total SKU", f"{df_profile['sku'].nunique():,}")

            with m2:
                st.metric("Total Cabang", f"{df_profile['cabang'].nunique()}")

            with m3:
                df_profile["eligible_model"] = df_profile["eligible_model"].fillna(0).astype(int)
                eligible_count = df_profile[df_profile["eligible_model"] == 1].shape[0]
                st.metric("Siap Training", f"{eligible_count:,}")

            with m4:
                df_profile["n_months"] = df_profile["n_months"].fillna(0)
                avg_months = df_profile["n_months"].mean()
                st.metric("Rata-rata Histori", f"{avg_months:.1f} Bulan")

    st.write("")

    col_data, col_action = st.columns([3, 1], gap="medium")

    with col_data:
        with st.container(border=True):
            if df_profile.empty:
                st.info("Belum ada data profil. Silakan Refresh Profil di panel kanan.")
            else:
                st.markdown(
                    """
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                        <h4 style="margin:0;">Data Profil SKU</h4>
                        <span class="help-icon"
                              title="Gunakan filter di bawah untuk mencari SKU atau cabang tertentu.">
                            ?
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                f1, f2 = st.columns(2)
                
                cabang_unique = sorted(df_profile["cabang"].unique().tolist())
                cabang_options = ["(Semua cabang)"] + cabang_unique
                with f1:
                    selected_cabang = st.selectbox("Pilih Cabang", options=cabang_options)

                if selected_cabang == "(Semua cabang)":
                    df_for_sku = df_profile
                else:
                    df_for_sku = df_profile[df_profile["cabang"] == selected_cabang]
                
                sku_options = ["(Semua SKU)"] + sorted(df_for_sku["sku"].unique().tolist())
                with f2:
                    selected_sku = st.selectbox("Pilih SKU", options=sku_options)

                mask = pd.Series(True, index=df_profile.index)
                if selected_cabang != "(Semua cabang)":
                    mask &= df_profile["cabang"] == selected_cabang
                if selected_sku != "(Semua SKU)":
                    mask &= df_profile["sku"] == selected_sku

                df_view = df_profile.loc[mask].copy()

                df_view["total_qty"] = df_view["total_qty"].fillna(0)
                df_view["zero_ratio"] = df_view["zero_ratio"].fillna(0)
                df_view = df_view.dropna(subset=["n_months"], how="all")

                st.write(f"Menampilkan **{len(df_view):,}** data")

                st.dataframe(
                    df_view,
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                    column_order=[
                        "cabang",
                        "sku",
                        "demand_level",
                        "eligible_model",
                        "n_months",
                        "total_qty",
                        "zero_ratio",
                        "cluster",
                    ],
                    column_config={
                        "cabang": st.column_config.TextColumn(
                            "Cabang", width="small",
                            help="Kode cabang."
                        ),
                        "sku": st.column_config.TextColumn(
                            "SKU", width="medium",
                            help="Kode produk."
                        ),
                        "demand_level": st.column_config.TextColumn(
                            "Pola Permintaan",
                            help="Kategori pola permintaan."
                        ),
                        "eligible_model": st.column_config.CheckboxColumn(
                            "Train?",
                            help="SKU ikut training."
                        ),
                        "n_months": st.column_config.ProgressColumn(
                            "Durasi Data",
                            format="%d bln",
                            min_value=0,
                            max_value=int(df_profile["n_months"].max()) if not df_profile.empty else 12,
                            help="Jumlah bulan histori."
                        ),
                        "total_qty": st.column_config.NumberColumn(
                            "Total Qty",
                            help="Total unit terjual."
                        ),
                        "zero_ratio": st.column_config.NumberColumn(
                            "% Kosong",
                            format="%.2f",
                            help="Persentase bulan tanpa penjualan."
                        ),
                        "cluster": st.column_config.TextColumn(
                            "Cluster",
                            help="Cluster hasil clustering."
                        ),
                    },
                )

    with col_action:
        with st.container(border=True):
            st.markdown("#### Pengaturan")
            st.caption("Update data profil dari tabel penjualan.")
            
            default_train_end = _get_default_train_end_for_profile()
            
            train_end_profile = st.date_input(
                "Cut-off Data",
                value=default_train_end,
                help="Data penjualan setelah tanggal ini tidak dihitung.",
                key="profile_train_end",
            )

            st.write("")

            def _do_refresh_profile():
                train_end_ts = pd.Timestamp(train_end_profile)
                n_rows = build_and_store_sku_profile(train_end=train_end_ts)
                st.session_state["last_profile_refresh_rows"] = int(n_rows)
                st.session_state["last_profile_refresh_done"] = True

            action_with_loader(
                key="refresh_sku_profile",
                button_label="Refresh Profil SKU",
                message="Menghitung ulang profil SKU...",
                fn=_do_refresh_profile,
                button_type="primary",
            )

            if st.session_state.get("last_profile_refresh_done"):
                n_rows = st.session_state.get("last_profile_refresh_rows", 0)
                st.success(f"Selesai. {n_rows} data diperbarui.")
                st.session_state["last_profile_refresh_done"] = False


# =========================================================
# ATURAN SELEKSI
# =========================================================

with tab2:

    # ===== TITLE BAR =====
    st.markdown(
        """
        <div class="section-title">
            Aturan Eligible SKU 
            <span class="tag">filter training</span>
            <span class="info-icon" title="SKU yang lolos aturan akan dipakai saat training model."></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-caption">
            Atur SKU mana yang ikut training model berdasarkan histori data dan pola penjualan.
        </div>
        """,
        unsafe_allow_html=True
    )

    df_profile = _load_sku_profile()

    if df_profile.empty:
        st.info("Belum ada data di sku_profile. Bangun profil dulu di tab Profil Data.")
    else:
        # Rapikan tipe data
        df_profile["eligible_model"] = df_profile["eligible_model"].fillna(0).astype(int)
        df_profile["demand_level"] = df_profile["demand_level"].fillna(0).astype(int)
        df_profile["cluster"] = df_profile["cluster"].fillna(-1).astype(int)
        df_profile["n_months"] = df_profile["n_months"].fillna(0).astype(int)
        df_profile["total_qty"] = df_profile["total_qty"].fillna(0.0).astype(float)
        df_profile["zero_ratio"] = df_profile["zero_ratio"].fillna(0.0).astype(float)

        # =========================================================
        # Aturan otomatis
        # =========================================================
        with st.container(border=True):

            st.markdown(
                """
                <div class="section-title">
                    Aturan Otomatis
                    <span class="info-icon" title="Sistem menghitung eligible SKU berdasarkan batas minimal histori dan volume penjualan."></span>
                </div>
                """,
                unsafe_allow_html=True
            )

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                min_n_months = st.number_input(
                    "Minimal bulan punya data",
                    min_value=1,
                    max_value=120,
                    value=36,
                    key="elig_min_n_months",
                    help=(
                        "Minimal jumlah bulan histori penjualan yang tersedia untuk 1 cabang-SKU.\n\n"
                        "Contoh: jika hanya punya 10 bulan data, sedangkan batas di sini 12, "
                        "maka SKU tersebut tidak akan dianggap eligible."
                    ),
                )

            with col_b:
                min_nonzero = st.number_input(
                    "Minimal bulan ada penjualan",
                    min_value=0,
                    max_value=120,
                    value=10,
                    key="elig_min_nonzero",
                    help=(
                        "Minimal berapa bulan dalam histori yang qty-nya > 0 (pernah terjual) untuk 1 cabang-SKU.\n\n"
                        "Tujuannya supaya SKU yang hampir selalu 0 tapi sekali muncul kecil tidak ikut training."
                    ),
                )

            with col_c:
                min_total_qty = st.number_input(
                    "Minimal total qty",
                    min_value=0.0,
                    value=30.0,
                    key="elig_total_qty",
                    help=(
                        "Batas minimal total penjualan (qty) untuk 1 cabang-SKU selama PERIODE TRAIN.\n\n"
                        "- Dihitung per kombinasi cabang + SKU.\n"
                        "- Dijumlahkan dari semua bulan yang masuk periode train (sampai TRAIN_END).\n"
                        "- Nilai qty sudah mengikuti pipeline (retur minus sudah dibalik).\n\n"
                        "Contoh: dalam 36 bulan train, satu SKU cabang A total terjual 18 unit, "
                        "maka total_qty = 18. Jika batas di sini 30, SKU itu tidak diikutkan training."
                    ),
                )

            with col_d:
                max_zero_ratio = st.number_input(
                    "Maks. % bulan tanpa penjualan",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="elig_max_zero_ratio",
                    help=(
                        "Batas maksimum proporsi bulan tanpa penjualan (qty = 0) terhadap total bulan histori.\n\n"
                        "- 0.0  = tidak pernah kosong (tiap bulan selalu ada penjualan)\n"
                        "- 0.7  = maksimal 70% bulan histori boleh kosong\n"
                        "- 1.0  = boleh 100% kosong (praktis filter ini dimatikan)\n"
                    ),
                )

            def _apply_rule():
                with engine.connect() as conn:
                    df_all = pd.read_sql(
                        """
                        SELECT id, n_months, total_qty, zero_months, zero_ratio, eligible_model
                        FROM sku_profile
                        """,
                        conn,
                    )

                if df_all.empty:
                    st.session_state["elig_rule_updates"] = 0
                    return

                df_all["n_months"] = df_all["n_months"].fillna(0).astype(int)
                df_all["total_qty"] = df_all["total_qty"].fillna(0.0).astype(float)
                df_all["zero_months"] = df_all["zero_months"].fillna(0).astype(int)
                df_all["zero_ratio"] = df_all["zero_ratio"].fillna(1).astype(float)
                df_all["eligible_model"] = df_all["eligible_model"].fillna(0).astype(int)

                df_all["nonzero_months"] = df_all["n_months"] - df_all["zero_months"]

                rule = (
                    (df_all["n_months"] >= min_n_months)
                    & (df_all["nonzero_months"] >= min_nonzero)
                    & (df_all["total_qty"] >= min_total_qty)
                    & (df_all["zero_ratio"] <= max_zero_ratio)
                )

                df_all["eligible_new"] = rule.astype(int)

                changed = df_all[df_all["eligible_new"] != df_all["eligible_model"]]
                if changed.empty:
                    st.session_state["elig_rule_updates"] = 0
                    return

                updates = changed[["id", "eligible_new"]].rename(
                    columns={"eligible_new": "eligible_model"}
                )
                n = _update_eligible_flags(updates)
                st.session_state["elig_rule_updates"] = n

            action_with_loader(
                key="apply_auto_rule",
                button_label="Terapkan Aturan",
                message="Menerapkan aturan eligibility ke semua SKU...",
                fn=_apply_rule,
                button_type="primary",
            )

            if "elig_rule_updates" in st.session_state:
                n = st.session_state["elig_rule_updates"]
                if n > 0:
                    st.success(f"{n} SKU berhasil diperbarui.")
                else:
                    st.info("Tidak ada perubahan eligibility.")

        # =========================================================
        # Edit manual
        # =========================================================
        
        st.markdown("### Edit Kelayakan SKU Manual")
        st.markdown(
            """
            <p style="color:gray; font-size:0.9rem; margin-top:-10px;">
            Override keputusan sistem. Tentukan secara manual SKU mana yang <b>wajib ikut</b> atau <b>diabaikan</b> dalam training.
            </p>
            """, 
            unsafe_allow_html=True
        )

        with st.container(border=True):
            
            # Ringkasan
            st.markdown("**Ringkasan Database**")
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric(
                    label="Total SKU",
                    value=f"{len(df_profile):,}",
                    help="Total seluruh SKU unik yang tercatat di profil."
                )
                
            with m2:
                total_eligible = df_profile["eligible_model"].sum()
                st.metric(
                    label="Status Eligible",
                    value=f"{total_eligible:,}",
                    delta="Siap Training",
                    help="Jumlah SKU yang saat ini statusnya 'Ikut Training' (dicentang)."
                )

            with m3:
                avg_sparsity = df_profile["zero_ratio"].mean()
                st.metric(
                    label="Rata-rata Data Kosong",
                    value=f"{avg_sparsity:.1%}",
                    help="Rata-rata persentase bulan tanpa penjualan di seluruh SKU."
                )
                
            with m4:
                st.empty()

            st.divider()

            # Filter
            st.markdown("**Filter Pencarian**", help="Gunakan filter di bawah untuk mempersempit tabel edit.")
            
            c_filter1, c_filter2, c_filter3, c_filter4 = st.columns(4)

            with c_filter1:
                cabang_list = sorted(df_profile["cabang"].unique())
                selected_cabang = st.multiselect(
                    "Cabang",
                    cabang_list,
                    default=[],
                    placeholder="Semua Cabang",
                )

            with c_filter2:
                cluster_list = sorted(df_profile["cluster"].dropna().unique())
                selected_cluster = st.multiselect(
                    "Cluster",
                    cluster_list,
                    default=[],
                    placeholder="Semua Cluster",
                )

            with c_filter3:
                demand_list = sorted(df_profile["demand_level"].dropna().unique())
                selected_demand = st.multiselect(
                    "Pola Demand",
                    demand_list,
                    default=[],
                    placeholder="Semua Pola",
                )

            with c_filter4:
                sku_keyword = st.text_input(
                    "Cari SKU",
                    placeholder="Kode / Nama SKU...",
                )

            mask = pd.Series(True, index=df_profile.index)

            if selected_cabang:
                mask &= df_profile["cabang"].isin(selected_cabang)
            if selected_cluster:
                mask &= df_profile["cluster"].isin(selected_cluster)
            if selected_demand:
                mask &= df_profile["demand_level"].isin(selected_demand)
            if sku_keyword:
                mask &= df_profile["sku"].str.contains(sku_keyword, case=False, na=False)

            view_df = df_profile[mask].copy()
            
            if len(view_df) != len(df_profile):
                st.caption(f"🔍 Menampilkan **{len(view_df)}** baris hasil filter.")

            # Tabel editable
            edited_df = st.data_editor(
                view_df[[
                    "id",
                    "eligible_model",
                    "cabang",
                    "sku",
                    "cluster",
                    "demand_level",
                    "n_months",
                    "total_qty",
                    "zero_ratio",
                ]],
                hide_index=True,
                use_container_width=True,
                height=500,
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "eligible_model": st.column_config.CheckboxColumn(
                        "Ikut Training?",
                        width="small",
                        help="Centang untuk menyertakan SKU ini dalam training model forecast.",
                    ),
                    "cabang": st.column_config.TextColumn("Cabang", width="small"),
                    "sku": st.column_config.TextColumn("SKU Product", width="medium"),
                    "cluster": st.column_config.NumberColumn(
                        "Cluster",
                        format="%d",
                        disabled=True
                    ),
                    "demand_level": st.column_config.NumberColumn(
                        "Demand",
                        disabled=True,
                        width="small"
                    ),
                    "n_months": st.column_config.NumberColumn(
                        "Durasi",
                        format="%d bln"
                    ),
                    "total_qty": st.column_config.NumberColumn(
                        "Total Qty",
                        help="Total unit terjual selama periode train."
                    ),
                    "zero_ratio": st.column_config.ProgressColumn(
                        "Sparsity (Data Kosong)",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                        help="Semakin penuh bar, semakin jarang SKU ini terjual (banyak nol)."
                    ),
                },
                key="editor_eligible_sku",
            )

            # Tombol simpan
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn, _ = st.columns([1, 4])
            
            def _save_manual_edit():
                merged = edited_df.merge(
                    view_df[["id", "eligible_model"]],
                    on="id",
                    how="left",
                    suffixes=("_new", "_old"),
                )
                changed = merged[merged["eligible_model_new"] != merged["eligible_model_old"]]
                
                if changed.empty:
                    st.session_state["save_eligible_msg"] = "nochange"
                    return

                updates = changed[["id", "eligible_model_new"]].rename(
                    columns={"eligible_model_new": "eligible_model"}
                )
                
                n = _update_eligible_flags(updates)
                st.session_state["save_eligible_msg"] = f"{n}"

            with col_btn:
                action_with_loader(
                    key="save_elig_manual",
                    button_label="Simpan Perubahan",
                    message="Sedang menyimpan ke database...",
                    fn=_save_manual_edit,
                    button_type="primary",
                )
            
            msg = st.session_state.get("save_eligible_msg")
            if msg == "nochange":
                st.toast("Tidak ada perubahan data yang dilakukan.", icon="ℹ️")
                st.session_state["save_eligible_msg"] = None
            elif msg is not None:
                st.toast(f"Berhasil memperbarui status untuk {msg} SKU!", icon="✅")
                st.session_state["save_eligible_msg"] = None


# =========================================================
# KONFIGURASI & TRAINING
# =========================================================

with tab3:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h3 style="margin:0; padding:0;">Konfigurasi & Training Model</h3>
            <p style="color:gray; font-size:0.9rem; margin-top:5px;">
                Panel admin untuk mengatur pembagian data dan melatih ulang algoritma.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    min_periode, max_periode = _get_sales_date_range()

    if min_periode is None or max_periode is None:
        st.error("Data penjualan tidak ditemukan. Silakan upload data sales_monthly terlebih dahulu.")
    else:
        with st.container(border=True):
            st.markdown("**Status Konfigurasi Saat Ini**")
            
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(
                    label="Data Tersedia",
                    value=f"{min_periode.strftime('%b %Y')} - {max_periode.strftime('%b %Y')}"
                )
            
            with m2:
                val_train = train_end_cfg.strftime('%Y-%m-%d') if train_end_cfg else "-"
                st.metric(
                    label="Batas Data Train",
                    value=val_train,
                    help="Data sebelum tanggal ini dipakai untuk latihan"
                )
                
            with m3:
                if test_start_cfg and test_end_cfg:
                    val_test = f"{test_start_cfg.strftime('%b %Y')} - {test_end_cfg.strftime('%b %Y')}"
                else:
                    val_test = "-"
                st.metric(
                    label="Periode Test",
                    value=val_test,
                    help="Data pada rentang ini dipakai untuk validasi"
                )

        default_train_end = train_end_cfg
        default_n_test_months = 4

        if not default_train_end:
            months_back = default_n_test_months
            tmp = max_periode.replace(day=1)
            for _ in range(months_back):
                year = tmp.year
                month = tmp.month - 1
                if month == 0:
                    month = 12
                    year -= 1
                tmp = tmp.replace(year=year, month=month)
            default_train_end = tmp

        with st.form("config_train_test"):
            st.write("### Ubah Pengaturan")
            col_a, col_b = st.columns(2)
            with col_a:
                train_end_input = st.date_input(
                    "Pilih Batas Akhir Data Train",
                    value=default_train_end,
                    min_value=min_periode,
                    max_value=max_periode,
                    help="Semua data setelah tanggal ini dianggap data test."
                )
            with col_b:
                n_test_months = st.selectbox(
                    "Durasi Test (Bulan)",
                    options=[1, 2, 3, 4, 5, 6],
                    index=default_n_test_months - 1
                    if default_n_test_months in [1, 2, 3, 4, 5, 6]
                    else 3,
                    help="Berapa bulan ke depan yang dipakai untuk validasi."
                )

            submitted_cfg = st.form_submit_button("Simpan Konfigurasi Baru", type="primary", use_container_width=True)

            if submitted_cfg:
                def add_month(d: dt.date) -> dt.date:
                    year = d.year
                    month = d.month + 1
                    if month == 13:
                        month = 1
                        year += 1
                    return dt.date(year, month, 1)

                test_start = add_month(train_end_input)
                test_end = test_start
                for _ in range(n_test_months - 1):
                    test_end = add_month(test_end)

                if test_end > max_periode:
                    test_end = max_periode

                cfg = {
                    "TRAIN_END": train_end_input.isoformat(),
                    "TEST_START": test_start.isoformat(),
                    "TEST_END": test_end.isoformat(),
                }
                _upsert_forecast_config(cfg, user_id=user_id)

                st.success(f"Berhasil disimpan. Train s/d {train_end_input}, Test: {test_start} s/d {test_end}.")
                st.rerun()

        st.divider()

        st.markdown("### Eksekusi Training")
        
        train_end, test_start, test_end = _get_forecast_config_from_db()
        
        if not train_end or not test_start or not test_end:
            st.warning("Konfigurasi belum lengkap. Harap simpan periode di atas terlebih dahulu.")
        else:
            st.info(
                f"Sistem siap training menggunakan data hingga **{train_end}**. "
                f"Validasi pada periode **{test_start} s/d {test_end}**."
            )

            def _run_lgbm_training():
                with st.status("Memproses data dan melatih model...", expanded=True) as status:
                    
                    train_end_ts = pd.Timestamp(train_end)
                    test_start_ts = pd.Timestamp(test_start)
                    test_end_ts = pd.Timestamp(test_end)

                    st.write("**1. Membangun dataset fitur lengkap...**")
                    try:
                        df = build_lgbm_full_fullfeat_from_db(
                            engine,
                            train_end=train_end_ts,
                            test_start=test_start_ts,
                            test_end=test_end_ts,
                        )
                    except Exception as e:
                        st.error(f"Gagal build dataset fullfeat: {e}")
                        status.update(label="Training Gagal", state="error")
                        return

                    if df.empty:
                        st.error("Dataset kosong. Cek eligible_model atau periode data.")
                        status.update(label="Training Gagal", state="error")
                        return

                    st.write(f"   Dataset size: {len(df):,} baris")

                    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_label = f"lgbm_{run_ts}"

                    run_dir = MODEL_OUT_DIR / run_label
                    model_dir = run_dir / "models"
                    metric_dir = run_dir / "metrics"

                    run_dir.mkdir(parents=True, exist_ok=True)
                    model_dir.mkdir(parents=True, exist_ok=True)
                    metric_dir.mkdir(parents=True, exist_ok=True)

                    st.write("**2. Profiling SKU & Clustering...**")

                    df["qty"] = df["qty"].astype(float)
                    df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

                    df_train = df[df["is_train"] == 1].copy()
                    profile = build_sku_profile(df_train)

                    profile_clustered = run_sku_clustering(profile, n_clusters=4)

                    df = df.merge(
                        profile_clustered[["cabang", "sku", "cluster", "demand_level"]],
                        on=["cabang", "sku"],
                        how="left",
                    )
                    df["cluster"] = df["cluster"].fillna(-1).astype(int)

                    df = add_hierarchy_features(df)
                    if "family" in df.columns:
                        fam_map = {
                            fam: idx
                            for idx, fam in enumerate(sorted(df["family"].astype(str).unique()))
                        }
                        df["family_idx"] = df["family"].astype(str).map(fam_map).astype("int16")

                    df = add_stabilizer_features(df)
                    df = winsorize_outliers(df)

                    df["log_qty"] = np.log1p(df["qty"])
                    df["log_qty_wins"] = np.log1p(df["qty_wins"])

                    df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

                    drop_cols = [
                        "area", "cabang", "sku", "periode", "qty", "qty_wins",
                        "log_qty", "log_qty_wins", "is_train", "is_test",
                        "sample_weight", "family",
                    ]
                    feature_cols = [c for c in df.columns if c not in drop_cols]
                    st.write(f"   Fitur digunakan: {len(feature_cols)} kolom")

                    obj_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
                    if obj_cols:
                        st.warning(f"Warning: Kolom object terdeteksi: {obj_cols}")

                    st.write("**3. Training LGBM per Cluster...**")

                    cluster_ids = sorted(df["cluster"].dropna().unique())
                    models = {}

                    st.write(f"   Target Clusters: {cluster_ids}")

                    for cid in cluster_ids:
                        if cid == -1:
                            continue

                        st.write(f"   -> Training Cluster {cid}...")
                        model = train_lgbm_per_cluster(
                            df=df,
                            cluster_id=int(cid),
                            feature_cols=feature_cols,
                            log_target=True,
                            n_trials=40,
                        )

                        if model is None:
                            st.write(f"      Cluster {cid}: Gagal terbentuk.")
                            continue

                        models[cid] = model
                        model_path = model_dir / f"lgbm_cluster_{cid}.txt"
                        model.save_model(str(model_path))

                    if not models:
                        st.error("Tidak ada model berhasil dilatih.")
                        status.update(label="Training Gagal", state="error")
                        return

                    st.write("**4. Menghitung Prediksi...**")

                    df_pred_list = []
                    for cid, model in models.items():
                        df_c = df[df["cluster"] == cid].copy()
                        if df_c.empty:
                            continue
                        X_c = df_c[feature_cols]
                        pred_log = model.predict(X_c)
                        pred_qty = np.expm1(pred_log)
                        df_c["pred_qty"] = pred_qty
                        df_c["pred_run_label"] = run_label
                        df_pred_list.append(df_c)

                    df_pred = pd.concat(df_pred_list, axis=0).sort_values(["cabang", "sku", "periode"])
                    pred_path = run_dir / "panel_with_predictions.csv"
                    df_pred.to_csv(pred_path, index=False)

                    st.write("**5. Kalkulasi Metrik Global...**")

                    metrics_global = []
                    for split_name, mask in [
                        ("train", df_pred["is_train"] == 1),
                        ("test", df_pred["is_test"] == 1),
                    ]:
                        if not mask.any():
                            continue
                        yt = df_pred.loc[mask, "qty"].values
                        yp = df_pred.loc[mask, "pred_qty"].values
                        metrics_global.append(
                            {
                                "split": split_name,
                                "n_obs": int(len(yt)),
                                "MSE": mse(yt, yp),
                                "RMSE": rmse(yt, yp),
                                "MAE": mae(yt, yp),
                                "MAPE": mape(yt, yp),
                                "sMAPE": smape(yt, yp),
                            }
                        )

                    global_df = pd.DataFrame(metrics_global)
                    global_metric_path = metric_dir / "global_metrics_lgbm_streamlit.csv"
                    global_df.to_csv(global_metric_path, index=False)

                    st.write("**6. Kalkulasi Metrik Granular (SKU Level)...**")

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
                            yt_tr, yp_tr = g_tr["qty"].values, g_tr["pred_qty"].values
                            row.update({
                                "train_mae": mae(yt_tr, yp_tr),
                                "train_mse": mse(yt_tr, yp_tr),
                                "train_rmse": rmse(yt_tr, yp_tr),
                                "train_mape": mape(yt_tr, yp_tr),
                                "train_smape": smape(yt_tr, yp_tr),
                            })
                        else:
                            row.update({"train_mae": np.nan, "train_mse": np.nan, "train_rmse": np.nan, "train_mape": np.nan, "train_smape": np.nan})

                        if len(g_te) > 0:
                            yt_te, yp_te = g_te["qty"].values, g_te["pred_qty"].values
                            row.update({
                                "test_mae": mae(yt_te, yp_te),
                                "test_mse": mse(yt_te, yp_te),
                                "test_rmse": rmse(yt_te, yp_te),
                                "test_mape": mape(yt_te, yp_te),
                                "test_smape": smape(yt_te, yp_te),
                            })
                        else:
                            row.update({"test_mae": np.nan, "test_mse": np.nan, "test_rmse": np.nan, "test_mape": np.nan, "test_smape": np.nan})

                        rows.append(row)

                    metrics_series = pd.DataFrame(rows)
                    metrics_series["gap_RMSE"] = metrics_series["test_rmse"] - metrics_series["train_rmse"]
                    metrics_series["ratio_RMSE"] = metrics_series["test_rmse"] / metrics_series["train_rmse"]

                    series_metric_path = metric_dir / "metrics_by_series_lgbm_streamlit.csv"
                    metrics_series.to_csv(series_metric_path, index=False)

                    st.write("**7. Finalisasi & Penyimpanan Metadata...**")

                    try:
                        train_mask = df_pred["is_train"] == 1
                        test_mask = df_pred["is_test"] == 1

                        train_start_dt = df_pred.loc[train_mask, "periode"].min().date()
                        train_end_dt = df_pred.loc[train_mask, "periode"].max().date()

                        if test_mask.any():
                            test_start_dt = df_pred.loc[test_mask, "periode"].min().date()
                            test_end_dt = df_pred.loc[test_mask, "periode"].max().date()
                        else:
                            test_start_dt = None
                            test_end_dt = None

                        global_train_rmse = None
                        global_test_rmse = None
                        global_train_mae = None
                        global_test_mae = None

                        if not global_df.empty:
                            if "train" in global_df["split"].values:
                                row_tr = global_df[global_df["split"] == "train"].iloc[0]
                                global_train_rmse = float(row_tr["RMSE"])
                                global_train_mae = float(row_tr["MAE"])
                            if "test" in global_df["split"].values:
                                row_te = global_df[global_df["split"] == "test"].iloc[0]
                                global_test_rmse = float(row_te["RMSE"])
                                global_test_mae = float(row_te["MAE"])

                        n_test_months_calc = 0
                        if test_start and test_end:
                            n_test_months_calc = ((test_end.year - test_start.year) * 12 + (test_end.month - test_start.month) + 1)

                        n_clusters = len([cid for cid in cluster_ids if cid != -1])
                        model_type = "lgbm"
                        description = f"LGBM · train {train_start_dt}-{train_end_dt} · test {test_start_dt}-{test_end_dt}"

                        params_payload = {
                            "run_dir": str(run_dir),
                            "panel_path": str(pred_path),
                            "train_end": train_end.isoformat(),
                            "test_start": test_start.isoformat(),
                            "test_end": test_end.isoformat(),
                            "n_test_months": n_test_months_calc,
                            "n_clusters": n_clusters,
                        }

                        with engine.begin() as conn:
                            insert_sql = """
                                INSERT INTO model_run (
                                    model_type, description, trained_at, trained_by,
                                    train_start, train_end, test_start, test_end,
                                    n_test_months, n_clusters, params_json, feature_cols_json,
                                    global_train_rmse, global_test_rmse, global_train_mae, global_test_mae
                                ) VALUES (
                                    :model_type, :description, :trained_at, :trained_by,
                                    :train_start, :train_end, :test_start, :test_end,
                                    :n_test_months, :n_clusters, :params_json, :feature_cols_json,
                                    :global_train_rmse, :global_test_rmse, :global_train_mae, :global_test_mae
                                )
                            """
                            result = conn.execute(
                                text(insert_sql),
                                {
                                    "model_type": model_type,
                                    "description": description,
                                    "trained_at": dt.datetime.now(),
                                    "trained_by": user_id,
                                    "train_start": train_start_dt,
                                    "train_end": train_end_dt,
                                    "test_start": test_start_dt,
                                    "test_end": test_end_dt,
                                    "n_test_months": n_test_months_calc,
                                    "n_clusters": n_clusters,
                                    "params_json": json.dumps(params_payload),
                                    "feature_cols_json": json.dumps(feature_cols),
                                    "global_train_rmse": global_train_rmse,
                                    "global_test_rmse": global_test_rmse,
                                    "global_train_mae": global_train_mae,
                                    "global_test_mae": global_test_mae,
                                },
                            )
                            run_id = result.lastrowid
                        
                        status.update(label="Training Selesai & Berhasil!", state="complete")
                        
                        st.success(f"Metadata tersimpan: ID Run = {run_id}")

                        st.divider()
                        st.markdown("#### Hasil Evaluasi Model")
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        col_m1.metric("Train RMSE", f"{global_train_rmse:.2f}" if global_train_rmse else "-")
                        col_m2.metric(
                            "Test RMSE",
                            f"{global_test_rmse:.2f}" if global_test_rmse else "-",
                            delta=f"{global_test_rmse - global_train_rmse:.2f}" if (global_test_rmse and global_train_rmse) else None,
                            delta_color="inverse"
                        )
                        col_m3.metric("Train MAE", f"{global_train_mae:.2f}" if global_train_mae else "-")
                        col_m4.metric("Test MAE", f"{global_test_mae:.2f}" if global_test_mae else "-")

                        with st.expander("Lihat Detail Tabel Metrik Global & Per-SKU"):
                             st.write("**Global Metrics:**")
                             st.dataframe(global_df, use_container_width=True, hide_index=True)
                             
                             st.write("**Sampel Metrik per SKU (Top 30):**")
                             st.dataframe(metrics_series.head(30), use_container_width=True, hide_index=True)

                    except Exception as e:
                        st.error(f"Gagal simpan ke DB: {e}")
                        status.update(label="Error saat menyimpan data", state="error")

            st.markdown("<br>", unsafe_allow_html=True)
            action_with_loader(
                key="train_lgbm_clusters",
                button_label="Mulai Training Model",
                message="Sedang menjalankan proses training...",
                fn=_run_lgbm_training,
                button_type="primary",
            )


# =========================================================
# RIWAYAT / MANAJEMEN MODEL
# =========================================================

with tab4:
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <div class="section-title">Manajemen Model <span class="tag">Production</span></div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    models = get_all_model_runs()

    if not models:
        st.info("Belum ada history training model.", icon="ℹ️")
    else:
        df_models = pd.DataFrame(models)

        if "trained_at" in df_models.columns:
            df_models = df_models.sort_values("trained_at", ascending=False)

        # reset index supaya selaras dengan selection row
        df_models = df_models.reset_index(drop=True)

        if "global_train_rmse" in df_models.columns and "global_test_rmse" in df_models.columns:
            df_models["rmse_gap"] = df_models["global_test_rmse"] - df_models["global_train_rmse"]
        else:
            df_models["rmse_gap"] = np.nan

        df_models["status_display"] = np.where(df_models["active_flag"] == 1, "✅ ACTIVE", "ARCHIVED")

        if "global_test_rmse" in df_models.columns:
            max_rmse = df_models["global_test_rmse"].max()
            if pd.isna(max_rmse) or max_rmse <= 0:
                max_rmse = 1.0
        else:
            max_rmse = 1.0

        st.caption("Pilih salah satu baris model untuk melihat detail atau mengaktifkannya.")

        display_cols = [
            "status_display",
            "id",
            "model_type",
            "n_test_months",
            "global_train_rmse",
            "global_test_rmse",
            "rmse_gap",
            "trained_at",
        ]

        df_display = df_models[display_cols]

        selection = st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            height=400,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "status_display": st.column_config.TextColumn(
                    "Status", width="small"
                ),
                "id": st.column_config.NumberColumn(
                    "ID", format="%d", width="small"
                ),
                "model_type": st.column_config.TextColumn(
                    "Algoritma", width="medium"
                ),
                "n_test_months": st.column_config.NumberColumn(
                    "Test (Bln)", format="%d bln"
                ),
                "global_train_rmse": st.column_config.ProgressColumn(
                    "Train RMSE",
                    format="%.2f",
                    min_value=0,
                    max_value=max_rmse,
                    help="Error data latih"
                ),
                "global_test_rmse": st.column_config.ProgressColumn(
                    "Test RMSE",
                    format="%.2f",
                    min_value=0,
                    max_value=max_rmse,
                    help="Error data uji"
                ),
                "rmse_gap": st.column_config.NumberColumn(
                    "Gap", format="%.2f", help="Selisih Test - Train"
                ),
                "trained_at": st.column_config.DatetimeColumn(
                    "Waktu Training", format="D MMM YY, HH:mm"
                ),
            }
        )

        if selection.selection["rows"]:
            selected_index = selection.selection["rows"][0]
            selected_row = df_models.iloc[selected_index]
            
            st.divider()
            
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                
                with c1:
                    st.markdown(f"#### Detail Model ID: {selected_row['id']}")
                    st.markdown(
                        f"**Periode Train:** {selected_row['train_start']} s/d {selected_row['train_end']} "
                        f"• **Periode Test:** {selected_row['test_start']} s/d {selected_row['test_end']}"
                    )
                    
                    is_active = (selected_row['active_flag'] == 1)
                    if is_active:
                        st.success("Model ini sedang **AKTIF** digunakan di dashboard.")
                    else:
                        st.warning("Model ini statusnya **ARSIP**.")

                with c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if not is_active:
                        if st.button("Aktifkan Model Ini", type="primary", use_container_width=True):
                            ok = activate_model(selected_row['id'])
                            if ok:
                                st.toast(f"Model {selected_row['id']} berhasil diaktifkan!", icon="✅")
                                st.rerun()
                            else:
                                st.error("Gagal update database.")
                    else:
                        st.button("Sedang Digunakan", disabled=True, use_container_width=True)
        else:
            st.info("Pilih salah satu baris di tabel untuk melihat opsi aktivasi.")

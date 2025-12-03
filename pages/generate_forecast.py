# pages/forecast_dashboard.py

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from app.db import engine
from app.ui.theme import inject_global_theme, render_sidebar_user_and_logout
from app.services.model_service import get_all_model_runs
from app.services.forecast_service import generate_and_store_forecast
from app.loading_utils import init_loading_css
from app.services.auth_guard import require_login  # guard standar


# =========================================================
# HELPER: LAST FORECAST GENERATION INFO
# =========================================================

def _get_last_forecast_generated(model_run_id: int):
    """
    Ambil info terakhir kali forecast future digenerate untuk model_run_id tertentu.
    Disimpan di tabel forecast_config dengan key: LAST_FORECAST_GEN_{model_run_id}
    """
    key = f"LAST_FORECAST_GEN_{model_run_id}"
    sql = """
        SELECT config_value, updated_by, updated_at
        FROM forecast_config
        WHERE config_key = :k
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"k": key}).mappings().fetchone()

    if not row:
        return None, None, None

    return row["config_value"], row.get("updated_by"), row.get("updated_at")


def _set_last_forecast_generated(
    model_run_id: int,
    user_id: int | None,
    when: dt.datetime | None = None,
):
    """
    Simpan timestamp terakhir generate forecast future ke forecast_config.
    """
    if when is None:
        when = dt.datetime.now()

    key = f"LAST_FORECAST_GEN_{model_run_id}"
    val = when.isoformat()

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
            {"k": key, "v": val, "u": user_id},
        )


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Forecast Penjualan",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Guard login
require_login()

# Tema global
inject_global_theme()

# Sidebar user + logout
render_sidebar_user_and_logout()

# CSS loading / spinner
init_loading_css()

# Double guard
if "user" not in st.session_state:
    st.error("Silakan login dulu.")
    st.stop()

user = st.session_state["user"]
user_id = user.get("user_id")
role = user.get("role", "user")
is_admin = str(role).lower() == "admin"


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .metric-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 4px;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 4px;
    }
    /* Hide default plotly modebar */
    .modebar { display: none !important; }

    /* Dropdown putih */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER & MODEL INFO
# =========================================================

models = get_all_model_runs()
active_model = None
if models:
    for m in models:
        if m.get("active_flag") == 1:
            active_model = m
            break

col_head1, col_head2 = st.columns([2, 1])

with col_head1:
    st.title("Forecast Data Penjualan")
    st.caption("Pantau histori penjualan dan forecast stok per cabang & SKU.")

with col_head2:
    if not active_model:
        st.warning("Belum ada model aktif. Set dulu di halaman Training / Kontrol Forecast.")
    else:
        with st.container(border=True):
            st.markdown(f"**Model Aktif:** {active_model['model_type']} (ID {active_model['id']})")
            st.caption(f"Train: {active_model.get('train_start')} s/d {active_model.get('train_end')}")
            st.caption(f"Test: {active_model.get('test_start')} s/d {active_model.get('test_end')}")

if not active_model:
    st.stop()


# =========================================================
# SECTION 1: FORECAST GENERATION
# =========================================================

last_val, last_by, last_at = _get_last_forecast_generated(active_model["id"])

with st.expander("Generate Forecast Future & Simpan ke Database", expanded=False):
    st.caption(
        "Tombol ini hitung ulang forecast masa depan untuk seluruh cabang-SKU lalu simpan ke tabel "
        "`forecast_monthly`."
    )

    c_gen1, c_gen2 = st.columns([1, 2])

    with c_gen1:
        horizon_months = st.slider(
            "Horizon Prediksi (bulan ke depan)",
            min_value=1,
            max_value=12,
            value=6,
            help="Berapa banyak bulan ke depan yang mau dihitung forecast-nya.",
        )

    with c_gen2:
        if last_at:
            ts_str = last_at.strftime("%d %b %Y, %H:%M")
            st.markdown(f"**Terakhir generate:** {ts_str}")
        else:
            st.markdown("**Terakhir generate:** Belum pernah")

        btn_generate = st.button(
            "Generate & Simpan Forecast",
            type="primary",
            use_container_width=True,
        )

    if btn_generate:
        try:
            with st.spinner("Sedang memproses forecast dan menyimpan ke database..."):
                n_rows = generate_and_store_forecast(horizon_months=horizon_months)
                _set_last_forecast_generated(active_model["id"], user_id)
            st.success(f"Berhasil. {n_rows} baris forecast tersimpan di database.")
            st.rerun()
        except Exception as e:
            st.error(f"Gagal generate forecast: {e}")


# =========================================================
# LOAD DATA
# =========================================================

with engine.connect() as conn:
    df = pd.read_sql(
        text(
            """
            SELECT
                model_run_id,
                area,
                cabang,
                sku,
                periode,
                qty_actual,
                pred_qty,
                is_train,
                is_test,
                is_future,
                horizon
            FROM forecast_monthly
            WHERE model_run_id = :mid
            ORDER BY cabang, sku, periode
            """
        ),
        conn,
        params={"mid": active_model["id"]},
        parse_dates=["periode"],
    )

if df.empty:
    st.warning("Data forecast kosong. Jalankan generate forecast dulu.")
    st.stop()

# Type Conversion
df["qty_actual"] = pd.to_numeric(df["qty_actual"], errors="coerce")
df["pred_qty"] = pd.to_numeric(df["pred_qty"], errors="coerce")
df["is_train"] = df["is_train"].astype(int)
df["is_test"] = df["is_test"].astype(int)
df["is_future"] = df["is_future"].astype(int)
df["horizon"] = df["horizon"].fillna(0).astype(int)

future_mask = df["is_future"] == 1
has_future = future_mask.any()


# =========================================================
# SECTION 2: FILTERS
# =========================================================
st.markdown("### Filter Tampilan")

with st.container(border=True):
    f1, f2, f3, f4 = st.columns(4)

    # Rentang periode
    with f1:
        view_mode = st.selectbox(
            "Rentang Periode",
            options=["12 bulan terakhir", "Semua histori + future"],
            index=0,
        )

    # Max horizon future, dinamis & rata dengan yang lain
    with f2:
        if has_future:
            max_future_h = int(df.loc[future_mask, "horizon"].max())
            horizon_options = list(range(1, max_future_h + 1))
            default_idx = len(horizon_options) - 1

            view_horizon = st.selectbox(
                "Max Horizon Future",
                options=horizon_options,
                index=default_idx,
                format_func=lambda x: f"{x}",
                help="Pilih sampai bulan ke berapa forecast future ditampilkan. Contoh: 3 = 3 bulan ke depan.",
            )
        else:
            view_horizon = None
            st.selectbox(
                "Max Horizon Future",
                options=["Tidak ada data future"],
                index=0,
                disabled=True,
            )

    # Cabang (wajib 1)
    with f3:
        cabang_list = sorted(df["cabang"].dropna().unique().tolist())
        if not cabang_list:
            st.error("Tidak ada data cabang di tabel forecast.")
            st.stop()

        cabang_selected = st.selectbox(
            "Pilih Cabang",
            options=cabang_list,
            index=0,
        )

    # SKU (wajib 1)
    with f4:
        df_for_sku = df[df["cabang"] == cabang_selected]
        sku_list = sorted(df_for_sku["sku"].dropna().unique().tolist())
        if not sku_list:
            st.error("Tidak ada SKU untuk cabang ini.")
            st.stop()

        sku_selected = st.selectbox(
            "Pilih SKU",
            options=sku_list,
            index=0,
        )

# Apply filter
df_view = df.copy()
df_view = df_view[df_view["cabang"] == cabang_selected]
df_view = df_view[df_view["sku"] == sku_selected]

if has_future and view_horizon is not None:
    df_view = df_view[
        (df_view["is_future"] == 0) | (df_view["horizon"] <= view_horizon)
    ]

if view_mode == "12 bulan terakhir":
    last_date = df_view["periode"].max()
    if pd.notna(last_date):
        cutoff = (last_date.to_period("M") - 11).to_timestamp()
        df_view = df_view[df_view["periode"] >= cutoff]

df_view = df_view.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

if df_view.empty:
    st.warning("Data kosong setelah filter. Coba ubah filter.")
    st.stop()


# =========================================================
# SECTION 3: KPI CARDS
# =========================================================

def _mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


future_view = df_view[df_view["is_future"] == 1]
total_pred_future = float(future_view["pred_qty"].sum()) if not future_view.empty else 0.0

test_mask = (df_view["is_test"] == 1) & df_view["qty_actual"].notna()
if test_mask.any():
    mape_test_val = _mape(
        df_view.loc[test_mask, "qty_actual"], df_view.loc[test_mask, "pred_qty"]
    )
else:
    mape_test_val = None

st.markdown("### Ringkasan Forecast")

k1, k2 = st.columns(2)


def render_card(col, title, value, sub, color=None):
    style_color = f"color: {color};" if color else ""
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value" style="{style_color}">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


render_card(
    k1,
    "Total Prediksi Future",
    f"{total_pred_future:,.0f}",
    "Total qty prediksi untuk cabang & SKU ini.",
)

mape_str = f"{mape_test_val:,.1f}%" if mape_test_val is not None else "-"
render_card(
    k2,
    "MAPE Data Test",
    mape_str,
    "Perbandingan prediksi vs aktual di periode test.",
    color="#2563eb",
)


# =========================================================
# SECTION 4: CHART
# =========================================================
st.markdown("### Visualisasi Tren & Prediksi")

with st.container(border=True):
    df_chart = df_view[["periode", "qty_actual", "pred_qty"]].copy()
    df_chart = df_chart[
        df_chart["qty_actual"].notna() | df_chart["pred_qty"].notna()
    ]
    df_chart = df_chart.sort_values("periode")

    if df_chart.empty:
        st.info("Tidak ada data untuk ditampilkan di grafik.")
    else:
        plot_df = df_chart.melt(
            id_vars=["periode"],
            value_vars=["qty_actual", "pred_qty"],
            var_name="series",
            value_name="qty",
        )
        series_map = {"qty_actual": "Aktual", "pred_qty": "Prediksi"}
        plot_df["series"] = plot_df["series"].map(series_map)

        fig = px.line(
            plot_df,
            x="periode",
            y="qty",
            color="series",
            markers=True,
            color_discrete_map={
                "Aktual": "#2563eb",   # biru
                "Prediksi": "#f97316"  # oranye
            },
            labels={"periode": "Periode", "qty": "Qty", "series": ""},
        )

        fig.update_traces(line=dict(width=2.5))
        fig.update_layout(
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title=None,
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
            height=400,
        )

        dmin, dmax = df_chart["periode"].min(), df_chart["periode"].max()
        months_span = (dmax.year - dmin.year) * 12 + (dmax.month - dmin.month)
        dtick_val = "M3" if months_span > 18 else "M1"
        fig.update_xaxes(dtick=dtick_val, tickformat="%b %Y")

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =========================================================
# SECTION 5: TABLE & DOWNLOAD
# =========================================================
st.markdown("### Detail Data Forecast")

with st.container(border=True):
    df_download = df_view.copy().sort_values(["cabang", "sku", "periode"])
    csv_bytes = df_download.to_csv(index=False).encode("utf-8-sig")

    col_d1, col_d2 = st.columns([5, 1])
    with col_d2:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"forecast_{active_model['id']}_{cabang_selected}_{sku_selected}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        df_download.head(1000),
        use_container_width=True,
        hide_index=True,
        column_config={
            "model_run_id": st.column_config.NumberColumn(
                "Model ID",
                help="ID model yang dipakai untuk forecast ini.",
            ),
            "area": st.column_config.TextColumn(
                "Area",
                help="Area / region cabang.",
            ),
            "cabang": st.column_config.TextColumn(
                "Cabang",
                help="Kode cabang.",
            ),
            "sku": st.column_config.TextColumn(
                "SKU",
                help="Kode barang.",
            ),
            "periode": st.column_config.DateColumn(
                "Periode",
                format="D MMM YYYY",
                help="Bulan transaksi / forecast.",
            ),
            "qty_actual": st.column_config.NumberColumn(
                "Qty Aktual",
                format="%.0f",
                help="Qty penjualan asli pada periode tersebut.",
            ),
            "pred_qty": st.column_config.NumberColumn(
                "Qty Prediksi",
                format="%.0f",
                help="Qty hasil prediksi model.",
            ),
            "is_train": st.column_config.CheckboxColumn(
                "Train",
                help="Centang = baris ini masuk data training.",
                disabled=True,
            ),
            "is_test": st.column_config.CheckboxColumn(
                "Test",
                help="Centang = baris ini masuk data test / validasi.",
                disabled=True,
            ),
            "is_future": st.column_config.CheckboxColumn(
                "Future",
                help="Centang = baris ini hasil forecast masa depan (tanpa aktual).",
                disabled=True,
            ),
            "horizon": st.column_config.NumberColumn(
                "Horizon",
                format="%d",
                help="Jarak bulan dari titik terakhir histori. 1 = bulan depan, 2 = dua bulan lagi, dst.",
            ),
        },
    )

    if len(df_download) > 1000:
        st.caption("Menampilkan 1.000 baris pertama. Untuk lengkapnya silakan download CSV.")

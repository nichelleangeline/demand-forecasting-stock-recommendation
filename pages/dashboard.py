# pages/dashboard.py
import io
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from app.db import engine
from app.ui.theme import inject_global_theme, render_sidebar_user_and_logout
from app.services.auth_guard import require_login
from app.services.model_service import get_all_model_runs
from app.loading_utils import init_loading_css
from app.services.page_loader import page_loading, init_page_loader_css


st.set_page_config(
    page_title="Dashboard Perencanaan",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_page_loader_css()

def _load_forecast_data():
    active_model_local = None
    models = get_all_model_runs()
    if not models:
        return None, None

    for m in models:
        if m.get("active_flag") == 1:
            active_model_local = m
            break

    if not active_model_local:
        return None, None

    mid = int(active_model_local["id"])

    sql = """
        SELECT area, cabang, sku, periode, qty_actual, pred_qty, is_future
        FROM forecast_monthly
        WHERE model_run_id = :mid
        ORDER BY periode
    """
    with engine.connect() as conn:
        df_local = pd.read_sql(
            text(sql),
            conn,
            params={"mid": mid},
            parse_dates=["periode"],
        )

    if df_local.empty:
        return None, active_model_local

    df_local["area"] = df_local["area"].astype(str).str.strip()
    df_local["cabang"] = df_local["cabang"].astype(str).str.strip()
    df_local["sku"] = df_local["sku"].astype(str).str.strip()

    df_local["qty_actual"] = pd.to_numeric(df_local["qty_actual"], errors="coerce").fillna(0.0)
    df_local["pred_qty"] = pd.to_numeric(df_local["pred_qty"], errors="coerce").fillna(0.0)
    df_local["is_future"] = pd.to_numeric(df_local["is_future"], errors="coerce").fillna(0).astype(int)

    return df_local, active_model_local


with page_loading("Menyiapkan dashboard & mengambil data forecast..."):
    require_login()
    inject_global_theme()
    render_sidebar_user_and_logout()
    init_loading_css()
    df_raw, active_model = _load_forecast_data()

if "user" not in st.session_state:
    st.stop()


st.markdown(
    """
    <style>
    .stApp { background-color: #f8fafc; }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stMarkdown p { color: #64748b; font-size: 0.9rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }

    .metric-container{
        display:flex; flex-direction:column; justify-content:center;
        background:#fff; border:1px solid #e2e8f0; border-radius:14px;
        padding:16px 20px;
        box-shadow:0 4px 6px -1px rgba(0,0,0,.05);
        height:100%;
        transition:transform .2s;
    }
    .metric-container:hover{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,.05);
    }
    .metric-label{
        font-size:.78rem; font-weight:600; color:#64748b;
        text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;
    }
    .metric-value{
        font-size:1.6rem; font-weight:700; color:#0f172a; margin-bottom:2px;
    }
    .metric-sub{ font-size:.75rem; color:#94a3b8; }

    .stSelectbox div[data-baseweb="select"] > div{
        background:#fff; border-radius:10px; border-color:#e2e8f0; min-height:2.4rem;
    }

    .download-btn{
        border:none; padding:.55rem 1.3rem; border-radius:8px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color:#f9fafb; font-size:.85rem; font-weight:500;
        cursor:pointer; transition: all .2s ease;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,.1);
    }
    .download-btn:hover{
        opacity:.92; transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,.1);
    }

    .stDataFrame{
        border-radius:12px; border:1px solid #e2e8f0;
        overflow:hidden; background:#fff;
    }
    .modebar{ display:none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def apply_chart_theme(fig):
    fig.update_layout(
        font_family="Inter, sans-serif",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="#e2e8f0",
            tickfont=dict(color="#64748b", size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            showline=False,
            tickfont=dict(color="#64748b", size=11),
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
            font=dict(color="#475569"),
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, sans-serif",
        ),
    )
    return fig


def format_si_short(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 1_000:
        return f"{v/1_000:.0f}k"
    return f"{v:,.0f}"


def clean_metric(col, label, val, sub):
    col.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_stok_policy_with_latest_stock(cabang_filter=None):
    sql = """
        SELECT
            ls.area,
            sp.cabang,
            sp.sku,
            sp.avg_qty,
            sp.max_baru,
            ls.last_stock
        FROM stok_policy sp
        LEFT JOIN latest_stock ls
          ON TRIM(UPPER(sp.cabang)) COLLATE utf8mb4_unicode_ci = TRIM(UPPER(ls.cabang)) COLLATE utf8mb4_unicode_ci
         AND TRIM(UPPER(sp.sku))    COLLATE utf8mb4_unicode_ci = TRIM(UPPER(ls.sku))    COLLATE utf8mb4_unicode_ci
        WHERE 1=1
    """
    params = {}
    if cabang_filter:
        sql += " AND TRIM(UPPER(sp.cabang)) = :cabang"
        params["cabang"] = str(cabang_filter).strip().upper()

    sql += " ORDER BY sp.cabang, sp.sku"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return df

    df["area"] = df["area"].astype(str).str.strip()
    df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    return df


def get_mape_global_and_per_sku(model_run_id: int):
    sql = """
        SELECT cabang, sku, qty_actual, pred_qty
        FROM forecast_monthly
        WHERE model_run_id = :mid
          AND is_test = 1
          AND qty_actual IS NOT NULL
          AND pred_qty IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"mid": int(model_run_id)})

    if df.empty:
        return None, None

    df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    df["qty_actual"] = pd.to_numeric(df["qty_actual"], errors="coerce")
    df["pred_qty"] = pd.to_numeric(df["pred_qty"], errors="coerce")
    df = df.dropna(subset=["qty_actual", "pred_qty"])

    if df.empty:
        return None, None

    y_true = df["qty_actual"].to_numpy(float)
    y_pred = df["pred_qty"].to_numpy(float)
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    mape_global = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

    def _mape_grp(g):
        y_t = g["qty_actual"].to_numpy(float)
        y_p = g["pred_qty"].to_numpy(float)
        denom2 = np.maximum(np.abs(y_t), eps)
        return float(np.mean(np.abs(y_t - y_p) / denom2) * 100.0)

    df_grp = (
        df.groupby(["cabang", "sku"], as_index=False)
        .apply(lambda g: pd.Series({"mape_sku": _mape_grp(g)}))
        .reset_index(drop=True)
    )

    return mape_global, df_grp


def map_mape_to_alpha(mape_value: float) -> float:
    if mape_value is None or np.isnan(mape_value):
        return 0.6
    if mape_value < 20:
        return 0.9
    if mape_value < 30:
        return 0.7
    if mape_value < 40:
        return 0.6
    return 0.4


st.title("Dashboard Monitoring Penjualan & Forecasting")

if df_raw is None or active_model is None:
    st.warning("Data forecast belum tersedia.")
    st.stop()

model_run_id = int(active_model["id"])

with st.container():
    f1, f2, f3 = st.columns([1.5, 1.5, 1.5])

    areas = sorted(df_raw["area"].dropna().unique().tolist())
    area_sel = f1.selectbox("Area", ["Semua Area"] + areas)

    df_tmp = df_raw[df_raw["area"] == area_sel] if area_sel != "Semua Area" else df_raw
    cabang_all = sorted(df_tmp["cabang"].dropna().unique().tolist())
    cabang_sel = f2.selectbox("Cabang", ["Semua Cabang"] + cabang_all)

    df_tmp2 = df_tmp[df_tmp["cabang"] == cabang_sel] if cabang_sel != "Semua Cabang" else df_tmp
    sku_all = sorted(df_tmp2["sku"].dropna().unique().tolist())
    sku_sel = f3.selectbox("SKU", ["Semua SKU"] + sku_all)

    df = df_raw.copy()
    if area_sel != "Semua Area":
        df = df[df["area"] == area_sel]
    if cabang_sel != "Semua Cabang":
        df = df[df["cabang"] == cabang_sel]
    if sku_sel != "Semua SKU":
        df = df[df["sku"] == sku_sel]

    if df.empty:
        st.warning("Tidak ada data untuk kombinasi filter ini.")
        st.stop()

hist_df = df[df["is_future"] == 0].copy()
future_df = df[df["is_future"] == 1].copy()

total_fc = float(future_df["pred_qty"].sum()) if not future_df.empty else 0.0
total_act = float(hist_df["qty_actual"].sum()) if not hist_df.empty else 0.0
monthly = hist_df.groupby("periode")["qty_actual"].sum() if not hist_df.empty else pd.Series(dtype=float)
avg_sales = float(monthly.mean()) if not monthly.empty else 0.0
last_month = float(monthly.iloc[-1]) if not monthly.empty else 0.0

c1, c2, c3, c4 = st.columns(4)
clean_metric(c1, "Forecast", f"{total_fc:,.0f}", "Total periode future")
clean_metric(c2, "Aktual", f"{total_act:,.0f}", "Total penjualan")
clean_metric(c3, "Rata-rata / Bulan", f"{avg_sales:,.0f}", "Penjualan per bulan")
clean_metric(c4, "Bulan Terakhir", f"{last_month:,.0f}", "Penjualan bulan terakhir")

st.write("")

df_chart = df.groupby("periode")[["qty_actual", "pred_qty"]].sum().reset_index()
df_chart = df_chart[(df_chart["qty_actual"] > 0) | (df_chart["pred_qty"] > 0)]

st.markdown("#### Perbandingan Penjualan vs Forecast")
if df_chart.empty:
    st.info("Belum ada data untuk ditampilkan.")
else:
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=df_chart["periode"], y=df_chart["qty_actual"], name="Aktual", marker_color="#0f172a"))
    fig_bar.add_trace(go.Bar(x=df_chart["periode"], y=df_chart["pred_qty"], name="Forecast", marker_color="#94a3b8"))
    fig_bar.update_layout(barmode="group", height=320)
    st.plotly_chart(apply_chart_theme(fig_bar), use_container_width=True)

c_left, c_right = st.columns([2, 1])

with c_left:
    st.markdown("#### Tren Volume Gabungan")
    if not df_chart.empty:
        df_trend = df_chart.copy()
        df_trend["total"] = np.where(df_trend["qty_actual"] > 0, df_trend["qty_actual"], df_trend["pred_qty"])
        fig_wave = go.Figure(
            go.Scatter(
                x=df_trend["periode"],
                y=df_trend["total"].rolling(2, min_periods=1).mean(),
                mode="lines",
                line=dict(shape="spline", width=3, color="#3b82f6"),
                fill="tozeroy",
            )
        )
        fig_wave.update_layout(height=320)
        st.plotly_chart(apply_chart_theme(fig_wave), use_container_width=True)
    else:
        st.info("Belum ada data volume.")

with c_right:
    TOP_N = 15
    top_sku = (
        hist_df.groupby("sku", as_index=False)["qty_actual"]
        .sum()
        .sort_values("qty_actual", ascending=False)
        .head(TOP_N)
        if not hist_df.empty
        else pd.DataFrame(columns=["sku", "qty_actual"])
    )

    st.markdown(f"#### Top {min(TOP_N, len(top_sku))} SKU")

    if top_sku.empty:
        st.info("Belum ada penjualan aktual untuk Top SKU.")
    else:
        chart_height = 24 * len(top_sku) + 80
        fig_top = px.bar(top_sku, x="qty_actual", y="sku", orientation="h", text_auto=".2s")
        fig_top.update_traces(marker_color="#1e293b")
        fig_top.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=chart_height,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig_top.update_xaxes(showgrid=False)
        st.plotly_chart(apply_chart_theme(fig_top), use_container_width=True)

st.write("")

st.markdown("### Rekomendasi Order & Penyesuaian Stok")

colh1, colh2 = st.columns([1.2, 1.2])

if not future_df.empty:
    future_periods = future_df["periode"].dt.to_period("M").drop_duplicates().sort_values()
    period_labels = [p.strftime("%b %Y") for p in future_periods]
    default_idx = len(period_labels) - 1
    label_to_period = dict(zip(period_labels, future_periods))

    selected_label = colh1.selectbox(
        "Bulan forecast",
        period_labels,
        index=default_idx,
        help="Pilih bulan dan tahun untuk dasar perhitungan order.",
    )
    selected_period = label_to_period[selected_label]
else:
    selected_period = None
    colh1.info("Belum ada data forecast ke depan.")

status_filter = colh2.selectbox("Status Stok", ["Semua", "Risiko Kekurangan", "Potensi Kelebihan", "Aman"])

cabang_filter = cabang_sel if cabang_sel != "Semua Cabang" else None

with page_loading("Mengambil data stok & policy dari database..."):
    df_stock_raw = load_stok_policy_with_latest_stock(cabang_filter)

if df_stock_raw is None or df_stock_raw.empty:
    st.info("Data stok_policy / latest_stock belum tersedia.")
    st.stop()

if area_sel != "Semua Area":
    df_stock_raw = df_stock_raw[df_stock_raw["area"] == area_sel]
if sku_sel != "Semua SKU":
    df_stock_raw = df_stock_raw[df_stock_raw["sku"] == sku_sel]

if df_stock_raw.empty:
    st.info("Tidak ada data stok untuk filter ini.")
    st.stop()

if selected_period is None:
    st.stop()

# forecast per bulan terpilih
df_future_sel = future_df[future_df["periode"].dt.to_period("M") == selected_period].copy()
df_future_agg = (
    df_future_sel.groupby(["cabang", "sku"], as_index=False)["pred_qty"]
    .sum()
    .rename(columns={"pred_qty": "forecast_total"})
)

df_stock = df_stock_raw.merge(df_future_agg, how="left", on=["cabang", "sku"])
df_stock["forecast_total"] = pd.to_numeric(df_stock["forecast_total"], errors="coerce").fillna(0.0)

# FILTER PENTING: hanya yang ada forecast
df_stock = df_stock[df_stock["forecast_total"] > 0].copy()
if df_stock.empty:
    st.info("Tidak ada SKU yang punya forecast di bulan terpilih untuk filter ini.")
    st.stop()

with page_loading("Mengambil performa model (MAPE) untuk perhitungan alpha..."):
    mape_global, df_mape = get_mape_global_and_per_sku(model_run_id)

base_mape = float(mape_global) if mape_global is not None else 60.0

if df_mape is not None and not df_mape.empty:
    df_stock = df_stock.merge(df_mape, how="left", on=["cabang", "sku"])
    df_stock["mape_used"] = pd.to_numeric(df_stock.get("mape_sku"), errors="coerce").fillna(base_mape)
else:
    df_stock["mape_used"] = base_mape

df_stock["alpha"] = df_stock["mape_used"].apply(map_mape_to_alpha)

for c in ["avg_qty", "max_baru", "last_stock", "forecast_total"]:
    df_stock[c] = pd.to_numeric(df_stock[c], errors="coerce")

df_stock["safety_stock"] = (0.8 * df_stock["max_baru"]).round()
df_stock["target_stock"] = np.maximum(df_stock["safety_stock"], (df_stock["alpha"] * df_stock["forecast_total"])).round()

df_stock["rec_order"] = np.where(
    df_stock["last_stock"].notna(),
    np.maximum(df_stock["target_stock"] - df_stock["last_stock"], 0),
    df_stock["target_stock"],
).round()

df_stock["coverage_month"] = (df_stock["last_stock"] / df_stock["avg_qty"].replace(0, np.nan)).round(2)

cond_short = df_stock["last_stock"].notna() & (df_stock["last_stock"] < df_stock["safety_stock"])
cond_excess = df_stock["last_stock"].notna() & (df_stock["last_stock"] > df_stock["target_stock"])

df_stock["Status"] = "Aman"
df_stock.loc[cond_short, "Status"] = "Risiko Kekurangan"
df_stock.loc[cond_excess, "Status"] = "Potensi Kelebihan"
df_stock.loc[df_stock["last_stock"].isna(), "Status"] = "Stok Tidak Tersedia"

st.divider()

df_view = df_stock.copy()
if status_filter != "Semua":
    df_view = df_view[df_view["Status"] == status_filter]

m1, m2, m3 = st.columns(3)
n_short = int((df_view["Status"] == "Risiko Kekurangan").sum())
n_excess = int((df_view["Status"] == "Potensi Kelebihan").sum())
avg_cov = float(df_view["coverage_month"].mean()) if not pd.isna(df_view["coverage_month"].mean()) else 0.0

m1.metric("SKU Risiko Kekurangan", n_short)
m2.metric("SKU Potensi Kelebihan", n_excess)
m3.metric("Rata-rata Coverage", f"{avg_cov:.2f} bln")

st.markdown("### Visual Prioritas Stok")

colA, colB = st.columns(2)

with colA:
    st.markdown("#### Kekurangan Stok (Prioritas Order)")
    shortage = df_view[df_view["Status"] == "Risiko Kekurangan"].copy()
    shortage = shortage[shortage["rec_order"] > 0]

    if shortage.empty:
        st.info("Tidak ada SKU kekurangan stok.")
    else:
        shortage_g = (
            shortage.groupby(["cabang", "sku"], as_index=False)["rec_order"]
            .sum()
            .sort_values("rec_order", ascending=False)
            .head(15)
        )
        shortage_g["label"] = shortage_g["cabang"] + " · " + shortage_g["sku"]
        shortage_g["rec_text"] = shortage_g["rec_order"].apply(format_si_short)

        fig_s1 = px.bar(shortage_g, x="rec_order", y="label", orientation="h", text="rec_text")
        fig_s1.update_traces(marker_color="#ef4444", textposition="outside")
        fig_s1.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title="Qty Order Disarankan", yaxis_title="Cabang · SKU")
        st.plotly_chart(apply_chart_theme(fig_s1), use_container_width=True)

with colB:
    st.markdown("#### Potensi Kelebihan Stok")
    excess = df_view[df_view["Status"] == "Potensi Kelebihan"].copy()
    excess["excess_qty"] = (excess["last_stock"] - excess["target_stock"]).clip(lower=0)
    excess = excess[excess["excess_qty"] > 0]

    if excess.empty:
        st.info("Tidak ada SKU kelebihan stok.")
    else:
        excess_g = (
            excess.groupby(["cabang", "sku"], as_index=False)["excess_qty"]
            .sum()
            .sort_values("excess_qty", ascending=False)
            .head(15)
        )
        excess_g["label"] = excess_g["cabang"] + " · " + excess_g["sku"]
        excess_g["excess_text"] = excess_g["excess_qty"].apply(format_si_short)

        fig_s2 = px.bar(excess_g, x="excess_qty", y="label", orientation="h", text="excess_text")
        fig_s2.update_traces(marker_color="#22c55e", textposition="outside")
        fig_s2.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title="Perkiraan Qty Berlebih", yaxis_title="Cabang · SKU")
        st.plotly_chart(apply_chart_theme(fig_s2), use_container_width=True)


st.markdown("#### Daftar Rekomendasi Stok")

cols_show = [
    "area", "cabang", "sku",
    "last_stock", "safety_stock", "target_stock",
    "forecast_total", "rec_order",
    "coverage_month",
    "Status",
]

df_display = (
    df_view[cols_show]
    .groupby(["area", "cabang", "sku", "Status"], as_index=False)
    .agg({
        "last_stock": "first",
        "safety_stock": "first",
        "target_stock": "first",
        "forecast_total": "first",
        "rec_order": "first",
        "coverage_month": "first",
    })
    .sort_values(["Status", "rec_order"], ascending=[True, False])
)

if df_display.empty:
    st.info("Tidak ada rekomendasi stok untuk kombinasi filter ini.")
else:
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "area": st.column_config.TextColumn("Area"),
            "cabang": st.column_config.TextColumn("Cabang"),
            "sku": st.column_config.TextColumn("SKU"),
            "last_stock": st.column_config.NumberColumn("Stok Saat Ini", format="%d", help="Jumlah stok terakhir di gudang"),
            "safety_stock": st.column_config.NumberColumn("Safety Stock", format="%d", help="Batas minimum stok aman"),
            "target_stock": st.column_config.NumberColumn("Target Stok", format="%d", help="Level stok yang diinginkan berdasarkan forecast"),
            "forecast_total": st.column_config.NumberColumn("Forecast (Bulan terpilih)", format="%d"),
            "rec_order": st.column_config.NumberColumn("Rekomendasi Order", format="%d"),
            "coverage_month": st.column_config.ProgressColumn(
                "Coverage (Bulan)",
                format="%.2f bln",
                min_value=0,
                max_value=6,
                help="Perkiraan berapa bulan stok cukup (visual dibatasi 6 bulan)",
            ),
            "Status": st.column_config.TextColumn("Status Stok", width="medium"),
        },
    )

    buffer = io.BytesIO()
    df_display.to_excel(buffer, index=False)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()

    st.markdown(
        f'<div style="text-align: right; margin-top: 10px;">'
        f'<a href="data:application/octet-stream;base64,{b64}" download="rekomendasi_stok.xlsx">'
        f'<button class="download-btn">Download Excel Rekomendasi</button></a>'
        f'</div>',
        unsafe_allow_html=True,
    )

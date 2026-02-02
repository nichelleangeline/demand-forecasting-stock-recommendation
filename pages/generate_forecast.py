import datetime as dt
import json
import re

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
from app.services.auth_guard import require_login


def _norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _parse_dt_any(val):
    if val is None:
        return None
    if isinstance(val, dt.datetime):
        return val
    s = str(val).strip()
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        try:
            ts = pd.to_datetime(s, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.to_pydatetime().replace(tzinfo=None)
        except Exception:
            return None


def _get_last_forecast_generated(model_run_id: int):
    key = f"LAST_FORECAST_GEN_{model_run_id}"
    sql = """
        SELECT config_value
        FROM forecast_config
        WHERE config_key = :k
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"k": key}).mappings().fetchone()
    if not row:
        return None
    return _parse_dt_any(row.get("config_value"))


def _set_last_forecast_generated(model_run_id: int, user_id: int | None, when: dt.datetime | None = None):
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


def _get_last_horizon(model_run_id: int) -> int | None:
    key = f"LAST_HORIZON_{model_run_id}"
    sql = """
        SELECT config_value
        FROM forecast_config
        WHERE config_key = :k
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"k": key}).mappings().fetchone()
    if not row:
        return None
    try:
        return int(str(row.get("config_value", "")).strip())
    except Exception:
        return None


def _set_last_horizon(model_run_id: int, user_id: int | None, horizon: int):
    key = f"LAST_HORIZON_{model_run_id}"
    val = str(int(horizon))
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


def _infer_max_lag_from_active_model(active_model: dict) -> int:
    raw = active_model.get("feature_cols_json")
    if not raw:
        return 1
    try:
        cols = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return 1

    max_lag = 0
    for c in cols:
        m = re.match(r"^qty_lag(\d+)$", str(c).strip())
        if m:
            max_lag = max(max_lag, int(m.group(1)))
    return max(1, int(max_lag))


def _mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


@st.cache_data(ttl=10, show_spinner=False)
def _load_forecast_join_snapshot(model_run_id: int) -> pd.DataFrame:
    sql = """
        SELECT
            f.model_run_id,
            f.area,
            f.cabang,
            f.sku,
            f.periode,
            f.qty_actual,
            f.pred_qty,
            f.is_train,
            f.is_test,
            f.is_future,
            f.horizon
        FROM forecast_monthly f
        INNER JOIN model_run_sku s
            ON  s.model_run_id = f.model_run_id
            AND s.cabang      = f.cabang
            AND s.sku         = f.sku
            AND s.eligible_model = 1
        WHERE f.model_run_id = :mid
        ORDER BY f.cabang, f.sku, f.periode
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text(sql),
            conn,
            params={"mid": int(model_run_id)},
            parse_dates=["periode"],
        )
    return df


st.set_page_config(page_title="Forecast Penjualan", layout="wide", initial_sidebar_state="collapsed")

with st.spinner("Menyiapkan halaman..."):
    require_login()
    inject_global_theme()
    render_sidebar_user_and_logout()
    init_loading_css()

if "user" not in st.session_state:
    st.error("Silakan login dulu.")
    st.stop()

user = st.session_state["user"]
user_id = user.get("user_id")


st.markdown(
    """
    <style>
    .header-wrap { margin-top: 6px; }
    .model-card{
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    }
    .model-k{
        font-size: 11px;
        font-weight: 900;
        color: #64748b;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .model-v{
        font-size: 14px;
        font-weight: 950;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .model-line{
        font-size: 12px;
        font-weight: 800;
        color: #475569;
        margin-top: 4px;
    }
    .section-title{
        font-size: 16px;
        font-weight: 950;
        color:#0f172a;
        margin: 8px 0 8px 0;
    }
    .metric-card {
        background:#fff;
        border:1px solid #e5e7eb;
        border-radius:14px;
        padding:14px 16px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        height:100%;
    }
    .metric-title{
        font-size:11px;
        font-weight:900;
        text-transform:uppercase;
        color:#64748b;
        letter-spacing:.08em;
        margin-bottom:6px;
    }
    .metric-value{
        font-size:24px;
        font-weight:950;
        color:#0f172a;
        line-height:1.1;
    }
    .metric-sub{
        font-size:12px;
        color:#64748b;
        margin-top:8px;
        font-weight:700;
    }
    [data-testid="stVerticalBlockBorderWrapper"]{
        border: 1px solid #e5e7eb !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        background: #ffffff !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        min-height: 44px !important;
        box-shadow: none !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span{
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    div[data-testid="stSelectbox"] label{
        color: #0f172a !important;
        font-weight: 900 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] svg{
        fill: #0f172a !important;
        color: #0f172a !important;
    }
    div[role="listbox"]{
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    div[role="option"]{
        color: #0f172a !important;
        background: #ffffff !important;
        font-weight: 800 !important;
    }
    div[role="option"]:hover{
        background: #f1f5f9 !important;
    }
    div[aria-selected="true"]{
        background: #eff6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Mengambil model aktif..."):
    models = get_all_model_runs()

active_model = None
if models:
    for m in models:
        if m.get("active_flag") == 1:
            active_model = m
            break

if not active_model:
    st.warning("Belum ada model aktif.")
    st.stop()

model_run_id = int(active_model["id"])
model_type = str(active_model.get("model_type", "")).upper()

gen_key_req = f"gen_req_{model_run_id}"
gen_key_run = f"gen_run_{model_run_id}"
gen_key_h = f"gen_h_{model_run_id}"

if gen_key_req not in st.session_state:
    st.session_state[gen_key_req] = False
if gen_key_run not in st.session_state:
    st.session_state[gen_key_run] = False
if gen_key_h not in st.session_state:
    st.session_state[gen_key_h] = None

st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
h1, h2 = st.columns([2.2, 1], gap="large")

with h1:
    st.title("Forecast Penjualan")

with h2:
    st.markdown(
        f"""
        <div class="model-card">
          <div class="model-k">Model aktif</div>
          <div class="model-v">{model_type} (ID {model_run_id})</div>
          <div class="model-line">Latih: {active_model.get('train_start')} s/d {active_model.get('train_end')}</div>
          <div class="model-line">Uji: {active_model.get('test_start')} s/d {active_model.get('test_end')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


last_at = _get_last_forecast_generated(model_run_id)
last_h = _get_last_horizon(model_run_id)

default_h = int(last_h) if last_h is not None else 1
default_h = max(1, min(240, default_h))

with st.expander("Generate Forecast Future", expanded=False):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2], gap="large")

    with c1:
        horizon_months = st.number_input(
            "Horizon (bulan)",
            min_value=1,
            max_value=240,
            value=int(st.session_state[gen_key_h] or default_h),
            step=1,
            key=f"horizon_input_{model_run_id}",
            disabled=bool(st.session_state[gen_key_run]),
        )

    with c2:
        st.write("")
        clicked = st.button(
            "Generate & Simpan",
            type="primary",
            use_container_width=True,
            key=f"btn_gen_{model_run_id}",
            disabled=bool(st.session_state[gen_key_run]),
        )

    with c3:
        st.write("")
        reset = st.button(
            "Reset",
            use_container_width=True,
            key=f"btn_gen_reset_{model_run_id}",
        )

    with c4:
        if last_at:
            st.markdown(f"Terakhir generate: {last_at.strftime('%d %b %Y %H:%M')}")
        else:
            st.markdown("Terakhir generate: -")

        if st.session_state[gen_key_run]:
            st.caption("Status: sedang proses. Kalau stuck, tekan Reset.")

    if reset:
        st.session_state[gen_key_req] = False
        st.session_state[gen_key_run] = False
        st.session_state[gen_key_h] = None
        st.cache_data.clear()
        st.rerun()

    if clicked:
        st.session_state[gen_key_h] = int(horizon_months)
        st.session_state[gen_key_req] = True
        st.session_state[gen_key_run] = True
        st.rerun()


if st.session_state[gen_key_req] and st.session_state[gen_key_run]:
    h = int(st.session_state[gen_key_h] or default_h)

    with st.spinner("Menghitung dan menyimpan forecast..."):
        try:
            n_rows = generate_and_store_forecast(model_run_id=model_run_id, horizon_months=h)

            _set_last_forecast_generated(model_run_id, user_id)
            _set_last_horizon(model_run_id, user_id, h)

            st.cache_data.clear()

            st.session_state[gen_key_req] = False
            st.session_state[gen_key_run] = False
            st.session_state[gen_key_h] = None

            st.success(f"Selesai. {int(n_rows):,} baris tersimpan.")
            st.rerun()
        except Exception:
            st.session_state[gen_key_req] = False
            st.session_state[gen_key_run] = False
            st.session_state[gen_key_h] = None
            st.error("Generate gagal. Silakan coba lagi atau hubungi admin.")
            st.stop()

with st.spinner("Memuat data forecast..."):
    df = _load_forecast_join_snapshot(model_run_id)

if df.empty:
    st.warning("Data forecast kosong. Generate dulu.")
    st.stop()

df["cabang"] = _norm_str_series(df["cabang"])
df["sku"] = _norm_str_series(df["sku"])
if "area" in df.columns:
    df["area"] = df["area"].astype(str).str.strip()

df["qty_actual"] = pd.to_numeric(df["qty_actual"], errors="coerce")
df["pred_qty"] = pd.to_numeric(df["pred_qty"], errors="coerce")
df["is_train"] = pd.to_numeric(df["is_train"], errors="coerce").fillna(0).astype(int)
df["is_test"] = pd.to_numeric(df["is_test"], errors="coerce").fillna(0).astype(int)
df["is_future"] = pd.to_numeric(df["is_future"], errors="coerce").fillna(0).astype(int)
df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").fillna(0).astype(int)

has_future = (df["is_future"] == 1).any()

st.markdown('<div class="section-title">Filter</div>', unsafe_allow_html=True)

with st.container(border=True):
    f1, f2, f3, f4 = st.columns(4, gap="large")

    with f1:
        view_mode = st.selectbox("Periode", ["12 bulan terakhir", "Semua histori + future"], index=0)

    with f2:
        if has_future:
            max_future_h = _safe_int(df.loc[df["is_future"] == 1, "horizon"].max(), default=0)
            max_future_h = max(1, max_future_h)
            horizon_options = list(range(1, max_future_h + 1))

            saved_h = _get_last_horizon(model_run_id)
            if saved_h is not None and int(saved_h) in horizon_options:
                default_idx = horizon_options.index(int(saved_h))
            else:
                default_idx = len(horizon_options) - 1

            view_horizon = st.selectbox(
                "Future sampai (bulan)",
                options=horizon_options,
                index=default_idx,
                key=f"view_h_{model_run_id}",
            )
        else:
            view_horizon = None
            st.selectbox("Future sampai (bulan)", options=["-"], index=0, disabled=True)

    with f3:
        cabang_list = sorted(df["cabang"].dropna().unique().tolist())
        cabang_selected = st.selectbox("Cabang", options=cabang_list, index=0)

    with f4:
        sku_list = sorted(df.loc[df["cabang"] == cabang_selected, "sku"].dropna().unique().tolist())
        sku_selected = st.selectbox("Produk", options=sku_list, index=0)

df_view = df[(df["cabang"] == cabang_selected) & (df["sku"] == sku_selected)].copy()

if has_future and view_horizon is not None:
    df_view = df_view[(df_view["is_future"] == 0) | (df_view["horizon"] <= int(view_horizon))]

df_view = df_view.sort_values("periode").reset_index(drop=True)
if df_view.empty:
    st.warning("Data kosong.")
    st.stop()

# hide in-sample prediction until lag complete
max_lag_used = _infer_max_lag_from_active_model(active_model)
first_obs = df_view.loc[df_view["qty_actual"].notna(), "periode"].min()
if pd.notna(first_obs):
    valid_pred_start = (first_obs.to_period("M") + int(max_lag_used)).to_timestamp()
    df_view.loc[(df_view["is_future"] == 0) & (df_view["periode"] < valid_pred_start), "pred_qty"] = np.nan

if view_mode == "12 bulan terakhir":
    last_date = df_view["periode"].max()
    if pd.notna(last_date):
        cutoff = (last_date.to_period("M") - 11).to_timestamp()
        df_view = df_view[df_view["periode"] >= cutoff].copy()

df_view = df_view.sort_values("periode").reset_index(drop=True)

st.markdown('<div class="section-title">Ringkasan</div>', unsafe_allow_html=True)

future_view = df_view[df_view["is_future"] == 1].copy()
total_pred_future = float(future_view["pred_qty"].sum()) if not future_view.empty else 0.0

test_mask = (df_view["is_test"] == 1) & df_view["qty_actual"].notna() & df_view["pred_qty"].notna()
mape_test_val = _mape(df_view.loc[test_mask, "qty_actual"], df_view.loc[test_mask, "pred_qty"]) if test_mask.any() else None

k1, k2 = st.columns(2, gap="large")


def _card(col, title, value, sub):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-title">{title}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


_card(k1, "Total prediksi future", f"{total_pred_future:,.0f}", "Total prediksi sesuai horizon yang dipilih.")
_card(k2, "MAPE test", (f"{mape_test_val:,.1f}%" if mape_test_val is not None else "-"), "Akurasi di periode test (kalau ada).")

st.markdown('<div class="section-title">Grafik</div>', unsafe_allow_html=True)

with st.container(border=True):
    df_chart = df_view[["periode", "qty_actual", "pred_qty"]].copy().sort_values("periode")

    actual = df_chart[df_chart["qty_actual"].notna()][["periode", "qty_actual"]]
    pred = df_chart[df_chart["pred_qty"].notna()][["periode", "pred_qty"]]

    if actual.empty and pred.empty:
        st.info("Tidak ada data grafik.")
    else:
        fig = px.line()

        if not actual.empty:
            fig.add_scatter(
                x=actual["periode"],
                y=actual["qty_actual"],
                mode="lines+markers",
                name="Aktual",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=6, color="#2563eb"),
            )

        if not pred.empty:
            fig.add_scatter(
                x=pred["periode"],
                y=pred["pred_qty"],
                mode="lines+markers",
                name="Prediksi",
                line=dict(color="#f97316", width=3, dash="dash"),
                marker=dict(size=6, color="#f97316"),
            )

        all_x = pd.concat([actual["periode"], pred["periode"]], ignore_index=True).dropna()
        xmin, xmax = all_x.min(), all_x.max()
        pad = pd.Timedelta(days=15)

        fig.update_xaxes(range=[xmin - pad, xmax + pad], fixedrange=True, showgrid=False, tickformat="%b %Y")
        fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#e5e7eb")

        fig.update_layout(
            template="plotly_white",
            height=380,
            hovermode="x unified",
            margin=dict(l=24, r=24, t=24, b=24),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": False,
                "editable": False,
                "staticPlot": False,
            },
        )

st.markdown('<div class="section-title">Tabel</div>', unsafe_allow_html=True)

with st.container(border=True):
    df_download = df_view.copy().sort_values(["cabang", "sku", "periode"])
    csv_bytes = df_download.to_csv(index=False).encode("utf-8-sig")

    col_d1, col_d2 = st.columns([5, 1], gap="large")
    with col_d2:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"forecast_{model_run_id}_{cabang_selected}_{sku_selected}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        df_download.head(1000),
        use_container_width=True,
        hide_index=True,
        column_config={
            "model_run_id": st.column_config.NumberColumn("Model ID"),
            "area": st.column_config.TextColumn("Area"),
            "cabang": st.column_config.TextColumn("Cabang"),
            "sku": st.column_config.TextColumn("Produk"),
            "periode": st.column_config.DateColumn("Periode", format="D MMM YYYY"),
            "qty_actual": st.column_config.NumberColumn("Aktual", format="%.0f"),
            "pred_qty": st.column_config.NumberColumn("Prediksi", format="%.0f"),
            "is_train": st.column_config.CheckboxColumn("Train", disabled=True),
            "is_test": st.column_config.CheckboxColumn("Test", disabled=True),
            "is_future": st.column_config.CheckboxColumn("Future", disabled=True),
            "horizon": st.column_config.NumberColumn("Horizon", format="%d"),
        },
    )

    if len(df_download) > 1000:
        st.caption("Tabel dibatasi 1.000 baris. Download untuk lengkap.")

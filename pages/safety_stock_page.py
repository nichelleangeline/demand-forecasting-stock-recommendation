import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text
from app.db import engine
from app.loading_utils import init_loading_css
from app.services.auth_guard import require_login
from app.services.page_loader import init_page_loader_css
from app.services.stok_policy_service import upsert_stok_policy_from_sales_monthly
from app.ui.theme import inject_global_theme, render_sidebar_user_and_logout

def _fmt_bulan(d: dt.date) -> str:
    return pd.Timestamp(d).strftime("%b %Y")


def _safe_date(x) -> dt.date | None:
    ts = pd.to_datetime(x, errors="coerce")
    return ts.date() if not pd.isna(ts) else None


def _norm_key(s: str) -> str:
    return str(s).strip().upper()


def _eps_denom(arr, eps=1e-8):
    arr = np.asarray(arr, dtype=float)
    return np.maximum(np.abs(arr), eps)


def map_mape_to_alpha(mape):
    if pd.isna(mape):
        return 0.6
    try:
        m = float(mape)
    except Exception:
        return 0.6

    if m < 20:
        return 0.9
    if m < 30:
        return 0.7
    if m < 40:
        return 0.6
    return 0.4

def get_active_model_run_id() -> int | None:
    sql = """
        SELECT id
        FROM model_run
        WHERE active_flag = 1
        ORDER BY trained_at DESC, id DESC
        LIMIT 1
    """
    try:
        with engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        if row and row[0] is not None:
            return int(row[0])
    except Exception:
        return None
    return None


def get_db_summary(active_model_run_id: int | None):
    with engine.begin() as conn:
        n_policy = conn.execute(text("SELECT COUNT(*) FROM stok_policy")).scalar() or 0

        try:
            sql_sync = """
                SELECT COUNT(DISTINCT CONCAT(TRIM(UPPER(ls.cabang)), '||', TRIM(UPPER(ls.sku))))
                FROM latest_stock ls
                INNER JOIN stok_policy sp
                  ON TRIM(UPPER(ls.cabang)) COLLATE utf8mb4_unicode_ci = TRIM(UPPER(sp.cabang)) COLLATE utf8mb4_unicode_ci
                 AND TRIM(UPPER(ls.sku))    COLLATE utf8mb4_unicode_ci = TRIM(UPPER(sp.sku))    COLLATE utf8mb4_unicode_ci
            """
            n_stock = conn.execute(text(sql_sync)).scalar() or 0
        except Exception:
            n_stock = 0

        try:
            if active_model_run_id is None:
                n_months_fc = 0
            else:
                n_months_fc = conn.execute(
                    text(
                        """
                        SELECT COUNT(DISTINCT periode)
                        FROM forecast_monthly
                        WHERE model_run_id = :mid AND is_future = 1
                        """
                    ),
                    {"mid": int(active_model_run_id)},
                ).scalar() or 0
        except Exception:
            n_months_fc = 0

    return {"policy": int(n_policy), "stock": int(n_stock), "fc_months": int(n_months_fc)}


def ensure_latest_stock_table():
    sql = """
    CREATE TABLE IF NOT EXISTS latest_stock (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        area VARCHAR(10) NOT NULL,
        cabang VARCHAR(10) NOT NULL,
        sku VARCHAR(100) NOT NULL,
        last_txn_date DATE NOT NULL,
        last_stock INT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_latest_stock_acs (area, cabang, sku)
    ) ENGINE=InnoDB;
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def upsert_latest_stock_records(records):
    if not records:
        return

    ensure_latest_stock_table()

    with engine.begin() as conn:
        rows = conn.execute(text("SELECT DISTINCT cabang, area FROM sales_monthly")).fetchall()
    cabang_to_area = {str(r[0]).strip().upper(): str(r[1]).strip() for r in rows}

    enriched = []
    for rec in records:
        cab = _norm_key(rec.get("cabang", ""))
        sku = _norm_key(rec.get("sku", ""))
        if not cab or not sku:
            continue

        area_val = cabang_to_area.get(cab, "UNKNOWN")

        raw_stock = rec.get("last_stock", 0)
        try:
            raw_stock = int(float(raw_stock))
        except Exception:
            raw_stock = 0

        last_txn_date = _safe_date(rec.get("last_txn_date", None)) or dt.date.today()

        enriched.append(
            {
                "area": area_val,
                "cabang": cab,
                "sku": sku,
                "last_txn_date": last_txn_date,
                "last_stock": max(0, raw_stock), 
            }
        )

    if not enriched:
        return

    sql = """
        INSERT INTO latest_stock (area, cabang, sku, last_txn_date, last_stock)
        VALUES (:area, :cabang, :sku, :last_txn_date, :last_stock)
        ON DUPLICATE KEY UPDATE
            last_txn_date = VALUES(last_txn_date),
            last_stock    = VALUES(last_stock),
            area          = VALUES(area)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), enriched)


def save_uploaded_stock(df_upload):
    if df_upload is None or df_upload.empty:
        return

    cols = {str(c).lower().strip(): c for c in df_upload.columns}

    col_sku = cols.get("sku")
    col_cabang = cols.get("cabang") or cols.get("location code") or cols.get("location") or cols.get("location_code")
    if not col_sku or not col_cabang:
        raise ValueError("Kolom SKU dan Cabang/Location wajib ada.")

    col_periode = cols.get("periode") or cols.get("posting date") or cols.get("tanggal") or cols.get("date")
    col_stock = (
        cols.get("stok_akhir")
        or cols.get("last_stock")
        or cols.get("stock")
        or cols.get("qty")
        or cols.get("quantity")
    )
    if not col_stock:
        raise ValueError("Kolom stok tidak ketemu. Pakai salah satu: stok_akhir / last_stock / stock / qty / quantity")

    df = df_upload.copy()

    df["cab_clean"] = df[col_cabang].astype(str).str.strip().str.upper()
    if str(col_cabang).lower().strip() in ["location code", "location_code", "location"]:
        df["cab_clean"] = df["cab_clean"].str[:3]

    df["sku_clean"] = df[col_sku].astype(str).str.strip().str.upper()
    df["stok_clean"] = pd.to_numeric(df[col_stock], errors="coerce").fillna(0).astype(int)

    if col_periode:
        df["date_clean"] = pd.to_datetime(df[col_periode], errors="coerce").dt.date
    else:
        df["date_clean"] = dt.date.today()

    records = []
    for _, row in df.dropna(subset=["date_clean"]).iterrows():
        records.append(
            {
                "cabang": row["cab_clean"],
                "sku": row["sku_clean"],
                "last_stock": int(row["stok_clean"]),
                "last_txn_date": row["date_clean"],
            }
        )

    upsert_latest_stock_records(records)


def load_stok_policy_with_latest_stock(cabang_filter=None):
    sql = """
        SELECT
            ls.area,
            TRIM(UPPER(sp.cabang)) AS cabang,
            TRIM(UPPER(sp.sku)) AS sku,
            sp.avg_qty,
            sp.max_baru,
            ls.last_txn_date,
            ls.last_stock
        FROM stok_policy sp
        LEFT JOIN latest_stock ls
          ON TRIM(UPPER(sp.cabang)) COLLATE utf8mb4_unicode_ci = TRIM(UPPER(ls.cabang)) COLLATE utf8mb4_unicode_ci
         AND TRIM(UPPER(sp.sku))    COLLATE utf8mb4_unicode_ci = TRIM(UPPER(ls.sku))    COLLATE utf8mb4_unicode_ci
        WHERE 1=1
    """
    params = {}
    if cabang_filter:
        sql += " AND TRIM(UPPER(sp.cabang)) COLLATE utf8mb4_unicode_ci = :cabang COLLATE utf8mb4_unicode_ci "
        params["cabang"] = _norm_key(cabang_filter)

    sql += " ORDER BY cabang, sku"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params, parse_dates=["last_txn_date"])

    df["last_stock"] = pd.to_numeric(df.get("last_stock"), errors="coerce")
    df["max_baru"] = pd.to_numeric(df.get("max_baru"), errors="coerce")
    df["avg_qty"] = pd.to_numeric(df.get("avg_qty"), errors="coerce")
    return df


def get_future_period_list(active_model_run_id: int, cabang_filter=None):
    sql = """
        SELECT DISTINCT periode
        FROM forecast_monthly
        WHERE model_run_id = :mid
          AND is_future = 1
          AND periode IS NOT NULL
    """
    params = {"mid": int(active_model_run_id)}
    if cabang_filter:
        sql += " AND TRIM(UPPER(cabang)) COLLATE utf8mb4_unicode_ci = :cabang COLLATE utf8mb4_unicode_ci "
        params["cabang"] = _norm_key(cabang_filter)

    sql += " ORDER BY periode"

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    periods = []
    for r in rows:
        d = _safe_date(r[0])
        if d:
            periods.append(d)

    return sorted(list(dict.fromkeys(periods)))


def load_forecast_for_month(active_model_run_id: int, target_per: dt.date, cabang_filter=None) -> pd.DataFrame:
    sql = """
        SELECT
            TRIM(UPPER(cabang)) AS cabang,
            TRIM(UPPER(sku)) AS sku,
            SUM(pred_qty) AS pred_qty
        FROM forecast_monthly
        WHERE model_run_id = :mid
          AND is_future = 1
          AND periode IS NOT NULL
          AND DATE(periode) = :p
    """
    params = {"mid": int(active_model_run_id), "p": target_per}

    if cabang_filter:
        sql += " AND TRIM(UPPER(cabang)) COLLATE utf8mb4_unicode_ci = :cabang COLLATE utf8mb4_unicode_ci "
        params["cabang"] = _norm_key(cabang_filter)

    sql += " GROUP BY TRIM(UPPER(cabang)), TRIM(UPPER(sku)) "

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["cabang", "sku", "pred_qty"])

    df["pred_qty"] = pd.to_numeric(df["pred_qty"], errors="coerce").fillna(0.0).astype(float)
    return df

def _try_load_mape_from_model_tables(active_model_run_id: int):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT global_test_mape FROM model_run WHERE id = :mid"),
            {"mid": int(active_model_run_id)},
        ).fetchone()

    if not row:
        return None

    mape_global = row[0]
    if mape_global is None:
        mape_global = None
    else:
        try:
            mape_global = float(mape_global)
        except Exception:
            mape_global = None

    sql = """
        SELECT
            TRIM(UPPER(cabang)) AS cabang,
            TRIM(UPPER(sku)) AS sku,
            test_mape AS mape_sku
        FROM model_run_sku
        WHERE model_run_id = :mid
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"mid": int(active_model_run_id)})

    if df is None:
        df = pd.DataFrame()

    if mape_global is None:
        return None

    if df.empty:
        return float(mape_global), pd.DataFrame(columns=["cabang", "sku", "mape_sku"])

    df["mape_sku"] = pd.to_numeric(df["mape_sku"], errors="coerce")
    df = df.dropna(subset=["cabang", "sku"])
    return float(mape_global), df[["cabang", "sku", "mape_sku"]]


def _load_mape_from_forecast_monthly_fallback(active_model_run_id: int):
    sql = """
        SELECT
            TRIM(UPPER(cabang)) AS cabang,
            TRIM(UPPER(sku)) AS sku,
            qty_actual,
            pred_qty
        FROM forecast_monthly
        WHERE model_run_id = :mid
          AND is_test = 1
          AND qty_actual IS NOT NULL
          AND pred_qty IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"mid": int(active_model_run_id)})

    if df.empty:
        return 30.0, pd.DataFrame(columns=["cabang", "sku", "mape_sku"])

    df["qty_actual"] = pd.to_numeric(df["qty_actual"], errors="coerce")
    df["pred_qty"] = pd.to_numeric(df["pred_qty"], errors="coerce")
    df = df.dropna(subset=["qty_actual", "pred_qty"])
    if df.empty:
        return 30.0, pd.DataFrame(columns=["cabang", "sku", "mape_sku"])

    y = df["qty_actual"].to_numpy(float)
    p = df["pred_qty"].to_numpy(float)
    denom = _eps_denom(y)
    mape_global = float(np.mean(np.abs(y - p) / denom) * 100.0)

    def agg_mape(g):
        y2 = g["qty_actual"].to_numpy(float)
        p2 = g["pred_qty"].to_numpy(float)
        denom2 = _eps_denom(y2)
        return float(np.mean(np.abs(y2 - p2) / denom2) * 100.0)

    g = (
        df.groupby(["cabang", "sku"], as_index=False)
        .apply(lambda x: pd.Series({"mape_sku": agg_mape(x)}))
        .reset_index(drop=True)
    )

    return mape_global, g


@st.cache_data(ttl=300)
def load_mape_maps(active_model_run_id: int):
    try:
        out = _try_load_mape_from_model_tables(active_model_run_id)
        if out is not None:
            return out
    except Exception:
        pass

    try:
        return _load_mape_from_forecast_monthly_fallback(active_model_run_id)
    except Exception:
        return 30.0, pd.DataFrame(columns=["cabang", "sku", "mape_sku"])

def render_kpi_box(label, value, info_text, color="#1E88E5"):
    st.markdown(
        f"""
        <div style="
            background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #E0E0E0;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05); min-height: 100px; margin-bottom: 20px;
        ">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="color: #757575; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">
                    {label}
                </div>
                <div title="{info_text}" style="cursor: help; color: #1E88E5; font-size: 14px; font-weight: bold;">
                    ⓘ
                </div>
            </div>
            <div style="color: {color}; font-size: 26px; font-weight: bold; margin-top: 8px;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_upsert_policy():
    upsert_stok_policy_from_sales_monthly(None)

st.set_page_config(page_title="Logistik - Safety Stock", layout="wide")
init_page_loader_css()

require_login()
inject_global_theme()
render_sidebar_user_and_logout()
init_loading_css()

active_model_run_id = get_active_model_run_id()

st.title("Manajemen Stok dan Rekomendasi Order")

if active_model_run_id is None:
    st.error("Belum ada model aktif. Minta Admin aktifkan model dulu di Riwayat Model.")
    st.stop()

summary = get_db_summary(active_model_run_id)

tab_setup, tab_order = st.tabs(["SETUP DATA", "REKOMENDASI ORDER"])

with engine.connect() as conn:
    try:
        res = conn.execute(text("SELECT DISTINCT cabang FROM stok_policy ORDER BY cabang")).fetchall()
        cabs_list = [r[0] for r in res] if res else []
    except Exception:
        cabs_list = []

with tab_setup:
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(
            "### Update Target Kebijakan Stok\n"
            "Hitung ulang batas aman stok berdasarkan histori penjualan."
        )
        with c2:
            st.write("##")
            if st.button("HITUNG TARGET", use_container_width=True):
                try:
                    with st.spinner("Sedang menghitung target stok..."):
                        run_upsert_policy()
                    st.toast("Target stok berhasil dihitung ulang.", icon="✅")
                    st.rerun()
                except Exception:
                    st.error("Gagal menghitung target. Coba ulang.")

    st.write("")

    with st.container(border=True):
        st.markdown("### Unggah Data Stok Akhir")
        up_file = st.file_uploader("Upload file gudang", type=["xlsx", "xls", "csv"], label_visibility="collapsed")
        if up_file and st.button("PROSES UNGGAH FILE", use_container_width=True):
            try:
                with st.spinner("Memproses file..."):
                    if up_file.name.lower().endswith(".csv"):
                        df_up = pd.read_csv(up_file)
                    else:
                        df_up = pd.read_excel(up_file)

                    save_uploaded_stock(df_up)

                st.toast("Data stok berhasil diperbarui.", icon="✅")
                st.rerun()
            except Exception:
                st.error("File tidak bisa diproses. Cek kolom: cabang/location, sku, stok, dan tanggal.")

    st.write("")

    with st.container(border=True):
        st.markdown("### Perbarui Stok Manual")
        sel_c_edit = st.selectbox("Pilih Cabang", ["Pilih Cabang"] + cabs_list, label_visibility="collapsed")
        if sel_c_edit != "Pilih Cabang":
            df_m = load_stok_policy_with_latest_stock(sel_c_edit)

            if df_m.empty:
                st.info("Data policy kosong untuk cabang ini. Jalankan 'Hitung Target' dulu.")
            else:
                editable = df_m[["cabang", "sku", "last_stock", "last_txn_date"]].copy()
                editable["last_stock"] = pd.to_numeric(editable["last_stock"], errors="coerce").fillna(0).astype(int)
                editable["last_stock"] = np.maximum(editable["last_stock"], 0)  # safety: jangan simpan negatif

                edited = st.data_editor(
                    editable,
                    column_config={
                        "last_stock": st.column_config.NumberColumn("Stok Akhir", format="%d"),
                        "last_txn_date": st.column_config.DateColumn("Tanggal Stok"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
                if st.button("SIMPAN MANUAL", use_container_width=True):
                    try:
                        with st.spinner("Menyimpan stok..."):
                            upsert_latest_stock_records(edited.to_dict("records"))
                        st.toast("Perubahan stok tersimpan.", icon="✅")
                        st.rerun()
                    except Exception:
                        st.error("Gagal simpan stok. Coba ulang.")

with tab_order:
    if summary["policy"] == 0:
        st.warning("Jalankan 'Hitung Target' terlebih dahulu di tab SETUP DATA.")
        st.stop()

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            sel_cab = st.selectbox("Cabang", ["SEMUA"] + cabs_list)

        cab_filter = None if sel_cab == "SEMUA" else sel_cab

        with f2:
            p_list = get_future_period_list(active_model_run_id, cab_filter)
            if not p_list:
                st.selectbox("Target Bulan", ["Belum ada forecast future"], disabled=True)
                sel_per = None
            else:
                sel_per = st.selectbox("Target Bulan", p_list, format_func=_fmt_bulan)

        with f3:
            sel_status = st.selectbox("Kondisi", ["Semua", "Risiko Kekurangan", "Potensi Kelebihan", "Aman"])

        with f4:
            sku_opts = ["SEMUA SKU"]
            if sel_per:
                df_fc_month = load_forecast_for_month(active_model_run_id, sel_per, cab_filter)
                if not df_fc_month.empty:
                    df_fc_month["pred_qty"] = pd.to_numeric(df_fc_month["pred_qty"], errors="coerce").fillna(0.0)
                    df_fc_month = df_fc_month[df_fc_month["pred_qty"] > 0].copy()
                    if not df_fc_month.empty:
                        sku_opts += sorted(df_fc_month["sku"].dropna().astype(str).unique().tolist())

            sel_sku = st.selectbox("Pilih SKU", sku_opts)

    if not sel_per:
        st.stop()

    show_detail = st.checkbox("Tampilkan detail perhitungan", value=False)

    try:
        with st.spinner("Menyiapkan rekomendasi order..."):
            df_p = load_stok_policy_with_latest_stock(cab_filter)
            if df_p.empty:
                st.info("Data policy kosong. Jalankan Hitung Target dulu.")
                st.stop()

            df_fc = load_forecast_for_month(active_model_run_id, sel_per, cab_filter)
            if df_fc.empty:
                st.info("Tidak ada forecast untuk bulan yang dipilih.")
                st.stop()

            df = df_p.merge(df_fc, on=["cabang", "sku"], how="inner")
            df["pred_qty"] = pd.to_numeric(df["pred_qty"], errors="coerce").fillna(0.0)
            df = df[df["pred_qty"] > 0].copy()
            if df.empty:
                st.info("Tidak ada SKU dengan forecast positif di bulan ini.")
                st.stop()

            mape_global, df_mape = load_mape_maps(active_model_run_id)
            if df_mape is not None and not df_mape.empty:
                df = df.merge(df_mape, on=["cabang", "sku"], how="left")
                df["mape_used"] = df["mape_sku"].fillna(float(mape_global))
            else:
                df["mape_used"] = float(mape_global)

            df["alpha"] = df["mape_used"].apply(map_mape_to_alpha)

            df["max_baru"] = pd.to_numeric(df.get("max_baru"), errors="coerce").fillna(0.0)
            df["last_stock"] = pd.to_numeric(df.get("last_stock"), errors="coerce")

            df["safety_stock"] = (0.8 * df["max_baru"]).fillna(0.0).round().astype(int)
            df["stock_minimum"] = df["safety_stock"].astype(int)

            # Target stok = max(minimum, alpha*forecast)
            df["target_stock"] = np.maximum(
                df["stock_minimum"].to_numpy(int),
                np.round(df["alpha"].to_numpy(float) * df["pred_qty"].to_numpy(float)).astype(int),
            )

            # Saran order = target - stok sekarang
            df["saran_order"] = np.where(
                df["last_stock"].notna(),
                np.maximum(df["target_stock"] - df["last_stock"], 0),
                df["target_stock"],
            ).round().astype(int)

            # Gap vs minimum = stok - minimum (negatif berarti kurang dari minimum)
            df["gap_vs_minimum"] = np.where(
                df["last_stock"].notna(),
                (df["last_stock"] - df["stock_minimum"]).astype(float),
                np.nan,
            )

            def get_status(row):
                stock = row["last_stock"]
                if pd.isna(stock):
                    return "DATA STOK KOSONG"
                if stock <= 0:
                    return "HABIS"
                if stock < row["stock_minimum"]:
                    return "KRITIS"
                if stock > (row["target_stock"] * 1.5):
                    return "OVERSTOCK"
                return "AMAN"

            df["Status"] = df.apply(get_status, axis=1)

            if sel_status != "Semua":
                status_map = {
                    "Risiko Kekurangan": ["HABIS", "KRITIS"],
                    "Potensi Kelebihan": ["OVERSTOCK"],
                    "Aman": ["AMAN"],
                }
                df = df[df["Status"].isin(status_map.get(sel_status, []))]

            if sel_sku != "SEMUA SKU":
                df = df[df["sku"] == sel_sku]

            if df.empty:
                st.info("Tidak ada data sesuai filter.")
                st.stop()

    except Exception:
        st.error("Gagal memproses rekomendasi. Coba refresh halaman.")
        st.stop()

    st.write("")
    r1, r2, r3 = st.columns(3)
    with r1:
        render_kpi_box("Rencana Order", f"{int(df['saran_order'].sum()):,} Unit", "Total barang disarankan beli.")
    with r2:
        render_kpi_box(
            "Butuh Segera",
            f"{len(df[df['Status'].isin(['HABIS', 'KRITIS'])]):,} SKU",
            "Barang stok kritis atau habis.",
            color="#D32F2F",
        )
    with r3:
        render_kpi_box(
            "SKU Aktif",
            f"{len(df):,} SKU",
            "SKU yang punya forecast > 0 di bulan terpilih.",
            color="#388E3C",
        )

    def _style_text(val):
        if val == "HABIS":
            return "color: #8b0000; font-weight: bold;"
        if val == "KRITIS":
            return "color: #ff4b4b; font-weight: bold;"
        if val == "OVERSTOCK":
            return "color: #ffa500; font-weight: bold;"
        if val == "AMAN":
            return "color: #28a745; font-weight: bold;"
        if val == "DATA STOK KOSONG":
            return "color: #6d6d6d; font-weight: bold;"
        return ""

    df_show = df.copy()
    df_show["Forecast"] = df_show["pred_qty"]

    df_show = df_show.rename(
        columns={
            "cabang": "Cabang",
            "sku": "SKU",
            "Status": "Kondisi",
            "last_stock": "Stok",
            "stock_minimum": "Stock Minimum",
            "gap_vs_minimum": "Gap vs Minimum",
            "mape_used": "MAPE%",
            "alpha": "Alpha",
            "saran_order": "Order",
        }
    )

    show_cols = ["Cabang", "SKU", "Kondisi", "Stok", "Stock Minimum", "Order"]

    if show_detail:
        show_cols = [
            "Cabang",
            "SKU",
            "Kondisi",
            "Stok",
            "Stock Minimum",
            "Gap vs Minimum",
            "Forecast",
            "MAPE%",
            "Alpha",
            "Order",
        ]

    show_cols = [c for c in show_cols if c in df_show.columns]

    col_cfg = {
        "Stok": st.column_config.NumberColumn(format="%d"),
        "Stock Minimum": st.column_config.NumberColumn(format="%d"),
        "Order": st.column_config.NumberColumn(format="%d"),
    }
    if "Forecast" in show_cols:
        col_cfg["Forecast"] = st.column_config.NumberColumn(format="%d")
    if "MAPE%" in show_cols:
        col_cfg["MAPE%"] = st.column_config.NumberColumn(format="%.1f")
    if "Alpha" in show_cols:
        col_cfg["Alpha"] = st.column_config.NumberColumn(format="%.2f")
    if "Gap vs Minimum" in show_cols:
        col_cfg["Gap vs Minimum"] = st.column_config.NumberColumn(format="%.0f")

    st.dataframe(
        df_show[show_cols].style.applymap(_style_text, subset=["Kondisi"]),
        column_config=col_cfg,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Laporan Order (CSV)",
        df_show[show_cols].to_csv(index=False).encode("utf-8"),
        f"order_{sel_per}.csv",
        "text/csv",
        use_container_width=True,
    )

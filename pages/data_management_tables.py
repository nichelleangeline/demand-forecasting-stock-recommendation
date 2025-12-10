# pages/data_management_tables.py

import streamlit as st
import pandas as pd
from sqlalchemy import text

from app.db import engine
from app.services.external_upload_service import handle_external_upload
from app.loading_utils import init_loading_css, action_with_loader
from app.ui.theme import inject_global_theme, render_sidebar_user_and_logout
from app.services.auth_guard import require_login
from app.services.page_loader import init_page_loader_css, page_loading
from app.services.sales_upload_service import (
    preprocess_sales_and_stock_df,
    save_sales_monthly_to_db,
    save_latest_stock_to_db,
)

# Konfigurasi halaman
st.set_page_config(
    page_title="Data Management",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Loader CSS global
init_page_loader_css()

# Setup awal pakai overlay
with page_loading("Menyiapkan halaman Data Management..."):
    require_login()
    inject_global_theme()
    render_sidebar_user_and_logout()
    init_loading_css()

# Cek user di session
if "user" not in st.session_state:
    st.error("Silakan login terlebih dahulu.")
    st.stop()

user = st.session_state["user"]
user_id = user.get("user_id")
role = user.get("role", "user")

# Kalau mau batasi per area, isi list-nya
allowed_areas = None  # Admin akses semua area

# Styling input dan tombol di konten utama
st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        color: #111827 !important;
    }
    
    input[data-baseweb="input"] {
        color: #111827 !important;
    }

    ::placeholder {
        color: #9CA3AF !important;
        opacity: 1;
    }
    
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }

    .main .block-container .stButton > button {
        min-width: 170px;
        padding: 0.45rem 1.4rem;
        border-radius: 999px;
        font-weight: 500;
        font-size: 0.95rem;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Data Management: Penjualan & Eksternal")
st.caption("Upload dan koreksi data penjualan bulanan serta faktor eksternal.")

# State kecil untuk upload dan konfirmasi hapus
if "sales_upload_version" not in st.session_state:
    st.session_state["sales_upload_version"] = 0

if "ext_upload_version" not in st.session_state:
    st.session_state["ext_upload_version"] = 0

if "ext_confirm_delete_stage" not in st.session_state:
    st.session_state["ext_confirm_delete_stage"] = 0

tab_penjualan, tab_external = st.tabs(["Penjualan Bulanan", "Data Eksternal"])


# TAB 1: PENJUALAN BULANAN
with tab_penjualan:
    st.markdown("### Manajemen Data Penjualan")

    # Upload transaksi harian → agregasi bulanan
    with st.container(border=True):
        st.markdown("Import Data Baru")
        st.info(
            "Format upload: data transaksi harian (.xlsx / .csv). "
            "Sistem mengubah otomatis ke data bulanan. "
            "Kolom wajib: Posting Date, Location Code, SKU, Quantity."
        )

        c_file, c_act = st.columns([3, 1])
        mode = "harian"
        sales_upload_key = f"upload_penjualan_v{st.session_state['sales_upload_version']}"

        with c_file:
            uploaded = st.file_uploader(
                "Pilih File",
                type=["xlsx", "xls", "csv"],
                key=sales_upload_key,
                label_visibility="collapsed",
            )

        if uploaded is not None:

            def process_sales_upload():
                if uploaded.name.lower().endswith(".csv"):
                    raw_df = pd.read_csv(uploaded)
                else:
                    raw_df = pd.read_excel(uploaded)

                df_monthly, df_latest = preprocess_sales_and_stock_df(raw_df, mode=mode)

                if allowed_areas:
                    df_monthly = df_monthly[df_monthly["area"].isin(allowed_areas)].copy()
                    if not df_latest.empty:
                        df_latest = df_latest[df_latest["area"].isin(allowed_areas)].copy()

                n_monthly = save_sales_monthly_to_db(
                    df_monthly=df_monthly,
                    source_filename=uploaded.name,
                    uploaded_by=user_id,
                )

                n_latest = 0
                if not df_latest.empty:
                    n_latest = save_latest_stock_to_db(
                        df_latest=df_latest,
                        source_filename=uploaded.name,
                        uploaded_by=user_id,
                    )

                st.success(
                    f"Proses selesai. {n_monthly} baris data penjualan disimpan, "
                    f"{n_latest} data stok diperbarui."
                )

                with st.expander("Lihat preview hasil"):
                    st.markdown("Agregat bulanan:")
                    st.dataframe(df_monthly.head(50), use_container_width=True)
                    if not df_latest.empty:
                        st.markdown("Stok terakhir:")
                        st.dataframe(df_latest.head(50), use_container_width=True)

                st.session_state["sales_upload_version"] += 1

            with c_act:
                st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                action_with_loader(
                    key="process_sales_upload",
                    button_label="Proses Upload",
                    message="Sedang memproses data...",
                    fn=process_sales_upload,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter + editor data penjualan bulanan
    with st.container(border=True):
        st.markdown("Filter & Edit Data Penjualan Bulanan")

        with page_loading("Mengambil filter cabang & periode penjualan..."):
            with engine.begin() as conn:
                if allowed_areas:
                    rows_cabang = conn.execute(
                        text(
                            "SELECT DISTINCT cabang FROM sales_monthly "
                            "WHERE area IN :areas ORDER BY cabang"
                        ),
                        {"areas": tuple(allowed_areas)},
                    ).fetchall()
                    row_minmax = conn.execute(
                        text(
                            "SELECT MIN(periode), MAX(periode) "
                            "FROM sales_monthly WHERE area IN :areas"
                        ),
                        {"areas": tuple(allowed_areas)},
                    ).fetchone()
                else:
                    rows_cabang = conn.execute(
                        text("SELECT DISTINCT cabang FROM sales_monthly ORDER BY cabang")
                    ).fetchall()
                    row_minmax = conn.execute(
                        text("SELECT MIN(periode), MAX(periode) FROM sales_monthly")
                    ).fetchone()

        cabang_list = [r[0] for r in rows_cabang]
        min_p, max_p = row_minmax[0], row_minmax[1]

        if not cabang_list or min_p is None:
            st.warning("Database penjualan kosong.")
        else:
            col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
            with col_f1:
                selected_cabang = st.selectbox(
                    "Cabang",
                    ["ALL"] + cabang_list,
                    index=0,
                    key="filter_cab_sales",
                    help="Pilih cabang yang ingin dicek datanya.",
                )
            with col_f2:
                start_date = st.date_input(
                    "Periode Mulai",
                    value=min_p,
                    min_value=min_p,
                    max_value=max_p,
                    key="f_start_sales",
                    help="Bulan awal data yang ingin ditampilkan.",
                )
            with col_f3:
                end_date = st.date_input(
                    "Periode Sampai",
                    value=max_p,
                    min_value=min_p,
                    max_value=max_p,
                    key="f_end_sales",
                    help="Bulan akhir data yang ingin ditampilkan.",
                )

            if start_date > end_date:
                st.error("Periode mulai tidak boleh lebih besar dari akhir.")
            else:
                query = """
                    SELECT area, cabang, sku, periode, qty
                    FROM sales_monthly
                    WHERE periode BETWEEN :start AND :end
                """
                params = {"start": start_date, "end": end_date}

                if allowed_areas:
                    query += " AND area IN :areas"
                    params["areas"] = tuple(allowed_areas)
                if selected_cabang != "ALL":
                    query += " AND cabang = :cabang"
                    params["cabang"] = selected_cabang

                with page_loading("Mengambil data penjualan dari database..."):
                    with engine.begin() as conn:
                        df_penj = pd.read_sql(text(query), conn, params=params)

                if df_penj.empty:
                    st.info("Data tidak ditemukan untuk filter yang dipilih.")
                else:
                    df_penj["periode"] = pd.to_datetime(df_penj["periode"]).dt.date
                    all_skus = sorted(df_penj["sku"].astype(str).unique().tolist())

                    sel_skus = st.multiselect(
                        "Filter SKU (opsional)",
                        options=all_skus,
                        placeholder="Ketik untuk cari SKU tertentu...",
                        help="Kosongkan kalau ingin menampilkan semua SKU.",
                    )

                    if sel_skus:
                        df_view = df_penj[df_penj["sku"].astype(str).isin(sel_skus)].reset_index(drop=True)
                    else:
                        df_view = df_penj.reset_index(drop=True)

                    st.divider()
                    st.markdown(f"Tabel Data Penjualan Bulanan ({len(df_view)} baris)")

                    edited_penj = st.data_editor(
                        df_view,
                        num_rows="fixed",
                        use_container_width=True,
                        height=400,
                        column_config={
                            "area": st.column_config.TextColumn(
                                "Area",
                                disabled=True,
                                help="Area / wilayah cabang.",
                            ),
                            "cabang": st.column_config.TextColumn(
                                "Cabang",
                                disabled=True,
                                help="Kode cabang penjualan.",
                            ),
                            "sku": st.column_config.TextColumn(
                                "SKU",
                                disabled=True,
                                help="Kode barang.",
                            ),
                            "periode": st.column_config.DateColumn(
                                "Periode (Bulan)",
                                disabled=True,
                                format="MMM YYYY",
                                help="Bulan penjualan.",
                            ),
                            "qty": st.column_config.NumberColumn(
                                "Qty Terjual",
                                step=1,
                                help="Jumlah barang terjual per bulan.",
                            ),
                        },
                        key="editor_sales_main",
                    )

                    def save_penjualan():
                        try:
                            records = []
                            for _, row in edited_penj.iterrows():
                                records.append(
                                    {
                                        "area": row["area"],
                                        "cabang": row["cabang"],
                                        "sku": str(row["sku"]),
                                        "periode": row["periode"],
                                        "qty": float(row["qty"]),
                                    }
                                )

                            with engine.begin() as conn:
                                conn.execute(
                                    text(
                                        """
                                        INSERT INTO sales_monthly (area, cabang, sku, periode, qty)
                                        VALUES (:area, :cabang, :sku, :periode, :qty)
                                        ON DUPLICATE KEY UPDATE
                                            qty = VALUES(qty),
                                            area = VALUES(area)
                                        """
                                    ),
                                    records,
                                )

                            st.success(f"Berhasil menyimpan {len(records)} baris data.")
                            try:
                                st.toast("Perubahan tersimpan.", icon="✅")
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"Gagal menyimpan: {e}")

                    c_spc, c_sv = st.columns([4, 1])
                    with c_sv:
                        action_with_loader(
                            key="btn_save_sales",
                            button_label="Simpan Perubahan",
                            message="Menyimpan...",
                            fn=save_penjualan,
                        )


# TAB 2: DATA EKSTERNAL
with tab_external:
    st.markdown("### Manajemen Data Eksternal")

    # Upload data eksternal
    with st.container(border=True):
        st.markdown("Import Data Eksternal")
        
        jenis_map = {
            "Event / Gathering": "event",
            "Hari Libur Nasional": "holiday",
            "Curah Hujan (Area Surabaya)": "rainfall_16c",
        }

        col_rad, col_upl, col_btn = st.columns([2, 2, 1])
        with col_rad:
            jenis_label = st.radio(
                "Jenis Data",
                list(jenis_map.keys()),
                key="rad_ext_type",
                help="Pilih jenis faktor eksternal yang akan di-upload.",
            )
            jenis_kode = jenis_map[jenis_label]
        
        with col_upl:
            ext_key = f"ext_up_v{st.session_state['ext_upload_version']}"
            uploaded_ext = st.file_uploader(
                "Upload File (.xlsx / .csv)",
                type=["xlsx", "xls", "csv"],
                key=ext_key,
            )

        if uploaded_ext:

            def process_ext():
                try:
                    df_up, n_rows = handle_external_upload(uploaded_ext, jenis_kode, user_id)
                    st.success(f"Sukses. {n_rows} baris data eksternal tersimpan.")
                    with st.expander("Preview data"):
                        st.dataframe(df_up.head(50), use_container_width=True)
                    st.session_state["ext_upload_version"] += 1
                except Exception as e:
                    st.error(f"Error: {e}")

            with col_btn:
                st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
                action_with_loader(
                    key="act_proc_ext",
                    button_label="Proses Upload",
                    message="Menyimpan...",
                    fn=process_ext,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter + editor data eksternal
    with st.container(border=True):
        st.markdown("Filter & Edit Data Eksternal")

        def get_ext_cabang(areas):
            sql = "SELECT DISTINCT cabang FROM external_data"
            params = {}
            if areas:
                sql += " WHERE area IN :areas"
                params["areas"] = tuple(areas)
            sql += " ORDER BY cabang"
            with engine.begin() as conn:
                res = conn.execute(text(sql), params).fetchall()
            return [r[0] for r in res]

        def get_ext_period(areas):
            sql = "SELECT MIN(periode), MAX(periode) FROM external_data"
            params = {}
            if areas:
                sql += " WHERE area IN :areas"
                params["areas"] = tuple(areas)
            with engine.begin() as conn:
                res = conn.execute(text(sql), params).fetchone()
            return res[0], res[1]

        with page_loading("Mengambil daftar cabang & periode data eksternal..."):
            cabs_ext = get_ext_cabang(allowed_areas)
            min_pe, max_pe = get_ext_period(allowed_areas)

        if not cabs_ext or min_pe is None:
            st.warning("Data eksternal kosong.")
        else:
            cf1, cf2, cf3 = st.columns([1, 1, 1])
            with cf1:
                sel_cab_ext = st.selectbox(
                    "Cabang",
                    ["ALL"] + cabs_ext,
                    key="ext_sel_cab",
                    help="Pilih cabang yang ingin dilihat datanya.",
                )
            with cf2:
                start_ext = st.date_input(
                    "Periode Mulai",
                    value=min_pe,
                    min_value=min_pe,
                    max_value=max_pe,
                    key="ext_sd",
                    help="Periode awal data eksternal.",
                )
            with cf3:
                end_ext = st.date_input(
                    "Periode Sampai",
                    value=max_pe,
                    min_value=min_pe,
                    max_value=max_pe,
                    key="ext_ed",
                    help="Periode akhir data eksternal.",
                )

            if start_ext > end_ext:
                st.error("Tanggal mulai tidak valid.")
            else:
                q_ext = """
                    SELECT cabang, area, periode, event_flag, holiday_count, rainfall
                    FROM external_data
                    WHERE periode BETWEEN :s AND :e
                """
                p_ext = {"s": start_ext, "e": end_ext}
                
                if allowed_areas:
                    q_ext += " AND area IN :areas"
                    p_ext["areas"] = tuple(allowed_areas)
                if sel_cab_ext != "ALL":
                    q_ext += " AND cabang = :cab"
                    p_ext["cab"] = sel_cab_ext
                
                q_ext += " ORDER BY cabang, periode"

                with page_loading("Mengambil data eksternal dari database..."):
                    with engine.begin() as conn:
                        df_ext = pd.read_sql(text(q_ext), conn, params=p_ext)

                if df_ext.empty:
                    st.info("Data tidak ditemukan untuk filter ini.")
                else:
                    df_ext["periode"] = pd.to_datetime(df_ext["periode"]).dt.date
                    df_ext["Hapus"] = False
                    
                    st.divider()
                    st.markdown(f"Tabel Data Eksternal ({len(df_ext)} baris)")
                    st.caption("Centang kolom 'Hapus' jika baris tersebut mau dihapus dari database.")

                    ed_ext = st.data_editor(
                        df_ext,
                        num_rows="fixed",
                        use_container_width=True,
                        column_config={
                            "cabang": st.column_config.TextColumn(
                                "Cabang",
                                disabled=True,
                                help="Kode cabang.",
                            ),
                            "area": st.column_config.TextColumn(
                                "Area",
                                disabled=True,
                                help="Wilayah cabang.",
                            ),
                            "periode": st.column_config.DateColumn(
                                "Periode",
                                disabled=True,
                                format="MMM YYYY",
                                help="Bulan berlakunya data eksternal.",
                            ),
                            "event_flag": st.column_config.NumberColumn(
                                "Ada Event? (0 / 1)",
                                help="Isi 1 jika ada event/gathering, 0 jika tidak ada.",
                            ),
                            "holiday_count": st.column_config.NumberColumn(
                                "Jumlah Hari Libur",
                                help="Total hari libur di periode tersebut.",
                            ),
                            "rainfall": st.column_config.NumberColumn(
                                "Curah Hujan (mm)",
                                help="Curah hujan total/rata-rata (mm).",
                            ),
                            "Hapus": st.column_config.CheckboxColumn(
                                "Hapus",
                                help="Centang kalau baris ini mau dihapus.",
                            ),
                        },
                        key="editor_ext_main",
                    )

                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                    
                    cb_del, cb_save, cb_spc = st.columns([2, 2, 6])

                    with cb_del:
                        sub_c1 = st.container()
                        with sub_c1:
                            if st.button(
                                "Hapus Data Tercentang",
                                type="secondary",
                                key="btn_init_del",
                                use_container_width=True,
                            ):
                                st.session_state["ext_confirm_delete_stage"] = 1
                        
                        if st.session_state["ext_confirm_delete_stage"] == 1:
                            to_del = ed_ext[ed_ext["Hapus"] == True]
                            n_del = len(to_del)

                            if n_del == 0:
                                st.info("Tidak ada baris yang dicentang untuk dihapus.")
                                st.session_state["ext_confirm_delete_stage"] = 0
                            else:
                                st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)
                                st.warning(f"⚠️ Hapus {n_del} baris data terpilih?")

                                def del_all():
                                    recs = [
                                        {"c": r["cabang"], "p": r["periode"]}
                                        for _, r in to_del.iterrows()
                                    ]
                                    with engine.begin() as conn:
                                        conn.execute(
                                            text(
                                                "DELETE FROM external_data "
                                                "WHERE cabang = :c AND periode = :p"
                                            ),
                                            recs,
                                        )
                                    st.session_state["ext_confirm_delete_stage"] = 0
                                    st.success("Data terpilih telah dihapus.")
                                    try:
                                        st.toast("Data eksternal terpilih dihapus.", icon="🗑️")
                                    except Exception:
                                        pass

                                action_with_loader(
                                    key="act_del_all_confirm",
                                    button_label="Ya, Hapus",
                                    message="Menghapus...",
                                    fn=del_all,
                                )

                    with cb_save:
                        def save_ext():
                            to_del = ed_ext[ed_ext["Hapus"] == True]
                            to_save = ed_ext[ed_ext["Hapus"] == False]
                            
                            with engine.begin() as conn:
                                upserts = []
                                for _, r in to_save.iterrows():
                                    upserts.append(
                                        {
                                            "c": r["cabang"],
                                            "a": r.get("area") or r["cabang"],
                                            "p": r["periode"],
                                            "ef": int(r["event_flag"]),
                                            "hc": int(r["holiday_count"]),
                                            "rf": float(r["rainfall"]),
                                            "sf": "manual_edit",
                                            "ub": user_id,
                                        }
                                    )
                                if upserts:
                                    conn.execute(
                                        text(
                                            """
                                            INSERT INTO external_data (
                                                cabang, area, periode,
                                                event_flag, holiday_count, rainfall,
                                                source_filename, uploaded_by
                                            )
                                            VALUES (
                                                :c, :a, :p,
                                                :ef, :hc, :rf,
                                                :sf, :ub
                                            )
                                            ON DUPLICATE KEY UPDATE
                                                event_flag    = VALUES(event_flag),
                                                holiday_count = VALUES(holiday_count),
                                                rainfall      = VALUES(rainfall),
                                                uploaded_by   = VALUES(uploaded_by)
                                            """
                                        ),
                                        upserts,
                                    )
                                
                                if not to_del.empty:
                                    dels = [
                                        {"c": r["cabang"], "p": r["periode"]}
                                        for _, r in to_del.iterrows()
                                    ]
                                    conn.execute(
                                        text(
                                            "DELETE FROM external_data "
                                            "WHERE cabang = :c AND periode = :p"
                                        ),
                                        dels,
                                    )
                            
                            st.success("Perubahan tersimpan.")
                            try:
                                st.toast("Database external_data diperbarui.", icon="✅")
                            except Exception:
                                pass

                        action_with_loader(
                            key="act_save_ext",
                            button_label="Simpan Perubahan",
                            message="Menyimpan...",
                            fn=save_ext,
                        )

                    with cb_spc:
                        st.empty()

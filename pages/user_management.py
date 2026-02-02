import streamlit as st
import pandas as pd
from app.ui.theme import inject_global_theme, render_sidebar_user_and_logout
from app.loading_utils import init_loading_css, action_with_loader
from app.services.page_loader import init_page_loader_css, page_loading
from app.services.user_management_service import (
    get_all_users,
    update_user_basic,
    delete_user_account,
)
from app.services.auth_service import (
    get_pending_reset_codes,
    cleanup_reset_codes,
    create_user,
    admin_approve_reset,
)
from app.services.auth_guard import require_login


st.set_page_config(
    page_title="Manajemen Pengguna",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_page_loader_css()

with page_loading("Menyiapkan halaman Manajemen Pengguna..."):
    require_login()
    inject_global_theme()
    render_sidebar_user_and_logout()
    init_loading_css()

st.markdown(
    """
    <style>
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        padding: 10px 14px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.05);
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 80px;
        text-align: left;
    }

    .kpi-header-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 2px;
    }

    .kpi-title {
        font-size: 11px;
        font-weight: 600;
        color: #9ca3af;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .kpi-value {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-top: 4px;
        text-align: left;
    }

    .kpi-help {
        width: 16px;
        height: 16px;
        border-radius: 9999px;
        border: 1px solid #cbd5e1;
        font-size: 10px;
        color: #6b7280;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: default;
    }

    .kpi-help:hover {
        background-color: #eff6ff;
        border-color: #93c5fd;
        color: #1d4ed8;
    }

    .delete-warning-box {
        background-color: #fff7ed;
        border-radius: 8px;
        border: 1px solid #fed7aa;
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 10px;
        color: #9a3412;
        font-size: 14px;
        line-height: 1.45;
    }

    .delete-user-meta {
        font-size: 13px;
        color: #4b5563;
        margin-top: 4px;
        margin-bottom: 10px;
    }

    .section-wrapper {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        padding: 16px 20px 18px 20px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        margin-top: 10px;
        margin-bottom: 16px;
    }

    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
    }

    .info-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-left: 6px;
        width: 18px;
        height: 18px;
        font-size: 11px;
        font-weight: 600;
        color: #475569;
        background-color: #e2e8f0;
        border-radius: 50%;
        cursor: default;
        transition: all 0.15s ease;
    }
    .info-icon:hover {
        background-color: #cbd5e1;
        color: #1e293b;
    }

    .reset-table div[data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        background-color: #ffffff;
    }

    .reset-table div[data-testid="stDataFrame"] thead tr {
        background-color: #f9fafb;
    }

    .reset-table div[data-testid="stDataFrame"] thead tr th {
        font-size: 12px;
        font-weight: 600;
        color: #4b5563;
        border-bottom: 1px solid #e5e7eb;
    }

    .reset-table div[data-testid="stDataFrame"] tbody tr {
        font-size: 13px;
        color: #111827;
    }

    .reset-table div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f9fafb;
    }

    .reset-table div[data-testid="stDataFrame"] tbody tr:hover {
        background-color: #eef2ff;
    }

    thead tr th:first-child { display: none; }
    tbody th { display: none; }

    div[data-testid="stTextInput"] input {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }

    div[data-testid="stTextInput"] input:focus {
        outline: none;
        border-color: #0f172a;
        box-shadow: 0 0 0 1px rgba(15,23,42,0.08);
    }

    div[data-testid="stSelectbox"] > div[data-baseweb="select"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        min-height: 38px;
    }

    div[data-testid="stSelectbox"] > div[data-baseweb="select"] div {
        background-color: transparent;
    }

    div[data-testid="stSelectbox"] > div[data-baseweb="select"]:focus-within {
        outline: none;
        border-color: #0f172a;
        box-shadow: 0 0 0 1px rgba(15,23,42,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cek login dan role
if "user" not in st.session_state:
    st.error("Sesi habis. Silakan login lagi.")
    st.stop()

current_user = st.session_state["user"]
if current_user.get("role") != "admin":
    st.error("Maaf, Anda tidak punya akses ke halaman ini.")
    st.stop()

# Ambil data user dan reset password
with page_loading("Mengambil data akun & status reset password..."):
    users = get_all_users() or []
    total_users = len(users)
    total_admins = len([u for u in users if u["role"] == "admin"])
    total_active = len([u for u in users if u.get("is_active", 1)])

    cleanup_reset_codes()
    reset_active_codes = get_pending_reset_codes() or []
    total_active_reset = len(reset_active_codes)

# Header + KPI
st.title("Manajemen Pengguna")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-header-row">
            <div class="kpi-title">TOTAL AKUN</div>
            <div class="kpi-help" title="Semua user yang terdaftar di sistem.">?</div>
          </div>
          <div class="kpi-value">{total_users}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-header-row">
            <div class="kpi-title">ADMIN</div>
            <div class="kpi-help" title="User dengan hak akses admin.">?</div>
          </div>
          <div class="kpi-value">{total_admins}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-header-row">
            <div class="kpi-title">AKUN AKTIF</div>
            <div class="kpi-help" title="Akun yang statusnya aktif dan boleh login.">?</div>
          </div>
          <div class="kpi-value">{total_active}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-header-row">
            <div class="kpi-title">KODE RESET AKTIF</div>
            <div class="kpi-help" title="Jumlah kode reset password yang masih berlaku.">?</div>
          </div>
          <div class="kpi-value">{total_active_reset}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

tab_list, tab_create, tab_reset = st.tabs(
    ["Daftar Akun", "Buat Akun Baru", "Manajemen Reset Password"]
)

with tab_list:
    f_col, s_col = st.columns([2, 2])
    with f_col:
        filter_option = st.radio(
            "Filter data akun",
            ["Semua", "Akun aktif", "Akun nonaktif"],
            horizontal=True,
            key="filter_role",
        )
    with s_col:
        search_text = st.text_input(
            "Cari user",
            key="filter_user_search",
            placeholder="Nama atau email",
        )

    if filter_option == "Akun aktif":
        base_filtered = [u for u in users if u.get("is_active", 1)]
    elif filter_option == "Akun nonaktif":
        base_filtered = [u for u in users if not u.get("is_active", 1)]
    else:
        base_filtered = users

    if search_text:
        q = search_text.strip().lower()
        filtered_users = [
            u
            for u in base_filtered
            if q in u["full_name"].lower() or q in u["email"].lower()
        ]
    else:
        filtered_users = base_filtered

    st.markdown("<br>", unsafe_allow_html=True)

    col_table, col_edit = st.columns([1.6, 1], gap="large")

    with col_table:
        if filter_option == "Akun aktif":
            judul_tabel = "Data Akun Aktif"
        elif filter_option == "Akun nonaktif":
            judul_tabel = "Data Akun Nonaktif"
        else:
            judul_tabel = "Semua Data Akun"

        st.subheader(judul_tabel)

        if not filtered_users:
            st.info("Tidak ada data untuk filter ini.")
        else:
            df_display = pd.DataFrame(filtered_users).rename(
                columns={
                    "full_name": "Nama Lengkap",
                    "email": "Email",
                    "role": "Jabatan",
                    "is_active": "Status",
                }
            )

            st.dataframe(
                df_display,
                column_order=("Nama Lengkap", "Email", "Jabatan", "Status"),
                use_container_width=True,
                hide_index=True,
            )

    with col_edit:
        st.subheader("Ubah Data")

        if not filtered_users:
            st.info("Tidak ada akun yang bisa diedit untuk filter ini.")
        else:
            user_options = [u["full_name"] for u in filtered_users]
            selected_idx = st.selectbox(
                "Pilih user",
                range(len(filtered_users)),
                format_func=lambda i: user_options[i],
            )

            selected_user = filtered_users[selected_idx]
            selected_id = selected_user["user_id"]

            edit_name_key = f"edit_name_{selected_id}"
            edit_role_key = f"edit_role_{selected_id}"
            active_key = f"edit_active_{selected_id}"
            confirm_key = f"confirm_delete_{selected_id}"

            if edit_name_key not in st.session_state:
                st.session_state[edit_name_key] = selected_user["full_name"]
            if edit_role_key not in st.session_state:
                st.session_state[edit_role_key] = selected_user["role"]
            if active_key not in st.session_state:
                st.session_state[active_key] = bool(selected_user.get("is_active", 1))
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = False

            with st.container(border=True):
                st.text_input("Nama", key=edit_name_key)
                st.text_input("Email", value=selected_user["email"], disabled=True)

                c_role, c_active = st.columns([1.5, 1])
                with c_role:
                    st.selectbox("Jabatan", ["admin", "user"], key=edit_role_key)
                with c_active:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.toggle("Akun aktif", key=active_key)

                st.markdown("<br>", unsafe_allow_html=True)

                save_col, _ = st.columns([1, 1])

                def simpan_data():
                    try:
                        name = st.session_state.get(edit_name_key, "").strip()
                        role_val = st.session_state.get(edit_role_key, "user")
                        is_active_val = int(st.session_state.get(active_key, True))

                        if not name:
                            st.error("Nama tidak boleh kosong.")
                            return

                        update_user_basic(
                            user_id=selected_id,
                            full_name=name,
                            role=role_val,
                            is_active=is_active_val,
                        )
                        st.toast("Data berhasil disimpan.")
                    except Exception as e:
                        st.error(f"Gagal menyimpan: {e}")

                with save_col:
                    action_with_loader(
                        key=f"save_btn_{selected_id}",
                        button_label="Simpan Perubahan",
                        message="Sedang menyimpan...",
                        fn=simpan_data,
                        button_type="primary",
                    )

                st.markdown(
                    "<hr style='border:0;border-top:1px solid #e5e7eb; margin:12px 0 10px 0;'>",
                    unsafe_allow_html=True,
                )

                if filter_option == "Akun nonaktif" and not selected_user.get(
                    "is_active", 1
                ):
                    if not st.session_state[confirm_key]:
                        if st.button(
                            "Hapus permanen",
                            key=f"show_confirm_{selected_id}",
                            type="secondary",
                        ):
                            st.session_state[confirm_key] = True
                            st.rerun()
                    else:
                        st.markdown(
                            """
                            <div class="delete-warning-box">
                                <strong>Hapus akun secara permanen?</strong><br>
                                Tindakan ini tidak dapat dibatalkan. Akun dan seluruh hak akses akan dihapus dari sistem.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"""
                            <div class="delete-user-meta">
                                <strong>Nama:</strong> {selected_user["full_name"]}<br>
                                <strong>Email:</strong> {selected_user["email"]} 
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        c_cancel, c_ok = st.columns(2)

                        with c_cancel:
                            if st.button(
                                "Batalkan",
                                key=f"cancel_delete_{selected_id}",
                                type="primary",
                                use_container_width=True,
                            ):
                                st.session_state[confirm_key] = False
                                st.rerun()

                        with c_ok:

                            def hapus_permanen():
                                try:
                                    delete_user_account(selected_id)
                                    st.toast("Akun berhasil dihapus permanen.")
                                except Exception as e:
                                    st.error(f"Gagal menghapus akun: {e}")
                                finally:
                                    st.session_state[confirm_key] = False
                                    st.rerun()

                            action_with_loader(
                                key=f"delete_btn_{selected_id}",
                                button_label="Ya, hapus",
                                message="Menghapus akun...",
                                fn=hapus_permanen,
                                button_type="secondary",
                            )
                else:
                    st.caption(
                        "Hapus permanen hanya tersedia untuk akun yang sudah dinonaktifkan."
                    )

with tab_create:
    st.subheader("Buat Akun Baru")

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            full_name = st.text_input("Nama Lengkap")
        with col_b:
            email = st.text_input("Email")

        col_c, col_d = st.columns(2)
        with col_c:
            role = st.selectbox(
                "Jabatan",
                ["user", "admin"],
            )
        with col_d:
            password_input = st.text_input(
                "Password (boleh dikosongkan)",
                type="password",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Buat Akun", key="btn_create_user", type="primary"):
            nm = full_name.strip()
            em = email.strip()
            rl = role
            raw_pw = password_input.strip()
            admin_isi_password = bool(raw_pw)
            pw_arg = raw_pw or None

            if not nm or not em:
                st.error("Nama dan Email wajib diisi.")
            else:
                try:
                    res = create_user(
                        full_name=nm,
                        email=em,
                        role=rl,
                        temp_password=pw_arg,
                    )

                    email_info = res["email"]
                    st.success(f"Akun untuk {email_info} berhasil dibuat.")

                    if admin_isi_password:
                        st.info("Password menggunakan input yang dimasukkan admin.")
                    else:
                        temp_pw = res.get("temp_password", "")
                        st.info(
                            f"Password sementara: {temp_pw}\n\n"
                            "Berikan password ini ke user. User akan diminta mengganti password saat login pertama."
                        )
                except Exception as e:
                    st.error(str(e))

with tab_reset:
    st.subheader("Manajemen Reset Password")

    cleanup_reset_codes()
    active_code = get_pending_reset_codes() or []

    st.markdown(
        """
        <div class="section-wrapper">
          <div class="section-title">
            Generate Kode Reset
            <span class="info-icon"
                  title="Pilih user aktif lalu klik tombol untuk membuat kode reset. Kode akan kadaluarsa otomatis setelah waktu tertentu.">?</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_users_only = [u for u in users if u.get("is_active", 1)]

    if not active_users_only:
        st.info("Tidak ada user aktif yang dapat dibuatkan kode reset.")
    else:
        email_list = [u["email"] for u in active_users_only]
        selected_email = st.selectbox(
            "Pilih user untuk dibuatkan kode reset:", email_list
        )

        if st.button("Generate Kode Reset", type="primary"):
            target_user = next((u for u in users if u["email"] == selected_email), None)

            if not target_user:
                st.error("User tidak ditemukan.")
            elif not target_user.get("is_active", 1):
                st.error("Akun sudah nonaktif, tidak bisa dibuatkan kode reset.")
            else:
                try:
                    code = admin_approve_reset(selected_email, ttl_minutes=30)
                    st.success(f"Kode reset untuk {selected_email} berhasil dibuat: {code}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal membuat kode reset: {e}")

    st.markdown("---")

    st.markdown(
        """
        <div class="section-wrapper">
          <div class="section-title">
            Kode Reset Aktif
            <span class="info-icon"
                  title="Kode reset yang dibuat admin dan masih berlaku. Akan hilang otomatis setelah dipakai atau kedaluwarsa.">?</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not active_code:
        st.info("Tidak ada kode reset aktif.")
    else:
        df_active = pd.DataFrame(active_code)
        df_active["Kadaluarsa"] = df_active["reset_expiry"]

        df_show = df_active[["email", "reset_token", "Kadaluarsa"]]
        df_show.rename(
            columns={
                "email": "Email",
                "reset_token": "Kode Reset",
            },
            inplace=True,
        )

        with st.container():
            st.markdown('<div class="reset-table">', unsafe_allow_html=True)
            st.dataframe(df_show, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

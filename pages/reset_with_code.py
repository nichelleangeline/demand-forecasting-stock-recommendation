import streamlit as st
import time

from app.loading_utils import init_loading_css
from app.services.auth_service import (
    verify_reset_code,
    mark_reset_code_used_and_flag_change,
)

# =========================================================
# 1. KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Verifikasi Kode",
    layout="centered",
    initial_sidebar_state="collapsed",
)

init_loading_css()

# =========================================================
# 2. CSS: LAYOUT + CARD + KOMPONEN (LEBAR TOMBOL = INPUT) + HIDE NAVBAR
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* HIDE DEFAULT STREAMLIT ELEMENTS (SIDEBAR + HEADER + FOOTER) */
    [data-testid="stSidebar"] {display: none;}
    header, footer {visibility: hidden;}

    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 460px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    /* Card utama: blok yang berisi text input */
    div[data-testid="stVerticalBlock"]:has(> div > div[data-testid="stTextInput"]) {
        background-color: #ffffff;
        padding: 32px 28px 28px 28px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.05);
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Typography judul */
    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 14px;
        color: #64748b;
        text-align: center;
        margin-bottom: 24px;
        line-height: 1.5;
        width: 100%;
    }

    /* Input fields — lebar maksimal 320px */
    .stTextInput {
        width: 100% !important;
        max-width: 320px;
    }

    .stTextInput label {
        color: #111827;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 4px;
        display: block;
    }

    .stTextInput input {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
        color: #0f172a;
        width: 100% !important;
        box-sizing: border-box;
    }
    .stTextInput input:focus {
        border-color: #0f172a;
        box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.1);
        outline: none;
    }

    /* Spasi antar field */
    div[data-testid="stVerticalBlock"]:has(> div > div[data-testid="stTextInput"])
        div[data-testid="stTextInput"]:nth-child(2) {
        margin-top: 16px;
    }

    /* ===== TOMBOL — LEBAR SAMA DENGAN INPUT & DI TENGAH ===== */
    /* Target form container */
    div[data-testid="stForm"] {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-top: 18px;
    }

    /* Container tombol di dalam form */
    div[data-testid="stForm"] > div {
        width: 100%;
        max-width: 320px;
    }

    /* Tombol submit — lebar penuh container */
    div[data-testid="stForm"] button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 11px 24px;
        font-weight: 600;
        border: none;
        font-size: 14px;
        transition: background-color 0.2s, box-shadow 0.2s;
        width: 100%;
        max-width: 100%;
        white-space: nowrap;
    }

    div[data-testid="stForm"] button:hover {
        background-color: #1f2937 !important;
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.18) !important;
    }

    /* Error message */
    .stAlert {
        margin-top: 12px;
        margin-bottom: 0;
        width: 100%;
        max-width: 320px;
    }

    /* Footer link */
    .footer-link { 
        text-align: center; 
        margin-top: 8px; 
    }
    .footer-link a { 
        color: #64748b; 
        text-decoration: none; 
        font-size: 13px; 
        font-weight: 500; 
    }
    .footer-link a:hover { 
        color: #0f172a; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. STATE MANAGEMENT
# =========================================================
if "verify_error" not in st.session_state:
    st.session_state["verify_error"] = None

# =========================================================
# 4. KONTEN HALAMAN (SEMUA DI DALAM FORM / CARD)
# =========================================================
with st.form("verify_code_form", clear_on_submit=False, border=False):
    st.markdown('<div class="page-title">Verifikasi Kode</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Masukkan email dan kode 6 digit yang Anda terima dari Admin untuk melanjutkan penggantian password.</div>',
        unsafe_allow_html=True,
    )

    email = st.text_input("Email", placeholder="nama@perusahaan.com")
    code = st.text_input("Kode Reset", placeholder="Contoh: 123456")

    if st.session_state["verify_error"]:
        st.error(st.session_state["verify_error"])

    submitted = st.form_submit_button("Verifikasi Kode")

    if submitted:
        e = email.strip()
        c = code.strip()

        st.session_state["verify_error"] = None

        if not e or not c:
            st.session_state["verify_error"] = "Email dan Kode wajib diisi."
        elif not verify_reset_code(e, c):
            st.session_state["verify_error"] = "Kode salah, tidak valid, atau sudah kadaluarsa."
        else:
            with st.spinner("Memproses..."):
                time.sleep(0.6)
                mark_reset_code_used_and_flag_change(e)
                st.session_state["force_change_email"] = e
                st.session_state["reset_reason"] = "forgot_code"

            st.success("Kode valid. Mengalihkan...")
            time.sleep(0.6)
            st.switch_page("pages/reset_first_login.py")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer-link">
        <a href="/" target="_self">← Kembali ke Halaman Login</a>
    </div>
    """,
    unsafe_allow_html=True,
)

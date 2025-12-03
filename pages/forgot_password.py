# pages/forgot_password.py

import streamlit as st
import time

from app.loading_utils import init_loading_css
from app.services.auth_service import get_user_by_email, set_reset_code

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Lupa Password",
    layout="centered",
    initial_sidebar_state="collapsed",
)

init_loading_css()

# =========================================================
# CSS (DISERAGAMKAN DENGAN HALAMAN VERIFIKASI KODE + HIDE NAVBAR)
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    [data-testid="stSidebar"] {display: none;}
    header, footer {visibility: hidden;}

    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 440px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    .duck-wrapper {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-bottom: 18px;
    }
    .duck-container-small {
        width: 50px;
        height: 50px;
        position: relative;
        animation: float 3s ease-in-out infinite;
    }
    .duck-container-small:nth-child(2) { animation-delay: 0.2s; }
    .duck-container-small:nth-child(3) { animation-delay: 0.4s; }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-6px); }
    }

    .duck-head {
        width: 50px;
        height: 50px;
        background:#4caf50;
        border-radius:50%;
        position:relative;
    }
    .duck-question {
        position:absolute;
        top:-5px;
        right:0px;
        font-size:18px;
        font-weight:bold;
        color:#2563eb;
    }
    .eye {
        position:absolute;
        top:20px;
        width:6px;
        height:6px;
        background:#333;
        border-radius:50%;
    }
    .eye.l { left:13px; }
    .eye.r { right:13px; }
    .eye::after {
        content:'';
        position:absolute;
        top:1px;
        right:1px;
        width:2px;
        height:2px;
        background:#fff;
        border-radius:50%;
    }
    .beak {
        position:absolute;
        bottom:12px;
        left:50%;
        transform:translateX(-50%);
        width:22px;
        height:9px;
        background:#ffc107;
        border-radius:10px;
        border-bottom:2px solid #ff9800;
    }

    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stForm"]) {
        background-color: #ffffff;
        padding: 32px 30px 26px 30px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.12);
        margin-top: 16px;
    }

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
        margin-bottom: 22px;
        line-height: 1.5;
    }

    .stTextInput {
        width: 100% !important;
        max-width: 320px;
        margin: 0 auto;
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

    /* Form dibikin tengah */
    div[data-testid="stForm"] {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-top: 18px;
    }

    /* Wrapper dalam form biar fix 320px */
    .form-inner-wrapper {
        width: 100%;
        max-width: 320px;
        margin: 0 auto;
    }

    /* Tombol submit full width sama seperti input */
    div[data-testid="stForm"] button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 11px 16px;
        font-weight: 600;
        border: none;
        font-size: 14px;
        transition: background-color 0.2s;
        width: 100%;
        white-space: nowrap;
    }

    div[data-testid="stForm"] button:hover {
        background-color: #334155 !important;
        color: #ffffff !important;
    }

    .separator {
        display: flex;
        align-items: center;
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin: 22px auto 8px auto;
    }
    .separator::before,
    .separator::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid #e2e8f0;
    }
    .separator::before { margin-right: .5em; }
    .separator::after  { margin-left: .5em; }

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
# CARD UTAMA
# =========================================================
with st.container():
    st.markdown('<div class="page-title">Lupa Password?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Masukkan email Anda untuk mencatat permintaan reset. Admin akan memberikan kode 6 digit.</div>',
        unsafe_allow_html=True,
    )

    with st.form("reset_password_form", clear_on_submit=False, border=False):
        # wrapper biar input & tombol punya lebar sama
        st.markdown('<div class="form-inner-wrapper">', unsafe_allow_html=True)

        email = st.text_input("Email", placeholder="nama@perusahaan.com")
        submitted = st.form_submit_button(
            "Kirim Permintaan Reset",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            email_clean = email.strip()
            if not email_clean:
                st.warning("Harap masukkan alamat email.")
            else:
                with st.spinner("Mengecek email..."):
                    time.sleep(0.6)
                    user = get_user_by_email(email_clean)

                if not user:
                    st.error("Email tersebut tidak terdaftar.")
                else:
                    set_reset_code(email_clean)
                    st.success("Permintaan reset password sudah dicatat.")
                    st.info("Hubungi Admin untuk mendapatkan kode verifikasi 6 digit.")

    st.markdown('<div class="separator">atau</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footer-link">
            <a href="/reset_with_code" target="_self">Punya Kode Reset</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer-link">
        <a href="/" target="_self">← Kembali ke Halaman Login</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# pages/login_page.py

import streamlit as st
from app.services.auth_service import get_user_by_email, hash_password

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Masuk | Sistem Perencanaan Stok",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CSS UTAMA LOGIN PAGE
# =========================================================
MODERN_CSS = """
<style>
/* HIDE DEFAULT STREAMLIT ELEMENTS DI HALAMAN LOGIN */
[data-testid="stSidebar"] {display: none;}
header, footer {visibility: hidden;}

.stApp {
    background-color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

.form-wrapper {
    max-width: 420px;
    margin: 80px auto 0;
    position: relative;
}

/* MASKOT BEBEK */
.maskot-container {
    position: absolute;
    top: -70px;
    left: 50%;
    transform: translateX(-50%);
    width: 110px;
    height: 90px;
    z-index: 10;
    display: flex;
    justify-content: center;
    align-items: flex-end;
    pointer-events: none;
    animation: duckFloat 3s ease-in-out infinite;
}

@keyframes duckFloat {
    0%   { transform: translateX(-50%) translateY(0px); }
    50%  { transform: translateX(-50%) translateY(-6px); }
    100% { transform: translateX(-50%) translateY(0px); }
}

.duck-head {
    width: 90px;
    height: 90px;
    background-color: #4caf50;
    border-radius: 50%;
    position: relative;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

.duck-tuft {
    position: absolute;
    top: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 22px;
    height: 20px;
    background-color: #4caf50;
    border-radius: 50% 50% 0 0;
}

.duck-eye {
    position: absolute;
    top: 34px;
    width: 10px;
    height: 10px;
    background-color: #1a202c;
    border-radius: 50%;
}
.duck-eye.left { left: 25px; }
.duck-eye.right { right: 25px; }
.duck-eye::after {
    content: '';
    position: absolute;
    top: 2px;
    right: 2px;
    width: 3px;
    height: 3px;
    background: white;
    border-radius: 50%;
}

.duck-beak {
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 36px;
    height: 16px;
    background-color: #ffc107;
    border-radius: 10px;
    border-bottom: 3px solid #ff9800;
}

/* FORM CARD */
div[data-testid="stForm"] {
    background-color: #ffffff;
    padding: 50px 30px 40px 30px;
    border-radius: 18px;
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.10);
    border: 1px solid #e2e8f0;
}

.sign-in-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}

.sign-in-subtitle {
    text-align: center;
    color: #718096;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

div[data-testid="stTextInput"] {
    margin-bottom: 12px;
}
div[data-testid="stTextInput"] input {
    padding: 10px 14px;
    border-radius: 10px;
    border: 1px solid #cbd5e0;
    background-color: #f8fafc;
    height: 46px;
}
div[data-testid="stTextInput"] input:focus {
    background-color: #ffffff;
    border-color: #0f172a;
}

div[data-testid="stFormSubmitButton"] {
    margin-top: 15px;
}
div[data-testid="stFormSubmitButton"] > button {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: none !important;
    width: 100%;
    max-width: 280px;
    padding: 12px 20px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    transition: 0.2s;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #333333 !important;
    transform: translateY(-2px);
}

.footer-link { text-align: center; margin-top: 15px; }
.footer-link a {
    color: #64748b;
    font-size: 13px;
    text-decoration: none;
}
.footer-link a:hover { color: #0f172a; }
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

# =========================================================
# AUTO-REDIRECT: SUDAH LOGIN → DASHBOARD
# =========================================================
if "user" in st.session_state:
    st.switch_page("pages/dashboard.py")
    st.stop()

# =========================================================
# FORM LOGIN
# =========================================================
st.markdown('<div class="form-wrapper">', unsafe_allow_html=True)

# Maskot
st.markdown(
    """
    <div class="maskot-container">
        <div class="duck-head">
            <div class="duck-tuft"></div>
            <div class="duck-eye left"></div>
            <div class="duck-eye right"></div>
            <div class="duck-beak"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("login_form"):
    st.markdown(
        '<div class="sign-in-title">Masuk ke Sistem</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='sign-in-subtitle'>Gunakan akun yang terdaftar untuk mengakses.</p>",
        unsafe_allow_html=True,
    )

    email = st.text_input("Email", placeholder="Alamat email")
    password = st.text_input("Password", placeholder="Kata sandi", type="password")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submitted = st.form_submit_button("Masuk", use_container_width=True)

    if submitted:
        e = email.strip()
        pw = password.strip()

        if not e or not pw:
            st.warning("Email dan kata sandi wajib diisi.")
        else:
            user = get_user_by_email(e)
            if not user:
                st.error("Email tidak terdaftar.")
            elif not user.get("is_active", 1):
                st.error("Akun tidak aktif.")
            elif hash_password(pw) != user.get("password_hash"):
                st.error("Kata sandi tidak sesuai.")
            else:
                # FIRST LOGIN? → PAKSA KE RESET PASSWORD
                if user.get("must_change_password", 0) == 1:
                    st.session_state["force_change_email"] = user["email"]
                    st.session_state["reset_reason"] = "first_login"
                    st.switch_page("pages/reset_first_login.py")
                else:
                    # LOGIN NORMAL
                    st.session_state["user"] = user
                    st.switch_page("pages/dashboard.py")

st.markdown(
    '<div class="footer-link"><a href="/forgot_password" target="_self">Lupa kata sandi?</a></div>',
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

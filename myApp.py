import streamlit as st
from app.loading_utils import init_loading_css
from app.services.auth_service import (
    get_user_by_email,
    hash_password,
    set_reset_code,
    verify_reset_code,
    mark_reset_code_used_and_flag_change,
)
from app.services.page_loader import (
    init_page_loader_css,
    page_loading,   
)

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Sistem Perencanaan Stok",
    layout="centered",
    initial_sidebar_state="collapsed",
)

init_loading_css()
init_page_loader_css()

# 2. CSS GLOBAL (LOGIN + FORGOT + VERIFY)
MODERN_CSS = """
<style>
/* Sembunyikan sidebar dan header di halaman login */
[data-testid="stSidebar"] {display: none;}
header, footer {visibility: hidden;}

/* Sembunyikan teks 'Press Enter to submit form' */
[data-testid="InputInstructions"] {
    display: none !important;
}

.stApp {
    background-color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

/* Container kartu form */
.form-wrapper {
    max-width: 420px;
    margin: 80px auto 0;
    position: relative;
}

/* Maskot bebek di atas kartu */
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

/* Warna utama bebek */
:root {
    --duck-green: #22c55e;
}

/* Bebek login / verify */
.duck-head-login {
    width: 90px;
    height: 90px;
    background-color: var(--duck-green);
    border-radius: 50%;
    position: relative;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
.duck-tuft-login {
    position: absolute;
    top: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 26px;
    height: 22px;
    background-color: var(--duck-green);
    border-radius: 50% 50% 0 0;
}

/* Bebek forgot (versi bingung) */
.duck-head-forgot {
    width: 90px;
    height: 90px;
    background-color: var(--duck-green);
    border-radius: 50%;
    position: relative;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
.duck-tuft-forgot {
    position: absolute;
    top: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 26px;
    height: 22px;
    background-color: var(--duck-green);
    border-radius: 50% 50% 0 0;
}
.duck-question {
    position: absolute;
    top: 2px;
    right: 2px;
    font-size: 30px;
    font-weight: 700;
    color: #0f172a;
}

/* Mata dan paruh (dipakai semua varian) */
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

/* Badge kunci untuk verifikasi kode */
.duck-verify {
    position: relative;
}
.duck-verify-badge {
    position: absolute;
    bottom: -8px;
    right: -8px;
    background: transparent;
    font-size: 30px;
}

/* Kartu form */
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

/* Input */
div[data-testid="stTextInput"] {
    margin-bottom: 12px;
}
div[data-testid="stTextInput"] label {
    font-size: 14px;
    font-weight: 500;
    color: #334155;
    margin-bottom: 4px;
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

/* Tombol submit hitam */
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

/* Semua tombol secondary tampil seperti link */
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    height: auto !important;
    border-radius: 0 !important;

    color: #111827 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    cursor: pointer !important;

    display: inline-block !important;
    white-space: nowrap !important;
}

/* Footer link di bawah form */
.footer-link {
    margin-top: 18px;
    text-align: center;
}

/* Separator "atau" */
.separator {
    display: flex;
    align-items: center;
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin: 25px 0 10px 0;
}
.separator::before,
.separator::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #e2e8f0;
}
.separator::before { margin-right: .75em; }
.separator::after  { margin-left: .75em; }
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

# 3. INIT STATE
if "auth_view" not in st.session_state:
    st.session_state["auth_view"] = "login"
if "verify_error" not in st.session_state:
    st.session_state["verify_error"] = None


def view_login():
    st.markdown('<div class="form-wrapper">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="maskot-container">
            <div class="duck-head-login">
                <div class="duck-tuft-login"></div>
                <div class="duck-eye left"></div>
                <div class="duck-eye right"></div>
                <div class="duck-beak"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        st.markdown('<div class="sign-in-title">Masuk ke Sistem</div>', unsafe_allow_html=True)
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
                with page_loading("Memverifikasi akun dan memuat dashboard..."):
                    user = get_user_by_email(e)
                    if not user:
                        st.error("Email tidak terdaftar.")
                        return
                    if not user.get("is_active", 1):
                        st.error("Akun tidak aktif.")
                        return
                    if hash_password(pw) != user.get("password_hash"):
                        st.error("Kata sandi tidak sesuai.")
                        return

                    if user.get("must_change_password", 0) == 1:
                        st.session_state["force_change_email"] = user["email"]
                        st.session_state["reset_reason"] = "first_login"
                        st.switch_page("pages/reset_first_login.py")
                    else:
                        st.session_state["user"] = user
                        st.session_state["_from_login"] = True
                        st.switch_page("pages/dashboard.py")

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3, 1.5, 3])
    with c2:
        if st.button("Lupa kata sandi?", key="btn_forgot"):
            st.session_state["auth_view"] = "forgot"
            st.rerun()


def view_forgot():
    st.markdown('<div class="form-wrapper">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="maskot-container">
            <div class="duck-head-forgot">
                <div class="duck-tuft-forgot"></div>
                <div class="duck-eye left"></div>
                <div class="duck-eye right"></div>
                <div class="duck-beak"></div>
                <div class="duck-question">?</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("forgot_form"):
        st.markdown('<div class="sign-in-title">Lupa Password?</div>', unsafe_allow_html=True)
        st.markdown(
            "<p class='sign-in-subtitle'>Masukkan email Anda untuk mencatat permintaan reset. Admin akan memberikan kode 6 digit.</p>",
            unsafe_allow_html=True,
        )

        email = st.text_input("Email", placeholder="nama@perusahaan.com")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            submitted = st.form_submit_button("Kirim Permintaan Reset", use_container_width=True)

        if submitted:
            email_clean = email.strip()
            if not email_clean:
                st.warning("Masukkan email.")
            else:
                with page_loading("Mencatat permintaan reset password..."):
                    user = get_user_by_email(email_clean)

                    if not user:
                        st.error("Email tidak terdaftar.")
                    elif not user.get("is_active", 1):
                        st.error("Akun tidak aktif. Hubungi Admin jika perlu mengaktifkan kembali.")
                    else:
                        set_reset_code(email_clean)
                        st.success("Permintaan reset dicatat.")
                        st.info("Hubungi Admin untuk mendapatkan kode 6 digit.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="separator">atau</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3, 1.5, 3])
    with c2:
        if st.button("Punya Kode Reset", key="btn_have_code"):
            st.session_state["auth_view"] = "verify"
            st.rerun()

    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c2:
        if st.button("← Kembali ke Halaman Login", key="btn_back_login_from_forgot"):
            st.session_state["auth_view"] = "login"
            st.rerun()


def view_verify():
    st.markdown('<div class="form-wrapper">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="maskot-container">
            <div class="duck-head-login duck-verify">
                <div class="duck-tuft-login"></div>
                <div class="duck-eye left"></div>
                <div class="duck-eye right"></div>
                <div class="duck-beak"></div>
                <div class="duck-verify-badge">🔑</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("verify_form"):
        st.markdown('<div class="sign-in-title">Verifikasi Kode</div>', unsafe_allow_html=True)
        st.markdown(
            "<p class='sign-in-subtitle'>Masukkan email dan kode 6 digit yang Anda terima dari Admin untuk melanjutkan penggantian password.</p>",
            unsafe_allow_html=True,
        )

        email = st.text_input("Email", placeholder="nama@perusahaan.com")
        code = st.text_input("Kode Reset", placeholder="Contoh: 123456")

        if st.session_state["verify_error"]:
            st.error(st.session_state["verify_error"])

        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c2:
            submitted = st.form_submit_button("Verifikasi Kode", use_container_width=True)

        if submitted:
            e = email.strip()
            c = code.strip()
            st.session_state["verify_error"] = None

            if not e or not c:
                st.session_state["verify_error"] = "Lengkapi data."
            else:
                with page_loading("Memverifikasi kode reset dan menyiapkan halaman ganti password..."):
                    if not verify_reset_code(e, c):
                        st.session_state["verify_error"] = "Kode salah, tidak valid, atau sudah kadaluarsa."
                    else:
                        user = get_user_by_email(e)
                        if not user or not user.get("is_active", 1):
                            st.session_state["verify_error"] = "Akun tidak aktif. Hubungi Admin."
                        else:
                            mark_reset_code_used_and_flag_change(e)
                            st.session_state["force_change_email"] = e
                            st.session_state["reset_reason"] = "forgot_code"
                            st.switch_page("pages/reset_first_login.py")

    st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c2:
        if st.button("← Kembali ke Lupa Password", key="btn_back_forgot"):
            st.session_state["auth_view"] = "forgot"
            st.rerun()

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c2:
        if st.button("← Kembali ke Halaman Login", key="btn_back_login_from_verify"):
            st.session_state["auth_view"] = "login"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# 4. ROUTER
view = st.session_state.get("auth_view", "login")

if view == "login":
    view_login()
elif view == "forgot":
    view_forgot()
else:
    view_verify()

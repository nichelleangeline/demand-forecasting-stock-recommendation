import streamlit as st
from app.loading_utils import init_loading_css
from app.services.auth_service import (
    get_user_by_email,
    update_password_and_clear_flag,
)

st.set_page_config(
    page_title="Ganti Password",
    layout="centered",
    initial_sidebar_state="collapsed",
)

init_loading_css()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Reset Streamlit Default */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stHeader"] {visibility: hidden;}
    header, footer {visibility: hidden;}

    /* Hilangkan 'Press Enter to apply' */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    
    .stApp {
        background-color: #f8fafc; /* Slate-50 */
        font-family: 'Inter', sans-serif;
    }

    /* Layout Container */
    .block-container {
        padding-top: 4rem;
        max-width: 480px;
    }

    .reset-card {
        background-color: #ffffff;
        padding: 40px 32px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                    0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
    }

    .icon-wrapper {
        display: flex;
        justify-content: center;
        margin-bottom: 24px;
    }
    .lock-circle {
        background-color: #eff6ff;
        color: #2563eb;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .lock-circle svg {
        width: 24px;
        height: 24px;
        stroke-width: 2;
    }

    .page-title {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-bottom: 8px;
        letter-spacing: -0.025em;
    }

    .page-subtitle {
        font-size: 14px;
        color: #64748b;
        text-align: center;
        margin-bottom: 32px;
        line-height: 1.6;
    }

    .account-badge {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 24px;
        gap: 10px;
    }
    .account-text {
        font-size: 14px;
        color: #334155;
        font-weight: 500;
    }
    .account-icon-svg {
        color: #94a3b8;
        display: flex;
        align-items: center;
    }

    .stTextInput label {
        color: #1e293b;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 6px;
    }

    .stTextInput input {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 14px;
        color: #0f172a;
        transition: border-color 0.15s ease-in-out;
    }

    .stTextInput input:focus {
        border-color: #0f172a;
        outline: none;
        box-shadow: none;
    }

    .password-hint {
        font-size: 12px;
        color: #64748b;
        margin-top: -10px;
        margin-bottom: 16px;
    }

    .stButton button {
        background-color: #0f172a;
        color: white;
        font-weight: 500;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        width: 100%;
        font-size: 14px;
        margin-top: 8px;
    }
    .stButton button:hover {
        background-color: #1e293b;
        border-color: #1e293b;
    }

    div[data-testid="stFormSubmitButton"] > button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        width: 100%;
        font-size: 14px;
        margin-top: 8px;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
    }
    
    .stAlert {
        padding: 12px;
        border-radius: 8px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "force_change_email" not in st.session_state:
    st.session_state["auth_view"] = "login"
    st.session_state.pop("verify_error", None)
    st.switch_page("myApp.py")
    st.stop()

email = st.session_state["force_change_email"]
reason = st.session_state.get("reset_reason", "forced")

if reason == "forgot_code":
    title_text = "Reset Password"
    subtitle_text = "Verifikasi berhasil. Silakan buat password baru."
elif reason == "first_login":
    title_text = "Selamat Datang"
    subtitle_text = "Untuk keamanan, harap ganti password default Anda."
else:
    title_text = "Ganti Password"
    subtitle_text = "Password Anda perlu diperbarui untuk melanjutkan."

with st.container():
    st.markdown(
        f"""
        <div class="icon-wrapper">
            <div class="lock-circle">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                     fill="none" stroke="currentColor"
                     stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
            </div>
        </div>
        <div class="page-title">{title_text}</div>
        <div class="page-subtitle">{subtitle_text}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="account-badge">
            <div class="account-icon-svg">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                     viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            </div>
            <span class="account-text">{email}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("reset_password_form"):
        new_pwd = st.text_input("Password Baru", type="password", key="new")
        st.markdown(
            '<div class="password-hint">Minimal 8 karakter.</div>',
            unsafe_allow_html=True,
        )

        confirm_pwd = st.text_input("Konfirmasi Password", type="password", key="confirm")

        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Simpan Password", use_container_width=True)

        if submitted:
            if not new_pwd or not confirm_pwd:
                st.error("Harap isi kedua kolom password.")
            elif new_pwd != confirm_pwd:
                st.error("Konfirmasi password tidak cocok.")
            elif len(new_pwd) < 8:
                st.error("Password terlalu pendek (minimal 8 karakter).")
            else:
                with st.spinner("Menyimpan perubahan..."):
                    update_password_and_clear_flag(email, new_pwd)

                    st.session_state.pop("user", None)
                    st.session_state.pop("force_change_email", None)
                    st.session_state.pop("reset_reason", None)
                    st.session_state["auth_view"] = "login"
                    st.session_state.pop("verify_error", None)

                st.success("Password berhasil diubah. Silakan login kembali dengan password baru.")
                st.switch_page("myApp.py")

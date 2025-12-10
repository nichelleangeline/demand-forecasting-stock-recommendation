import streamlit as st

# Warna utama
COLOR_BG_APP       = "#F3F4F6"
COLOR_BG_SIDEBAR   = "#000000"
COLOR_TEXT_PASIF   = "#9CA3AF"
COLOR_TEXT_HOVER   = "#FFFFFF"
COLOR_ACTIVE_BG    = "#FFFFFF"
COLOR_ACTIVE_TEXT  = "#000000"

def inject_global_theme():
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

        <style>
        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif;
        }}

        .stApp {{
            background-color: {COLOR_BG_APP} !important;
        }}

        footer, .stDeployButton {{
            display: none !important;
        }}

        /* Header diset transparan tapi tetap punya tinggi agar tombol >> tidak hilang */
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
            height: 42px !important;
            padding-top: 6px !important;
        }}

        header[data-testid="stHeader"] > div:first-child {{
            padding-top: 4px !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COLOR_BG_SIDEBAR} !important;
            border-right: 1px solid #333333;
            width: 280px !important;
        }}

        section[data-testid="stSidebar"] > div {{
            height: 100vh;
            display: flex;
            flex-direction: column;
            padding-top: 0;
            background-color: {COLOR_BG_SIDEBAR} !important;
        }}

        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}

        .sidebar-user-card {{
            padding: 40px 24px 30px 24px;
            color: #FFFFFF;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 10px;
        }}

        .sidebar-user-greeting {{
            font-size: 10px;
            color: {COLOR_TEXT_PASIF};
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .sidebar-user-name {{
            font-size: 20px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 8px;
            line-height: 1.2;
            letter-spacing: 0.5px;
        }}

        .sidebar-user-role {{
            font-size: 10px;
            color: #FFFFFF;
            font-weight: 600;
            background: #262626;
            border: 1px solid #404040;
            padding: 4px 12px;
            border-radius: 6px;
            display: inline-block;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .sidebar-nav-custom {{
            flex: 1;
            overflow-y: auto;
            padding: 10px 16px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
            background: transparent;
            border: none;
            color: {COLOR_TEXT_PASIF} !important;
            font-size: 13px;
            font-weight: 500;
            padding: 12px 16px;
            border-radius: 12px !important;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            margin: 0 !important;
        }}

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] * {{
            color: inherit !important;
        }}

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
            color: {COLOR_TEXT_HOVER} !important;
            background-color: #1A1A1A;
            transform: translateX(4px);
        }}

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
            background-color: {COLOR_ACTIVE_BG} !important;
            color: {COLOR_ACTIVE_TEXT} !important;
            font-weight: 700;
            box-shadow: 0 0 15px rgba(255,255,255,0.15);
        }}

        section[data-testid="stSidebar"] ::-webkit-scrollbar {{
            width: 4px;
        }}

        section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
            background: #333333;
            border-radius: 10px;
        }}

        #logout-area {{
            margin-top: auto;
            padding: 24px;
            background: linear-gradient(to top, {COLOR_BG_SIDEBAR} 40%, transparent);
        }}

        #logout-area .stButton {{
            width: 100%;
            display: flex;
            justify-content: center;
        }}

        #logout-area .stButton button {{
            width: 100%;
            background-color: #FFFFFF !important;
            color: #111827 !important;
            font-family: 'Poppins', sans-serif;
            font-weight: 500;
            font-size: 14px;
            border-radius: 9999px !important;
            border: none;
            padding: 10px 0;
            transition: all 0.2s ease-in-out;
        }}

        #logout-area .stButton button:hover {{
            background-color: #F3F4F6 !important;
            transform: translateY(-2px);
        }}

        #logout-area .stButton button:active {{
            transform: translateY(0);
            background-color: #E5E7EB !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_base_theme():
    inject_global_theme()


def render_sidebar_user_and_logout():
    def safe_page_link(path: str, label: str):
        try:
            st.page_link(path, label=label)
        except Exception:
            pass

    user = st.session_state.get("user", {})
    full_name = user.get("full_name") or user.get("name") or "User"
    role = user.get("role", "Admin").title()

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-user-card">
                <div class="sidebar-user-greeting">Welcome Back,</div>
                <div class="sidebar-user-name">{full_name}</div>
                <div class="sidebar-user-role">{role}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-nav-custom">', unsafe_allow_html=True)
        safe_page_link("pages/dashboard.py", " Dashboard")
        safe_page_link("pages/generate_forecast.py", "Forecast")
        safe_page_link("pages/safety_stock_page.py", "Stock & Recommendation")
        safe_page_link("pages/data_management_tables.py", "Manajemen Data")

        if role.lower() == "admin":
            safe_page_link("pages/user_management.py", "Kelola User")
            safe_page_link("pages/admin_forecast_train.py", "Training Model Forecast")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div id="logout-area">', unsafe_allow_html=True)
        if st.button("Sign Out", key="sidebar_logout_btn"):
            st.session_state.clear()
            st.switch_page("myApp.py")
        st.markdown("</div>", unsafe_allow_html=True)

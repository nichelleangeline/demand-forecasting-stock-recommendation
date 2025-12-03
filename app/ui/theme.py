# app/ui/theme.py
import streamlit as st

# =========================================================
# KONFIGURASI WARNA (PREMIUM PITCH BLACK)
# =========================================================
COLOR_BG_APP       = "#F3F4F6"    # Background luar (Abu muda bersih)
COLOR_BG_SIDEBAR   = "#000000"    # Hitam Pekat (Pitch Black) - Premium Look
COLOR_TEXT_PASIF   = "#9CA3AF"    # Abu medium (text menu pasif)
COLOR_TEXT_HOVER   = "#FFFFFF"    # Putih (text saat hover)
COLOR_ACTIVE_BG    = "#FFFFFF"    # Putih (background menu aktif)
COLOR_ACTIVE_TEXT  = "#000000"    # Hitam (teks menu aktif)

def inject_global_theme():
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        
        <style>
        /* ================= GLOBAL RESET ================= */
        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif;
        }}

        /* ====== BACKGROUND UTAMA ====== */
        .stApp {{
            background-color: {COLOR_BG_APP} !important;
        }}

        /* Sembunyikan elemen bawaan yang mengganggu */
        header, footer, .stDeployButton {{
            display: none !important;
        }}

        /* ================= SIDEBAR (HITAM PEKAT) ================= */
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_BG_SIDEBAR} !important;
            border-right: 1px solid #333333; /* Border abu gelap tipis sebagai pemisah */
            width: 280px !important;
        }}

        /* Kontainer dalam sidebar */
        section[data-testid="stSidebar"] > div {{
            height: 100vh;
            display: flex;
            flex-direction: column;
            padding-top: 0;
            background-color: {COLOR_BG_SIDEBAR} !important;
        }}

        /* Matikan Navigasi Otomatis */
        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}

        /* ================= 1. PROFIL USER (CLEAN) ================= */
        .sidebar-user-card {{
            padding: 40px 24px 30px 24px;
            color: #FFFFFF;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1); /* Garis pemisah halus */
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
            background: #262626; /* Abu gelap solid agar terlihat premium di atas hitam */
            border: 1px solid #404040;
            padding: 4px 12px;
            border-radius: 6px;
            display: inline-block;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* ================= 2. NAVIGASI (PILL SHAPE) ================= */
        .sidebar-nav-custom {{
            flex: 1;            
            overflow-y: auto;
            padding: 10px 16px;
            display: flex;
            flex-direction: column;
            gap: 6px; /* Jarak antar menu */
        }}

        /* Link Style */
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
            background: transparent;
            border: none;
            color: {COLOR_TEXT_PASIF} !important; 
            font-size: 13px;
            font-weight: 500;
            padding: 12px 16px;
            border-radius: 12px !important;
            transition: all 0.2s ease;
            text-decoration: none;
            display: flex;
            align-items: center;
            opacity: 1 !important;
            margin: 0 !important;
        }}

        /* Reset warna text di dalam link */
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] * {{
            color: inherit !important;
        }}

        /* Hover State */
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
            color: {COLOR_TEXT_HOVER} !important;
            background-color: #1A1A1A; /* Highlight abu sangat gelap saat hover */
            transform: translateX(4px);
        }}

        /* Active State (Putih Bersih) */
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
            background-color: {COLOR_ACTIVE_BG} !important;
            color: {COLOR_ACTIVE_TEXT} !important;
            font-weight: 700;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.15); /* Glow putih halus */
            transform: none;
        }}

        /* Scrollbar Sidebar */
        section[data-testid="stSidebar"] ::-webkit-scrollbar {{
            width: 4px;
        }}
        section[data-testid="stSidebar"] ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
            background: #333333;
            border-radius: 10px;
        }}

        /* ================= 3. LOGOUT (CAPSULE / PILL SHAPE) ================= */
        #logout-area {{
            margin-top: auto;
            padding: 24px;
            /* Gradient transisi dari hitam ke transparan */
            background: linear-gradient(to top, {COLOR_BG_SIDEBAR} 40%, transparent); 
        }}

        /* Container Tombol */
        #logout-area .stButton {{
            width: 100%;
            display: flex;
            justify-content: center; /* Posisi Tengah */
        }}

        /* Tombol Logout - CAPSULE STYLE */
        #logout-area .stButton button {{
            width: 100%;
            background-color: #FFFFFF !important;
            color: #111827 !important; /* Teks Hitam agak soft (Dark Slate) */
            font-family: 'Poppins', sans-serif;
            font-weight: 500; /* Font weight sedikit lebih tipis agar clean */
            font-size: 14px;
            
            /* KUNCI: Radius Besar untuk bentuk Kapsul */
            border-radius: 9999px !important; 
            
            border: none;
            padding: 10px 0; /* Padding vertikal pas */
            
            /* Shadow halus */
            box-shadow: 0 4px 6px -1px rgba(255, 255, 255, 0.1), 0 2px 4px -1px rgba(255, 255, 255, 0.06);
            transition: all 0.2s ease-in-out;
        }}

        #logout-area .stButton button:hover {{
            background-color: #F3F4F6 !important; /* Putih sedikit abu saat hover */
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(255, 255, 255, 0.2), 0 4px 6px -2px rgba(255, 255, 255, 0.1);
        }}

        #logout-area .stButton button:active {{
            transform: translateY(0);
            background-color: #E5E7EB !important;
        }}
        
        #logout-area .stButton button:focus:not(:active) {{
            border-color: transparent;
            color: #111827;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def apply_base_theme():
    """Alias for backward compatibility"""
    inject_global_theme()

# =========================
# SIDEBAR CONTENT
# =========================
def render_sidebar_user_and_logout():
    def safe_page_link(path: str, label: str):
        """Helper render link polos tanpa parameter icon sama sekali"""
        try:
            st.page_link(path, label=label)
        except Exception:
            pass

    user = st.session_state.get("user", {})
    full_name = user.get("full_name") or user.get("name") or "User"
    role = user.get("role", "Admin").title()

    with st.sidebar:
        # 1. Profil User
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

        # 2. Navigasi (Teks Polos, Tanpa Emot)
        st.markdown('<div class="sidebar-nav-custom">', unsafe_allow_html=True)
        safe_page_link("pages/dashboard.py", label=" Dashboard")
        safe_page_link("pages/generate_forecast.py", label="Forecast")
        safe_page_link("pages/safety_stock_page.py", label="Stock & Recommendation")
        safe_page_link("pages/data_management_tables.py", label="Manajemen Data")
        
        if role.lower() == "admin":
            safe_page_link("pages/user_management.py", label="Kelola User")
            safe_page_link("pages/admin_forecast_train.py", label="Training Model Forecast")

        st.markdown("</div>", unsafe_allow_html=True)

        # 3. Logout (Capsule Button)
        st.markdown('<div id="logout-area">', unsafe_allow_html=True)
        if st.button("Sign Out", key="sidebar_logout_btn"):
            # bersihkan session
            st.session_state.clear()
            # langsung pindah ke halaman utama (myApp.py)
            st.switch_page("myApp.py")
        st.markdown("</div>", unsafe_allow_html=True)

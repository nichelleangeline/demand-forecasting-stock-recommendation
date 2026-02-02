from contextlib import contextmanager
import html
import streamlit as st


def init_page_loader_css():
    if st.session_state.get("_loader_injected", False):
        return

    st.markdown(
        """
        <style>
        /* Overlay nutup layar */
        .app-loading-overlay {
            position: fixed;
            inset: 0;
            z-index: 999999;
            background: rgba(15,23,42,0.25);
            backdrop-filter: blur(2px);
            display: none;
            align-items: center;
            justify-content: center;
        }
        .app-loading-overlay.show { display: flex; }

        .app-loading-card {
            background: #0f172a;
            color: #e5e7eb;
            padding: 14px 20px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }

        .app-loading-spinner {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 2px solid rgba(255,255,255,0.4);
            border-top-color: #facc15;
            animation: spin 0.7s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Tutup status bawaan streamlit */
        [data-testid="stStatusWidget"],
        section[aria-label="Status"] {
            display: none !important;
        }
        </style>

        <div id="app-loader" class="app-loading-overlay">
            <div class="app-loading-card">
                <div class="app-loading-spinner"></div>
                <div id="app-loader-text">Memuat data...</div>
            </div>
        </div>

        <script>
        window.showAppLoader = function(msg){
            const el = document.getElementById("app-loader");
            const txt = document.getElementById("app-loader-text");
            if(!el) return;
            txt.textContent = msg || "Memuat data...";
            el.classList.add("show");
        }
        window.hideAppLoader = function(){
            const el = document.getElementById("app-loader");
            if(el) el.classList.remove("show");
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

    st.session_state["_loader_injected"] = True


@contextmanager
def page_loading(message="Memuat data..."):
    safe = html.escape(message)
    st.markdown(
        f"<script>window.showAppLoader('{safe}');</script>",
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        st.markdown(
            "<script>window.hideAppLoader();</script>",
            unsafe_allow_html=True,
        )

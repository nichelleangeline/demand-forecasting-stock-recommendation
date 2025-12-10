# app/services/page_loader.py

from contextlib import contextmanager
import streamlit as st


def init_page_loader_css():
    """
    CSS loader yang nempel di area konten utama (block-container),
    bukan satu layar full. Sidebar tetap kelihatan.
    """
    st.markdown(
        """
        <style>
        /* Pastikan container utama bisa jadi anchor untuk overlay */
        .main .block-container {
            position: relative;
        }

        .page-loading-overlay {
            position: absolute;
            inset: 0;
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: none;  /* biar scroll lama masih keblok sementara */
        }

        .page-loading-backdrop {
            position: absolute;
            inset: 0;
            border-radius: 16px;
            background: linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.05),
                rgba(148, 163, 184, 0.08)
            );
            backdrop-filter: blur(2px);
        }

        .page-loading-card {
            position: relative;
            min-width: 260px;
            max-width: 380px;
            padding: 14px 18px;
            border-radius: 999px;
            background: #0f172a;
            color: #e5e7eb;
            box-shadow:
                0 10px 25px rgba(15, 23, 42, 0.25),
                0 0 0 1px rgba(148, 163, 184, 0.3);
            display: inline-flex;
            align-items: center;
            gap: 12px;
            pointer-events: auto;  /* biar tooltips dll nggak ke-disable */
        }

        .page-loading-spinner {
            width: 20px;
            height: 20px;
            border-radius: 999px;
            border: 2px solid rgba(148, 163, 184, 0.6);
            border-top-color: #facc15;
            animation: page-spin 0.7s linear infinite;
        }

        .page-loading-text {
            font-size: 0.85rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }

        @keyframes page-spin {
            to { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def page_loading(message: str = "Memuat data..."):
    """
    Pakai sebagai:

        from app.services.page_loader import page_loading

        with page_loading("Mengambil data dari database..."):
            # query berat di sini

    Overlay cuma muncul di area konten, sidebar tetap kelihatan.
    """
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class="page-loading-overlay">
          <div class="page-loading-backdrop"></div>
          <div class="page-loading-card">
            <div class="page-loading-spinner"></div>
            <div class="page-loading-text">{message}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()

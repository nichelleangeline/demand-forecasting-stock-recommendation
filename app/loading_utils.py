# app/ui/loading_utils.py

import streamlit as st


def init_loading_css():
    """
    Panggil sekali di awal setiap page.
    Inject CSS untuk overlay loading.
    """
    st.markdown(
        """
        <style>
        .overlay-loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.35);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .overlay-box {
            background: #ffffff;
            padding: 24px 32px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.25);
            text-align: center;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            animation: spin 1s linear infinite;
            margin: 0 auto 12px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_overlay(message: str):
    """
    Dipanggil internal: tampilkan overlay dengan pesan.
    """
    st.markdown(
        f"""
        <div class="overlay-loading">
          <div class="overlay-box">
            <div class="loader"></div>
            <div>{message}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_with_loader(
    key: str,
    button_label: str,
    message: str,
    fn,
    button_type: str = "primary",
):
    """
    Pattern umum:
      - Render tombol
      - Klik pertama -> set state "running" + rerun
      - Rerun berikutnya -> tampilkan overlay + jalankan fn()
      - Setelah selesai -> reset state + rerun lagi

    Cara pakai di page:
      def do_something():
          ... kerja berat, tulis ke DB, dll ...

      action_with_loader(
          key="save_penjualan",
          button_label="Simpan perubahan penjualan",
          message="Menyimpan perubahan penjualan ke database...",
          fn=do_something,
      )
    """
    state_key = f"{key}__state"
    msg_key = f"{key}__msg"

    # init state
    if state_key not in st.session_state:
        st.session_state[state_key] = "idle"
        st.session_state[msg_key] = ""

    # PHASE 2: kalau sedang running -> tampilkan overlay + eksekusi fn
    if st.session_state[state_key] == "running":
        _render_overlay(st.session_state[msg_key] or message)
        try:
            fn()
        finally:
            # selesai, reset dan rerun agar overlay hilang
            st.session_state[state_key] = "idle"
            st.session_state[msg_key] = ""
            st.rerun()

    # PHASE 1: render tombol
    if st.button(button_label, type=button_type, key=f"btn__{key}"):
        st.session_state[state_key] = "running"
        st.session_state[msg_key] = message
        st.rerun()

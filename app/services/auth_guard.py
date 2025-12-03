import streamlit as st

def require_login():
    """
    Kalau user belum login, redirect ke halaman login (myApp.py)
    """
    if "user" not in st.session_state:
        st.switch_page("myApp.py")

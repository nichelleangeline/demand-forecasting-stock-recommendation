import streamlit as st

def require_login():
    if "user" not in st.session_state:
        st.switch_page("myApp.py")

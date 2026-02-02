from __future__ import annotations

from typing import Any, Dict
import streamlit as st


def require_login() -> Dict[str, Any]:
    if "user" not in st.session_state:
        st.error("Silakan login dulu.")
        st.stop()

    user = st.session_state["user"]
    if not isinstance(user, dict):
        st.error("Session login tidak valid, silakan login ulang.")
        st.stop()

    return user


def get_current_user() -> Dict[str, Any] | None:
    user = st.session_state.get("user")
    if user is None:
        return None
    if not isinstance(user, dict):
        return None
    return user

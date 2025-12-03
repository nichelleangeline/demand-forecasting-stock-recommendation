# app/ui/components.py

import streamlit as st


def card(title: str, body_func, key: str | None = None):
    """
    Wrapper sederhana untuk "kartu" UI.

    body_func: fungsi tanpa argumen yang berisi isi kartu,
               dipanggil di dalam context kartu.
    """
    with st.container():
        st.markdown(f"#### {title}")
        st.write("")  # sedikit spacer
        body_func()


def metric_box(label: str, value, delta=None, help_text: str | None = None):
    """
    Wrapper kecil untuk st.metric.
    """
    st.metric(label=label, value=value, delta=delta, help=help_text)

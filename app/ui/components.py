import streamlit as st


def card(title: str, body_func, key: str | None = None):
    with st.container():
        st.markdown(f"#### {title}")
        st.write("")  
        body_func()


def metric_box(label: str, value, delta=None, help_text: str | None = None):
    st.metric(label=label, value=value, delta=delta, help=help_text)

from pathlib import Path

import streamlit as st


def render_workspace_ui():
    styles = (Path(__file__).resolve().parents[1] / "app" / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.page_link("app.py", label="Dashboard", icon="🔥")
        st.page_link("pages/1_Adversarial_Lab.py", label="Adversarial Lab", icon="🧪")


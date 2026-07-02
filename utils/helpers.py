from pathlib import Path
import streamlit as st


def load_css():

    css_path = Path("assets/css/style.css")

    if css_path.exists():

        with open(css_path) as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )


def page_title(title, subtitle=""):

    st.markdown(
        f"<div class='main-title'>{title}</div>",
        unsafe_allow_html=True
    )

    if subtitle:

        st.markdown(
            f"<div class='page-subtitle'>{subtitle}</div>",
            unsafe_allow_html=True
        )


def section_title(title):

    st.markdown(
        f"<div class='section-title'>{title}</div>",
        unsafe_allow_html=True
    )


def footer():

    st.markdown(
        """
        <div class='footer'>
        Explainable AI for Human-Centric Predictive Maintenance |
        MSc Data Science & Artificial Intelligence |
        University of Suffolk
        </div>
        """,
        unsafe_allow_html=True
    )
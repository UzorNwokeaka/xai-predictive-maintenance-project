import streamlit as st

from utils.helpers import (
    load_css,
    page_title,
    footer,
)

from utils.sidebar import render_sidebar


def render_page(
    title: str,
    subtitle: str = "",
):
    """
    Standard page layout.
    """

    load_css()

    render_sidebar()

    page_title(
        title,
        subtitle,
    )


def end_page():
    """
    Render footer.
    """

    st.divider()

    footer()
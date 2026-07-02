import streamlit as st


def render_sidebar():
    """
    Render the global application sidebar.
    """

    st.sidebar.markdown(
        """
        ## Explainable AI

        **Predictive Maintenance**

        ---
        **MSc Data Science & Artificial Intelligence**

        University of Suffolk

        ---
        **Version 1.1**
        """
    )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Human-Centric Decision Support for Industrial Predictive Maintenance"
    )
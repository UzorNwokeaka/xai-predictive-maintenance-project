import streamlit as st


def render_sidebar():
    """
    Render the global application sidebar.
    """

    st.sidebar.markdown(
        """
        ## Explainable AI (XAI)

        **Predictive Maintenance**

        ---
        **MSc Data Science & Artificial Intelligence**

        University of Suffolk

        ---
        **Version 1.0**
        """
    )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Human-Centred Decision Support for Industrial Predictive Maintenance"
    )
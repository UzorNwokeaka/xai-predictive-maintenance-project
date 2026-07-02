import streamlit as st

from config import APP_TITLE
from utils.data_loader import (
    load_dashboard_data,
    load_model_comparison,
    load_model_metadata,
)
from utils.layout import render_page, end_page
from utils.helpers import section_title


st.set_page_config(
    page_title=f"Home | {APP_TITLE}",
    layout="wide",
)

render_page(
    "Explainable AI for Human-Centric Predictive Maintenance",
    "Decision Support System for Remaining Useful Life prediction, explainability and maintenance planning."
)

dashboard_df = load_dashboard_data()
model_df = load_model_comparison()
metadata = load_model_metadata()

# =====================================================
# Research Overview
# =====================================================

section_title("Research Overview")

st.markdown(
    """
    This application implements an Explainable AI framework for predictive
    maintenance using NASA C-MAPSS turbofan engine data. It predicts Remaining
    Useful Life, translates predictions into health states, and provides
    maintenance recommendations to support human-centred decision-making.
    """
)

# =====================================================
# Key Project Metrics
# =====================================================

section_title("Key Project Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Dataset",
    metadata.get("dataset", "NASA C-MAPSS FD001"),
)

col2.metric(
    "Best Predictive Model",
    metadata.get("best_model", "Improved LSTM"),
)

col3.metric(
    "RMSE",
    f"{float(metadata.get('rmse', model_df['RMSE'].min())):.2f} cycles",
)

col4.metric(
    "Explainability Model",
    metadata.get("explainability_model", "Random Forest Tuned"),
)

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Engines Analysed",
    dashboard_df["engine_id"].nunique(),
)

col6.metric(
    "Window Size",
    f"{metadata.get('window_size', 30)} cycles",
)

col7.metric(
    "RUL Cap",
    metadata.get("rul_cap", 125),
)

col8.metric(
    "Features Used",
    metadata.get("feature_count", 17),
)

# =====================================================
# System Workflow
# =====================================================

section_title("System Workflow")

workflow_cols = st.columns(6)

workflow_steps = [
    "Industrial IoT Data",
    "Data Preparation",
    "Predictive Modelling",
    "Explainable AI",
    "Health Translation",
    "Decision Support",
]

for col, step in zip(workflow_cols, workflow_steps):
    with col:
        st.markdown(f"**{step}**")

st.markdown(
    """
    The system follows a structured workflow from industrial sensor data processing
    through to AI prediction, explanation, health-state classification and
    maintenance recommendation.
    """
)

# =====================================================
# Application Modules
# =====================================================

section_title("Application Modules")

m1, m2 = st.columns(2)

with m1:
    st.markdown("### Prediction")
    st.write("Estimate Remaining Useful Life using the trained LSTM model.")

    st.markdown("### Fleet Health")
    st.write("Review health status, risk levels and maintenance priority across the fleet.")

with m2:
    st.markdown("### Explainability")
    st.write("Explore global feature importance, local explanation and temporal trends.")

    st.markdown("### Model Comparison")
    st.write("Compare traditional ML and deep learning model performance.")

# =====================================================
# Decision-Support Positioning
# =====================================================

section_title("Human-Centred AI Positioning")

st.info(
    """
    The application is designed to support maintenance engineers and managers.
    It provides predictive insights and explanations, but final operational
    decisions remain under human control.
    """
)

end_page()
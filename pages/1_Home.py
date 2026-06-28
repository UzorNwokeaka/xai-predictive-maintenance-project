import streamlit as st

from config import APP_TITLE
from utils.data_loader import load_dashboard_data, load_model_comparison, load_model_metadata


st.set_page_config(page_title=f"Home | {APP_TITLE}", layout="wide")

st.title("Explainable AI for Human-Centric Predictive Maintenance")

st.markdown(
    """
    This application implements an Explainable AI decision-support framework for
    predictive maintenance using NASA C-MAPSS turbofan engine degradation data.

    The system predicts Remaining Useful Life, translates predictions into health
    states, provides maintenance recommendations, and presents explainability
    insights to support human-centred decision-making.
    """
)

dashboard_df = load_dashboard_data()
model_df = load_model_comparison()
metadata = load_model_metadata()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Dataset", metadata.get("dataset", "NASA C-MAPSS FD001"))
col2.metric("Engines Analysed", dashboard_df["engine_id"].nunique())
col3.metric("Best Model", metadata.get("best_model", "Improved LSTM"))
col4.metric("RMSE", round(float(metadata.get("rmse", model_df["RMSE"].min())), 2))

st.divider()

st.subheader("Framework Architecture")

st.markdown(
    """
    The implemented framework consists of six main layers:

    1. **Industrial IoT Data Layer** – NASA C-MAPSS turbofan sensor data.
    2. **Data Preparation Layer** – preprocessing, RUL generation, scaling and feature selection.
    3. **Predictive Modelling Layer** – machine learning and deep learning model development.
    4. **Explainable AI Layer** – global, local and temporal explanations.
    5. **Decision Translation Layer** – conversion of RUL into health status and risk level.
    6. **Human-Centred Decision Support Layer** – maintenance recommendation and dashboard.
    """
)

st.subheader("Application Modules")

st.markdown(
    """
    - **Prediction**: estimates Remaining Useful Life using the trained LSTM model.
    - **Explainability**: presents global feature importance, local engine explanation and temporal degradation trends.
    - **Fleet Health**: summarises health status and maintenance priority across the test fleet.
    - **Model Comparison**: compares machine learning and deep learning model performance.
    - **About**: describes the research context, dataset and implementation.
    """
)

st.info(
    "This system is designed to support maintenance engineers, not replace human judgement."
)
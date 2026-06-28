import streamlit as st

from config import APP_TITLE
from utils.data_loader import load_model_metadata


st.set_page_config(page_title=f"About | {APP_TITLE}", layout="wide")

st.title("About This Research Project")

metadata = load_model_metadata()

st.subheader("Dissertation Title")

st.markdown(
    """
    **Explainable AI for Human-Centric Predictive Maintenance Using Industrial IoT Sensor Data:
    A Case Study of Remaining Useful Life Prediction in NASA Turbofan Engines**
    """
)

st.subheader("Research Aim")

st.markdown(
    """
    The aim of this project is to design, implement and evaluate an Explainable AI
    framework that predicts Remaining Useful Life, provides interpretable insights,
    and supports human-centred predictive maintenance decision-making within an
    Industry 5.0 context.
    """
)

st.subheader("Dataset")

st.markdown(
    f"""
    The project uses the **{metadata.get("dataset", "NASA C-MAPSS FD001")}**
    dataset. The dataset contains multivariate turbofan engine sensor readings
    recorded over operational cycles. It is widely used for Remaining Useful Life
    prediction and predictive maintenance research.
    """
)

st.subheader("Final Model Configuration")

col1, col2, col3 = st.columns(3)

col1.metric("Best Model", metadata.get("best_model", "Improved LSTM"))
col2.metric("Window Size", metadata.get("window_size", 30))
col3.metric("RUL Cap", metadata.get("rul_cap", 125))

col4, col5, col6 = st.columns(3)

col4.metric("RMSE", round(float(metadata.get("rmse", 0)), 2))
col5.metric("MAE", round(float(metadata.get("mae", 0)), 2))
col6.metric("R²", round(float(metadata.get("r2", 0)), 3))

st.subheader("Technology Stack")

st.markdown(
    """
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - XGBoost
    - TensorFlow/Keras
    - SHAP / feature importance analysis
    - Streamlit
    - Plotly
    - Git and GitHub
    """
)

st.subheader("Industry 5.0 Alignment")

st.markdown(
    """
    The system aligns with Industry 5.0 by emphasising human-centred decision support.
    The AI model provides predictions and explanations, while maintenance engineers
    remain responsible for interpreting outputs and making final operational decisions.
    """
)

st.subheader("System Purpose")

st.success(
    """
    This application demonstrates how predictive modelling, explainability and
    maintenance recommendations can be integrated into a practical decision-support
    system for intelligent manufacturing and predictive maintenance environments.
    """
)
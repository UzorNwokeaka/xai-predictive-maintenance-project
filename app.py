import streamlit as st

from config import APP_TITLE
from utils.data_loader import load_dashboard_data, load_model_comparison, load_feature_importance


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)

dashboard_df = load_dashboard_data()
model_df = load_model_comparison()
feature_df = load_feature_importance()

st.title(APP_TITLE)

st.markdown(
    """
    This application demonstrates an Explainable AI framework for human-centric
    predictive maintenance using NASA C-MAPSS turbofan engine sensor data.
    """
)

col1, col2, col3 = st.columns(3)

col1.metric("Engines Analysed", dashboard_df["engine_id"].nunique())
col2.metric("Best Model", model_df.sort_values("RMSE").iloc[0]["Model"])
col3.metric("Best RMSE", round(model_df["RMSE"].min(), 2))

st.subheader("Research Workflow")

st.markdown(
    """
    1. Industrial IoT sensor data processing  
    2. Remaining Useful Life prediction  
    3. Explainable AI analysis  
    4. Health-state translation  
    5. Maintenance recommendation  
    6. Human-centred decision support  
    """
)

st.subheader("System Preview")

st.dataframe(dashboard_df.head(10), use_container_width=True)
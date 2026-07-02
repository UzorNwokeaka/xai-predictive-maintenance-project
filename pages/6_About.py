import streamlit as st

from config import APP_TITLE
from utils.layout import render_page, end_page
from utils.helpers import section_title
from utils.data_loader import load_model_metadata


st.set_page_config(
    page_title=f"About | {APP_TITLE}",
    layout="wide",
)

render_page(
    "About This Project",
    "Project overview, methodology and research contribution."
)

metadata = load_model_metadata()


# ==========================================================
# 1. Project Overview
# ==========================================================

section_title("1. Project Overview")

st.markdown(
    """
This application was developed as part of an MSc dissertation in Data Science
and Artificial Intelligence at the University of Suffolk.

The project investigates how Explainable Artificial Intelligence (XAI) can
support human-centred predictive maintenance for industrial assets using
NASA C-MAPSS turbofan engine sensor data.

The deployed system combines deep learning prediction,
health-state classification and explainability to support maintenance
decision-making.
"""
)

st.divider()


# ==========================================================
# 2. Research Information
# ==========================================================

section_title("2. Research Information")

left, right = st.columns(2)

with left:

    st.metric(
        "Dataset",
        metadata.get("dataset", "NASA C-MAPSS FD001")
    )

    st.metric(
        "Predictive Model",
        metadata.get("best_model", "Improved LSTM")
    )

    st.metric(
        "Explainability",
        metadata.get(
            "explainability_model",
            "Random Forest Tuned"
        )
    )

with right:

    st.metric(
        "Window Size",
        metadata.get("window_size", 30)
    )

    st.metric(
        "Feature Count",
        metadata.get("feature_count", 17)
    )

    st.metric(
        "Deployment",
        metadata.get(
            "deployment_platform",
            "Streamlit"
        )
    )

st.divider()


# ==========================================================
# 3. Decision Support Workflow
# ==========================================================

section_title("3. Decision Support Workflow")

workflow = st.columns(6)

steps = [
    "Industrial IoT",
    "Data Preparation",
    "Prediction",
    "Explainability",
    "Health Classification",
    "Decision Support",
]

for col, step in zip(workflow, steps):
    with col:
        st.markdown(f"**{step}**")

st.markdown(
    """
The application follows a structured workflow that transforms multivariate
sensor measurements into Remaining Useful Life predictions,
health classifications and maintenance recommendations.
"""
)

st.divider()


# ==========================================================
# 4. Research Contribution
# ==========================================================

section_title("4. Research Contribution")

st.markdown(
    """
The principal contributions of this research include:

- Development of a deep learning model for Remaining Useful Life prediction.
- Comparative evaluation of machine learning and deep learning approaches.
- Integration of Explainable AI techniques into predictive maintenance.
- Translation of AI predictions into health status and maintenance recommendations.
- Development of an interactive decision-support dashboard.
"""
)

st.divider()


# ==========================================================
# 5. Technologies
# ==========================================================

section_title("5. Technology Stacks")

st.markdown(
"""
- Python
- TensorFlow / Keras
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Streamlit
- Pandas
- NumPy
"""
)

st.divider()


# ==========================================================
# 6. Disclaimer
# ==========================================================

section_title("6. Disclaimer")

st.info(
"""
This application was developed for academic research purposes.

Predictions and recommendations are intended to support
maintenance engineers and decision-makers.

Final operational decisions should always remain under
human supervision.
"""
)

end_page()
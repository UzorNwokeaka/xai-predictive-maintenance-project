import streamlit as st

from config import APP_TITLE
from utils.layout import render_page, end_page
from utils.helpers import section_title
from utils.data_loader import load_model_metadata


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title=f"About | {APP_TITLE}",
    layout="wide",
)

render_page(
    "About This Project",
    "Project overview, research methodology, technical implementation and contribution."
)

metadata = load_model_metadata()


# ==========================================================
# 1. Project Overview
# ==========================================================

section_title("1. Project Overview")

st.markdown(
"""
This application was developed as part of an **MSc Data Science and Artificial
Intelligence dissertation** at the **University of Suffolk**.

The research investigates how **Explainable Artificial Intelligence (XAI)** can
support **human-centred predictive maintenance** by improving the transparency
and trustworthiness of Remaining Useful Life (RUL) prediction for industrial
equipment.

Using the NASA **C-MAPSS FD001** turbofan engine dataset, the project combines
deep learning prediction, Explainable AI and interactive decision-support
visualisation to help maintenance engineers understand not only **what**
the model predicts but also **why** those predictions are produced.
"""
)

st.divider()


# ==========================================================
# 2. Research Summary
# ==========================================================

section_title("2. Research Summary")

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
        "Primary XAI",
        "TimeSHAP"
    )

    st.metric(
        "Supplementary XAI",
        "Integrated Gradients"
    )

with right:

    st.metric(
        "Window Size",
        metadata.get("window_size", 30)
    )

    st.metric(
        "Selected Features",
        metadata.get("feature_count", 17)
    )

    st.metric(
        "Deployment Platform",
        metadata.get(
            "deployment_platform",
            "Streamlit"
        )
    )

    st.metric(
        "Dataset Engines",
        "100"
    )

st.divider()


# ==========================================================
# 3. Decision-Support Workflow
# ==========================================================

section_title("3. Decision-Support Workflow")

workflow = st.columns(7)

steps = [
    "IIoT Data",
    "Pre-processing",
    "Improved LSTM",
    "RUL Prediction",
    "TimeSHAP",
    "Health Assessment",
    "Decision Support",
]

for col, step in zip(workflow, steps):
    with col:
        st.markdown(f"**{step}**")

st.markdown(
"""
The application follows a complete Explainable AI workflow that transforms
multivariate engine sensor measurements into Remaining Useful Life predictions,
health classifications, maintenance recommendations and transparent AI
explanations to support engineering decision making.
"""
)

st.divider()


# ==========================================================
# 4. Research Objectives
# ==========================================================

section_title("4. Research Objectives")

st.markdown(
"""
The objectives of this research were to:

- develop an accurate Remaining Useful Life prediction model using deep learning;
- compare traditional machine learning and deep learning approaches;
- investigate Explainable Artificial Intelligence techniques for predictive maintenance;
- improve transparency and user trust in AI-assisted maintenance decisions;
- develop an interactive decision-support application for engineering practice.
"""
)

st.divider()


# ==========================================================
# 5. Research Contributions
# ==========================================================

section_title("5. Research Contributions")

st.markdown(
"""
The principal contributions of this dissertation include:

- development of an Improved LSTM model for Remaining Useful Life prediction;
- comparative evaluation of multiple machine learning and deep learning algorithms;
- direct explanation of the deployed LSTM using **TimeSHAP**;
- independent explanation validation using **Integrated Gradients**;
- translation of AI predictions into health status, risk level and maintenance recommendations;
- development of a complete Explainable AI decision-support dashboard;
- alignment of predictive maintenance with the principles of Industry 5.0 and human-centred AI.
"""
)

st.divider()


# ==========================================================
# 6. Technology Stack
# ==========================================================

section_title("6. Technology Stack")

st.markdown(
"""
### Machine Learning & Deep Learning

- Python
- TensorFlow / Keras
- Scikit-learn
- XGBoost

### Explainable AI

- TimeSHAP
- Integrated Gradients
- SHAP

### Data Science

- Pandas
- NumPy

### Visualisation & Deployment

- Plotly
- Streamlit
"""
)

st.divider()


# ==========================================================
# 7. Industry 5.0 Alignment
# ==========================================================

section_title("7. Industry 5.0 Alignment")

st.success(
"""
This project aligns with the principles of **Industry 5.0** by placing
human decision-makers at the centre of AI-assisted maintenance.

Rather than replacing maintenance engineers, the system provides
transparent predictions and interpretable explanations that improve
trust, accountability and informed decision making.

The integration of TimeSHAP and Integrated Gradients supports
responsible and trustworthy Artificial Intelligence for predictive
maintenance applications.
"""
)

st.divider()


# ==========================================================
# 8. Future Research
# ==========================================================

section_title("8. Future Research")

st.markdown(
"""
Future work may include:

- live Industrial IoT sensor streaming;
- Digital Twin integration;
- Edge AI deployment;
- Federated Learning for distributed predictive maintenance;
- online model updating;
- multi-failure mode prediction;
- deployment within real industrial maintenance environments.
"""
)

st.divider()


# ==========================================================
# 9. Disclaimer
# ==========================================================

section_title("9. Disclaimer")

st.info(
"""
This application was developed as an academic research prototype.

Predictions, explanations and maintenance recommendations are intended
to support maintenance engineers and operational decision-makers.

Final engineering decisions should always remain under appropriate
human supervision and organisational maintenance procedures.
"""
)

st.divider()


# ==========================================================
# 10. Research Information
# ==========================================================

section_title("10. Research Information")

st.markdown(
"""
**Dissertation Title**

*Explainable AI for Human-Centred Predictive Maintenance Using Industrial IoT
Sensor Data: A Case Study of Remaining Useful Life Prediction in NASA Turbofan
Engines.*

**Programme**

MSc Data Science and Artificial Intelligence

**Institution**

University of Suffolk

**Year**

2026

**Researcher**

Uzordinma Malcolm Nwokeaka
"""
)

end_page()
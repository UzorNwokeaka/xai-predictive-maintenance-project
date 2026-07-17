import streamlit as st

from config import (
    APP_TITLE,
    PRIMARY_XAI_METHOD,
    SUPPLEMENTARY_XAI_METHOD,
    PREDICTIVE_MODEL_NAME,
)

from utils.data_loader import (
    load_dashboard_data,
    load_model_comparison,
    load_model_metadata,
    load_xai_comparison_summary,
)

from utils.layout import render_page, end_page
from utils.helpers import section_title


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title=f"Home | {APP_TITLE}",
    layout="wide",
)

render_page(
    "Explainable AI for Human-Centred Predictive Maintenance",
    (
        "Decision-support system for Remaining Useful Life prediction, "
        "direct LSTM explainability and maintenance planning."
    ),
)


# ============================================================
# Load application artefacts
# ============================================================

dashboard_df = load_dashboard_data()
model_df = load_model_comparison()
metadata = load_model_metadata()
xai_summary = load_xai_comparison_summary()


# ============================================================
# Research Overview
# ============================================================

section_title("Research Overview")

st.markdown(
    """
    This application implements an end-to-end Explainable Artificial
    Intelligence framework for predictive maintenance using the NASA
    C-MAPSS FD001 turbofan engine dataset.

    The system predicts Remaining Useful Life using an Improved LSTM,
    explains the model directly using TimeSHAP and Integrated Gradients,
    translates predictions into health states and risk levels, and
    provides maintenance recommendations to support human-centred
    engineering decision-making.
    """
)


# ============================================================
# Key Project Metrics
# ============================================================

section_title("Key Project Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Dataset",
    metadata.get(
        "dataset",
        "NASA C-MAPSS FD001",
    ),
)

col2.metric(
    "Predictive Model",
    PREDICTIVE_MODEL_NAME,
)

col3.metric(
    "Best RMSE",
    (
        f"{float(metadata.get('rmse', model_df['RMSE'].min())):.2f} "
        "cycles"
    ),
)

col4.metric(
    "Primary XAI Method",
    PRIMARY_XAI_METHOD,
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
    "Features Used",
    metadata.get(
        "feature_count",
        17,
    ),
)

col8.metric(
    "Supplementary XAI",
    SUPPLEMENTARY_XAI_METHOD,
)


# ============================================================
# Explainability Evidence
# ============================================================

section_title("Explainability Evidence")

e1, e2, e3, e4 = st.columns(4)

rank_correlation = xai_summary.get(
    "feature_rank_spearman",
    0,
)

top_five_overlap = xai_summary.get(
    "top_five_feature_overlap",
    0,
)

timeshap_runtime = xai_summary.get(
    "timeshap_runtime_seconds",
    0,
)

ig_runtime = xai_summary.get(
    "integrated_gradients_runtime_seconds",
    0,
)

e1.metric(
    "Feature-Rank Agreement",
    f"{rank_correlation:.4f}",
)

e2.metric(
    "Top-Five Feature Overlap",
    f"{top_five_overlap:.0%}",
)

e3.metric(
    "TimeSHAP Runtime",
    f"{timeshap_runtime:.2f} sec",
)

e4.metric(
    "IG Runtime",
    f"{ig_runtime:.2f} sec",
)

st.info(
    """
    TimeSHAP and Integrated Gradients independently identified
    highly consistent degradation indicators, providing cross-method
    evidence that the explanations are robust and not dependent on
    a single attribution technique.
    """
)


# ============================================================
# System Workflow
# ============================================================

section_title("System Workflow")

workflow_cols = st.columns(6)

workflow_steps = [
    "Industrial IoT Data",
    "Data Preparation",
    "Improved LSTM",
    "Direct Explainability",
    "Health Translation",
    "Decision Support",
]

for col, step in zip(
    workflow_cols,
    workflow_steps,
):
    with col:
        st.markdown(f"**{step}**")

st.markdown(
    """
    The system follows a structured workflow from multivariate sensor
    processing through sequence-based RUL prediction, direct model
    explanation, health-state classification and maintenance
    recommendation.
    """
)


# ============================================================
# Application Modules
# ============================================================

section_title("Application Modules")

m1, m2 = st.columns(2)

with m1:
    st.markdown("### Prediction")

    st.write(
        (
            "Estimate Remaining Useful Life using the trained "
            "Improved LSTM and translate the result into health "
            "status, risk level and maintenance guidance."
        )
    )

    st.markdown("### Fleet Health")

    st.write(
        (
            "Review fleet-wide health status, Remaining Useful Life, "
            "risk levels and maintenance priorities."
        )
    )

with m2:
    st.markdown("### Explainability")

    st.write(
        (
            "Explore TimeSHAP feature, event and coalition explanations, "
            "Integrated Gradients feature and sensor-time attribution, "
            "faithfulness testing and critical degradation windows."
        )
    )

    st.markdown("### Model Comparison")

    st.write(
        (
            "Compare the performance of machine-learning and "
            "deep-learning models using RMSE, MAE and R²."
        )
    )


# ============================================================
# Research Contribution
# ============================================================

section_title("Research Contribution")

st.markdown(
    """
    The principal contribution of this system is the direct explanation
    of the best-performing Improved LSTM using two complementary
    explainability techniques.

    TimeSHAP provides sequence-aware feature, event and coalition
    explanations, while Integrated Gradients provides independent
    feature, temporal and sensor-time validation. The resulting
    explanations are converted into clear maintenance information
    designed to support, rather than replace, engineering judgement.
    """
)


# ============================================================
# Human-Centred AI Positioning
# ============================================================

section_title("Human-Centred AI Positioning")

st.info(
    """
    The application is designed to assist maintenance engineers and
    operational managers. Predictions, explanations and recommendations
    provide decision support, while responsibility for final maintenance
    actions remains with qualified human personnel.
    """
)

end_page()
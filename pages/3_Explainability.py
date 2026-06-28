import streamlit as st
import plotly.express as px

from config import APP_TITLE
from utils.data_loader import (
    load_dashboard_data,
    load_feature_importance,
    load_temporal_explainability,
)


st.set_page_config(
    page_title=f"Explainability | {APP_TITLE}",
    layout="wide",
)

st.title("Explainable AI Dashboard")

st.markdown(
    """
    This page presents global, local and temporal explainability outputs
    supporting the predictive maintenance decision-support framework.
    """
)

dashboard_df = load_dashboard_data()
feature_importance_df = load_feature_importance()
temporal_df = load_temporal_explainability()

st.subheader("Global Explainability")

top_n = st.slider(
    "Number of top features",
    min_value=5,
    max_value=min(17, len(feature_importance_df)),
    value=10,
)

top_features_df = feature_importance_df.head(top_n)

fig_global = px.bar(
    top_features_df.sort_values("Importance"),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Global Feature Importance from Explainable Reference Model",
)

st.plotly_chart(fig_global, use_container_width=True)

st.dataframe(top_features_df, use_container_width=True)

st.divider()

st.subheader("Local Engine-Level Explanation")

engine_ids = dashboard_df["engine_id"].sort_values().unique()

selected_engine = st.selectbox("Select Engine ID", engine_ids)

engine_record = dashboard_df[
    dashboard_df["engine_id"] == selected_engine
].iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Actual RUL", f"{engine_record['actual_rul']:.2f} cycles")
c2.metric("Predicted RUL", f"{engine_record['predicted_rul_lstm']:.2f} cycles")
c3.metric("Health Status", engine_record["health_status"])
c4.metric("Risk Level", engine_record["risk_level"])

st.info(
    f"""
    Engine **{int(selected_engine)}** is predicted to have approximately
    **{engine_record['predicted_rul_lstm']:.2f} operational cycles**
    remaining.

    The engine is classified as **{engine_record['health_status']}**
    with a **{engine_record['risk_level']}** level.

    Recommended action: **{engine_record['recommendation']}**
    """
)

st.divider()

st.subheader("Temporal Explainability")

available_temporal_engines = temporal_df["engine_id"].sort_values().unique()

selected_temporal_engine = st.selectbox(
    "Select engine for temporal trend analysis",
    available_temporal_engines,
)

engine_temporal_df = temporal_df[
    temporal_df["engine_id"] == selected_temporal_engine
].sort_values("cycle")

feature_columns = [
    col for col in engine_temporal_df.columns
    if col not in ["engine_id", "cycle"]
]

selected_temporal_features = st.multiselect(
    "Select features to visualise over time",
    feature_columns,
    default=feature_columns[: min(3, len(feature_columns))],
)

if selected_temporal_features:
    temporal_long_df = engine_temporal_df.melt(
        id_vars=["engine_id", "cycle"],
        value_vars=selected_temporal_features,
        var_name="Feature",
        value_name="Scaled Value",
    )

    fig_temporal = px.line(
        temporal_long_df,
        x="cycle",
        y="Scaled Value",
        color="Feature",
        markers=True,
        title=f"Last Operational Window Trend - Engine {int(selected_temporal_engine)}",
    )

    st.plotly_chart(fig_temporal, use_container_width=True)

    st.markdown(
        """
        The chart shows how the selected high-importance sensor and setting
        variables evolved during the last operational window used for prediction.
        Rising, falling or unstable patterns may indicate degradation behaviour
        contributing to the Remaining Useful Life estimate.
        """
    )
else:
    st.warning("Select at least one feature to display temporal trends.")
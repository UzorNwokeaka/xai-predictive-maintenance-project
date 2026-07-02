import streamlit as st
import plotly.express as px

from config import APP_TITLE
from utils.data_loader import (
    load_dashboard_data,
    load_feature_importance,
    load_temporal_explainability,
    load_model_metadata,
)
from utils.layout import render_page, end_page
from utils.helpers import section_title


st.set_page_config(
    page_title=f"Explainability | {APP_TITLE}",
    layout="wide",
)

render_page(
    "Explainable AI Dashboard",
    "Understand global, local and temporal factors supporting Remaining Useful Life prediction."
)

dashboard_df = load_dashboard_data()
feature_importance_df = load_feature_importance()
temporal_df = load_temporal_explainability()
metadata = load_model_metadata()


# ==========================================================
# 1. Global Explainability
# ==========================================================

section_title("1. Global Explainability")

st.markdown(
    """
    Global explainability identifies the sensor and operational variables that
    contribute most strongly to Remaining Useful Life prediction across the fleet.
    """
)

top_n = st.slider(
    "Number of top features to display",
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

top_5_features = feature_importance_df["Feature"].head(5).tolist()

st.info(
    f"""
    **Engineering Interpretation**

    The most influential variables identified by the explainability layer are:
    **{", ".join(top_5_features)}**.

    These variables provide the strongest global indication of degradation-related
    behaviour within the tuned Random Forest explainability model.
    """
)

st.divider()


# ==========================================================
# 2. Local Explainability
# ==========================================================

section_title("2. Local Engine-Level Explanation")

st.markdown(
    """
    Local explainability focuses on a selected engine and links its predicted RUL
    to health status, risk level and recommended maintenance action.
    """
)

engine_ids = dashboard_df["engine_id"].sort_values().unique()

selected_engine = st.selectbox(
    "Select Engine ID",
    engine_ids,
    key="local_engine_selector",
)

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
    **Engine {int(selected_engine)} Decision Explanation**

    The Improved LSTM predicts approximately
    **{engine_record['predicted_rul_lstm']:.2f} operational cycles**
    remaining for this engine.

    Based on this prediction, the engine is classified as
    **{engine_record['health_status']}** with a **{engine_record['risk_level']}**
    level.

    Recommended action: **{engine_record['recommendation']}**
    """
)

st.divider()


# ==========================================================
# 3. Temporal Explainability
# ==========================================================

section_title("3. Temporal Explainability")

st.markdown(
    """
    Temporal explainability shows how important sensor and operational variables
    changed during the final operational window used for prediction.
    """
)

available_temporal_engines = temporal_df["engine_id"].sort_values().unique()

selected_temporal_engine = st.selectbox(
    "Select engine for temporal trend analysis",
    available_temporal_engines,
    key="temporal_engine_selector",
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

    trend_summaries = []

    for feature in selected_temporal_features:
        start_value = engine_temporal_df[feature].iloc[0]
        end_value = engine_temporal_df[feature].iloc[-1]
        change = end_value - start_value

        if change > 0.05:
            trend = "increased"
        elif change < -0.05:
            trend = "decreased"
        else:
            trend = "remained relatively stable"

        trend_summaries.append(
            f"{feature} {trend} over the final operational window"
        )

    st.info(
        f"""
        **Engineering Interpretation**

        For Engine **{int(selected_temporal_engine)}**, the selected variables show
        the following temporal behaviour:

        - {"; ".join(trend_summaries)}.

        These patterns help maintenance engineers understand how sensor behaviour
        evolved before the RUL prediction was produced.
        """
    )

else:
    st.warning("Select at least one feature to display temporal trends.")

st.divider()


# ==========================================================
# 4. Explainability Summary
# ==========================================================

section_title("4. Explainability Summary")

left, right = st.columns(2)

with left:
    st.metric("Predictive Model", metadata.get("best_model", "Improved LSTM"))
    st.metric("Dataset", metadata.get("dataset", "NASA C-MAPSS FD001"))

with right:
    st.metric("Explainability Model", metadata.get("explainability_model", "Random Forest Tuned"))
    st.metric("Explanation Types", "Global, Local, Temporal")

st.caption(
    """
    The deployed application uses the Improved LSTM as the final predictive model
    and the tuned Random Forest as the explainable reference model. Together, these
    support transparent, human-centred predictive maintenance decision-making.
    """
)

end_page()
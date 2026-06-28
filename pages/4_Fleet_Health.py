import streamlit as st
import plotly.express as px

from config import APP_TITLE
from utils.data_loader import load_dashboard_data

st.set_page_config(
    page_title=f"Fleet Health | {APP_TITLE}",
    layout="wide"
)

st.title("Fleet Health Monitoring Dashboard")

st.markdown("""
This dashboard provides a fleet-wide overview of engine health,
predicted Remaining Useful Life (RUL), maintenance priority,
and recommended maintenance actions.
""")

df = load_dashboard_data()

# ==========================================================
# Fleet Summary
# ==========================================================

healthy = (df["health_status"] == "Healthy").sum()
warning = (df["health_status"] == "Warning").sum()
critical = (df["health_status"] == "Critical").sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Engines",
    df["engine_id"].nunique()
)

col2.metric(
    "Healthy",
    healthy
)

col3.metric(
    "Warning",
    warning
)

col4.metric(
    "Critical",
    critical
)

st.metric(
    "Average Remaining Useful Life (RUL)",
    f"{df['predicted_rul_lstm'].mean():.2f} cycles"
)

st.divider()

# ==========================================================
# Fleet Health Distribution
# ==========================================================

left, right = st.columns(2)

with left:

    health_counts = (
        df["health_status"]
        .value_counts()
        .reset_index()
    )

    health_counts.columns = [
        "Health Status",
        "Count"
    ]

    fig_health = px.pie(
        health_counts,
        names="Health Status",
        values="Count",
        hole=0.45,
        title="Fleet Health Distribution"
    )

    st.plotly_chart(
        fig_health,
        use_container_width=True
    )

with right:

    risk_counts = (
        df["risk_level"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Count"
    ]

    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        text="Count",
        title="Fleet Risk Distribution"
    )

    fig_risk.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

st.divider()

# ==========================================================
# Remaining Useful Life Distribution
# ==========================================================

fig_hist = px.histogram(
    df,
    x="predicted_rul_lstm",
    nbins=25,
    title="Distribution of Predicted Remaining Useful Life"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

st.divider()

# ==========================================================
# Maintenance Priority List
# ==========================================================

st.subheader("Highest Maintenance Priority")

priority_df = (
    df.sort_values("predicted_rul_lstm")
    .head(10)
)

priority_df = priority_df[
    [
        "engine_id",
        "predicted_rul_lstm",
        "health_status",
        "risk_level",
        "recommendation"
    ]
]

priority_df.columns = [
    "Engine ID",
    "Predicted RUL",
    "Health",
    "Risk",
    "Recommended Action"
]

st.dataframe(
    priority_df,
    use_container_width=True
)

st.divider()

# ==========================================================
# Fleet Prediction Table
# ==========================================================

st.subheader("Complete Fleet Predictions")

st.dataframe(
    df.sort_values("predicted_rul_lstm"),
    use_container_width=True
)
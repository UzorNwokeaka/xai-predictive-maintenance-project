import streamlit as st
import plotly.express as px

from config import APP_TITLE
from utils.layout import render_page, end_page
from utils.helpers import section_title
from utils.data_loader import load_dashboard_data


st.set_page_config(
    page_title=f"Fleet Health | {APP_TITLE}",
    layout="wide",
)

render_page(
    "Fleet Health Dashboard",
    "Monitor fleet condition and maintenance priorities using AI predictions."
)

dashboard_df = load_dashboard_data()


# ==========================================================
# 1. Fleet Overview
# ==========================================================

section_title("1. Fleet Overview")

total_engines = dashboard_df["engine_id"].nunique()

healthy = (
    dashboard_df["health_status"] == "Healthy"
).sum()

warning = (
    dashboard_df["health_status"] == "Warning"
).sum()

critical = (
    dashboard_df["health_status"] == "Critical"
).sum()

avg_rul = dashboard_df["predicted_rul_lstm"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Engines", total_engines)
c2.metric("Healthy", healthy)
c3.metric("Warning", warning)
c4.metric("Critical", critical)

st.metric(
    "Average Predicted Remaining Useful Life",
    f"{avg_rul:.2f} cycles",
)

st.divider()


# ==========================================================
# 2. Fleet Health Distribution
# ==========================================================

section_title("2. Fleet Health Distribution")

health_counts = (
    dashboard_df["health_status"]
    .value_counts()
    .reset_index()
)

health_counts.columns = [
    "Health Status",
    "Count",
]

fig_health = px.pie(
    health_counts,
    names="Health Status",
    values="Count",
    hole=0.55,
    title="Fleet Health Classification",
    color="Health Status",
    color_discrete_map={
        "Healthy": "#2E8B57",     # Green
        "Warning": "#87CEFA",     # Light Blue
        "Critical": "#DC143C"     # Crimson Red
    },
)

st.plotly_chart(
    fig_health,
    use_container_width=True,
)

st.divider()


# ==========================================================
# 3. Remaining Useful Life Distribution
# ==========================================================

section_title("3. Remaining Useful Life Distribution")

fig_rul = px.histogram(
    dashboard_df,
    x="predicted_rul_lstm",
    nbins=20,
    title="Distribution of Predicted Remaining Useful Life",
)

fig_rul.update_layout(
    xaxis_title="Predicted Remaining Useful Life (Cycles)",
    yaxis_title="Number of Engines",
)

st.plotly_chart(
    fig_rul,
    use_container_width=True,
)

st.divider()


# ==========================================================
# 4. Engines Requiring Immediate Attention
# ==========================================================

section_title("4. Engines Requiring Immediate Attention")

priority_df = dashboard_df.sort_values(
    "predicted_rul_lstm"
).head(10)

st.dataframe(
    priority_df[
        [
            "engine_id",
            "predicted_rul_lstm",
            "health_status",
            "risk_level",
            "recommendation",
        ]
    ],
    use_container_width=True,
)

st.divider()


# ==========================================================
# 5. Fleet Summary
# ==========================================================

section_title("5. Fleet Summary")

critical_pct = (critical / total_engines) * 100
warning_pct = (warning / total_engines) * 100
healthy_pct = (healthy / total_engines) * 100

st.info(
    f"""
### Engineering Interpretation

The AI system analysed **{total_engines} engines** within the fleet.

- **Healthy:** {healthy} engines ({healthy_pct:.1f}%)
- **Warning:** {warning} engines ({warning_pct:.1f}%)
- **Critical:** {critical} engines ({critical_pct:.1f}%)

The histogram illustrates the distribution of Remaining Useful Life predictions,
while the priority table highlights the engines requiring the earliest maintenance
intervention.

This dashboard enables maintenance managers to prioritise inspection,
maintenance scheduling and operational planning using AI-assisted decision support.
"""
)

st.divider()

# ==========================================================
# 6. Prediction Summary for All Engines
# ==========================================================

section_title("6. Prediction Summary for All Engines")

all_engines_df = dashboard_df[
    [
        "engine_id",
        "predicted_rul_lstm",
        "health_status",
        "risk_level",
        "recommendation",
    ]
].sort_values("predicted_rul_lstm")

all_engines_df.columns = [
    "Engine ID",
    "Predicted RUL",
    "Health Status",
    "Risk Level",
    "Recommendation",
]

st.dataframe(
    all_engines_df,
    use_container_width=True,
)

end_page()
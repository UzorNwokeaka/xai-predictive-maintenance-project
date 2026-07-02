import streamlit as st
import plotly.express as px

from config import APP_TITLE
from utils.layout import render_page, end_page
from utils.helpers import section_title
from utils.data_loader import load_model_comparison, load_model_metadata


st.set_page_config(
    page_title=f"Model Comparison | {APP_TITLE}",
    layout="wide",
)

render_page(
    "Model Performance Comparison",
    "Evaluate why the Improved LSTM was selected as the final predictive model."
)

model_df = load_model_comparison().sort_values("RMSE")
metadata = load_model_metadata()

best_model = model_df.iloc[0]


# ==========================================================
# 1. Best Model Summary
# ==========================================================

section_title("1. Best Model Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Best Model", best_model["Model"])
c2.metric("RMSE", f"{best_model['RMSE']:.2f}")
c3.metric("MAE", f"{best_model['MAE']:.2f}")
c4.metric("R²", f"{best_model['R2']:.3f}")

st.info(
    f"""
    The best-performing model is **{best_model['Model']}**, achieving an RMSE of
    **{best_model['RMSE']:.2f} cycles**, MAE of **{best_model['MAE']:.2f} cycles**
    and R² of **{best_model['R2']:.3f}**.

    This indicates that the final model provides the strongest predictive
    performance among the evaluated traditional machine learning and deep learning
    approaches.
    """
)

st.divider()


# ==========================================================
# 2. RMSE Comparison
# ==========================================================

section_title("2. RMSE Comparison")

fig_rmse = px.bar(
    model_df,
    x="Model",
    y="RMSE",
    color="Experiment_Type",
    text="RMSE",
    title="Model Comparison by RMSE",
)

fig_rmse.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

fig_rmse.update_layout(
    xaxis_title="Model",
    yaxis_title="RMSE",
    xaxis_tickangle=-35,
)

st.plotly_chart(fig_rmse, use_container_width=True)

st.divider()


# ==========================================================
# 3. MAE Comparison
# ==========================================================

section_title("3. MAE Comparison")

fig_mae = px.bar(
    model_df,
    x="Model",
    y="MAE",
    color="Experiment_Type",
    text="MAE",
    title="Model Comparison by MAE",
)

fig_mae.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

fig_mae.update_layout(
    xaxis_title="Model",
    yaxis_title="MAE",
    xaxis_tickangle=-35,
)

st.plotly_chart(fig_mae, use_container_width=True)

st.divider()


# ==========================================================
# 4. R² Comparison
# ==========================================================

section_title("4. R² Score Comparison")

fig_r2 = px.bar(
    model_df,
    x="Model",
    y="R2",
    color="Experiment_Type",
    text="R2",
    title="Model Comparison by R² Score",
)

fig_r2.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
)

fig_r2.update_layout(
    xaxis_title="Model",
    yaxis_title="R² Score",
    xaxis_tickangle=-35,
)

st.plotly_chart(fig_r2, use_container_width=True)

st.divider()


# ==========================================================
# 5. Full Experiment Results
# ==========================================================

section_title("5. Full Experiment Results")

st.dataframe(
    model_df,
    use_container_width=True,
)

st.divider()


# ==========================================================
# 6. Evaluation Interpretation
# ==========================================================

section_title("6. Evaluation Interpretation")

st.markdown(
    f"""
    The model comparison confirms that **{metadata.get("best_model", "Improved LSTM")}**
    achieved the strongest overall performance. This supports the use of a recurrent
    deep learning architecture for modelling temporal degradation patterns in
    multivariate turbofan sensor data.

    The tuned Random Forest and XGBoost models also performed competitively,
    demonstrating that traditional ensemble learning methods remain effective for
    predictive maintenance tasks. However, the Improved LSTM achieved the lowest
    prediction error and was therefore selected as the final predictive model for
    the deployed decision-support system.

    The tuned Random Forest was retained as the explainable reference model because
    it provides stable global feature-importance outputs that support transparent
    interpretation of maintenance-related sensor contributors.
    """
)

end_page()
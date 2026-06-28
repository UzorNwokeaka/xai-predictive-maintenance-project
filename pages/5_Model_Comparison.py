import streamlit as st
import plotly.express as px

from config import APP_TITLE
from utils.data_loader import load_model_comparison


st.set_page_config(
    page_title=f"Model Comparison | {APP_TITLE}",
    layout="wide"
)

st.title("Model Performance Comparison")

model_df = load_model_comparison().sort_values("RMSE")

st.dataframe(model_df, use_container_width=True)

fig = px.bar(
    model_df,
    x="Model",
    y="RMSE",
    color="Experiment_Type",
    text="RMSE",
    title="Model Comparison by RMSE"
)

fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig.update_layout(xaxis_tickangle=-35)

st.plotly_chart(fig, use_container_width=True)

best_model = model_df.iloc[0]

st.success(
    f"The best performing model is {best_model['Model']} "
    f"with RMSE = {best_model['RMSE']:.2f}, "
    f"MAE = {best_model['MAE']:.2f}, "
    f"and R² = {best_model['R2']:.3f}."
)
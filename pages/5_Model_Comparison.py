import streamlit as st
import plotly.express as px

from config import APP_TITLE
from utils.layout import render_page, end_page
from utils.helpers import section_title
from utils.data_loader import (
    load_model_comparison,
    load_model_metadata,
)


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title=f"Model Comparison | {APP_TITLE}",
    layout="wide",
)

render_page(
    "Model Performance Comparison",
    (
        "Compare the evaluated machine learning and deep learning "
        "models and review why the Improved LSTM was selected as "
        "the final predictive model."
    ),
)


# ==========================================================
# Load Evaluation Results
# ==========================================================

model_df = (
    load_model_comparison()
    .sort_values("RMSE")
    .reset_index(drop=True)
)

metadata = load_model_metadata()

best_model = model_df.iloc[0]


# ==========================================================
# 1. Best Model Summary
# ==========================================================

section_title("1. Best Model Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Best Model",
    best_model["Model"],
)

c2.metric(
    "RMSE",
    f"{best_model['RMSE']:.2f} cycles",
)

c3.metric(
    "MAE",
    f"{best_model['MAE']:.2f} cycles",
)

c4.metric(
    "R²",
    f"{best_model['R2']:.3f}",
)

st.info(
    f"""
    The **{best_model['Model']}** achieved the best predictive
    performance, with an RMSE of **{best_model['RMSE']:.2f} cycles**,
    an MAE of **{best_model['MAE']:.2f} cycles**, and an R² score of
    **{best_model['R2']:.3f}**.

    These results demonstrate the advantage of sequence-based deep
    learning for modelling temporal degradation in turbofan engines.
    Consequently, the Improved LSTM was selected as the deployed
    predictive model for the Explainable AI decision-support system.
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
    yaxis_title="RMSE in cycles",
    xaxis_tickangle=-35,
)

st.plotly_chart(
    fig_rmse,
    use_container_width=True,
)

st.caption(
    """
    Lower RMSE values indicate better overall predictive accuracy.
    The Improved LSTM achieved the lowest RMSE among the evaluated models.
    """
)

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
    yaxis_title="MAE in cycles",
    xaxis_tickangle=-35,
)

st.plotly_chart(
    fig_mae,
    use_container_width=True,
)

st.caption(
    """
    Lower MAE values indicate smaller average prediction errors.
    MAE is expressed in operational cycles.
    """
)

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

st.plotly_chart(
    fig_r2,
    use_container_width=True,
)

st.caption(
    """
    Higher R² values indicate that a larger proportion of the variation
    in Remaining Useful Life is explained by the model.
    """
)

st.divider()


# ==========================================================
# 5. Full Experiment Results
# ==========================================================

section_title("5. Full Experiment Results")

display_columns = [
    column
    for column in [
        "Model",
        "RMSE",
        "MAE",
        "R2",
        "Experiment_Type",
    ]
    if column in model_df.columns
]

st.dataframe(
    model_df[display_columns],
    use_container_width=True,
    hide_index=True,
)

st.divider()


# ==========================================================
# 6. Evaluation Interpretation
# ==========================================================

section_title("6. Evaluation Interpretation")

st.markdown(
    f"""
The comparative evaluation demonstrates that
**{metadata.get("best_model", "Improved LSTM")}**
achieved the strongest overall predictive performance across the
evaluated machine learning and deep learning models.

The Improved LSTM produced the lowest overall prediction error while
capturing temporal degradation patterns present in multivariate
turbofan sensor sequences. This makes the recurrent architecture
particularly suitable for Remaining Useful Life prediction in
Industrial Internet of Things predictive-maintenance applications.

Traditional ensemble models, including the Tuned Random Forest and
Tuned XGBoost, also produced competitive results and provided valuable
benchmarks against which the deep learning models were evaluated.

The final Explainable AI framework explains the deployed Improved LSTM
directly. **TimeSHAP** is used as the primary explainability method,
while **Integrated Gradients** provides supplementary validation of the
resulting feature and temporal attributions.

Together, the two explanation methods support:

- global feature attribution;
- local engine-level interpretation;
- temporal event attribution;
- critical degradation-window identification;
- sensor-time attribution;
- feature-deletion faithfulness analysis.

This combination provides accurate Remaining Useful Life prediction
together with transparent and human-centred maintenance decision
support.
"""
)

st.divider()


# ==========================================================
# 7. Final Model Selection
# ==========================================================

section_title("7. Final Model Selection")

selection_col1, selection_col2 = st.columns(2)

with selection_col1:
    st.success(
        """
### Deployed Predictive Model

**Improved LSTM**

The Improved LSTM is the final model used by the application to predict
Remaining Useful Life from multivariate operational and sensor data.

It was selected because it achieved the strongest overall predictive
performance among the evaluated models.
"""
    )

with selection_col2:
    st.success(
        """
### Explainable AI Framework

**Primary method: TimeSHAP**

**Supplementary method: Integrated Gradients**

Both methods explain the deployed Improved LSTM directly and provide
complementary feature-level and temporal insights for maintenance
decision support.
"""
    )


# ==========================================================
# Page Footer
# ==========================================================

end_page()
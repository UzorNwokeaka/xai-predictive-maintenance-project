import streamlit as st

from config import APP_TITLE
from utils.layout import render_page, end_page
from utils.helpers import section_title
from utils.data_loader import (
    load_lstm_model,
    load_scaler,
    load_selected_features,
    load_feature_importance,
)
from utils.prediction import predict_rul, generate_decision_output


st.set_page_config(
    page_title=f"Prediction | {APP_TITLE}",
    layout="wide",
)

render_page(
    "AI Predictive Maintenance Decision Support",
    "Predict Remaining Useful Life, understand AI outputs and support maintenance planning."
)


# ==========================================================
# Load AI Artefacts
# ==========================================================

lstm_model = load_lstm_model()
scaler = load_scaler()
selected_features = load_selected_features()
feature_importance = load_feature_importance()


# ==========================================================
# Reset Logic
# ==========================================================

if "prediction_completed" not in st.session_state:
    st.session_state.prediction_completed = False

if "input_reset_counter" not in st.session_state:
    st.session_state.input_reset_counter = 0


def clear_prediction_page():
    st.session_state.prediction_completed = False
    st.session_state.input_reset_counter += 1


reset_id = st.session_state.input_reset_counter


# ==========================================================
# 1. Operational Settings
# ==========================================================

section_title("1. Operational Settings")

settings = {}

setting_cols = st.columns(3)

setting_names = ["setting_1", "setting_2", "setting_3"]

for idx, setting in enumerate(setting_names):
    with setting_cols[idx]:
        settings[setting] = st.number_input(
            setting.replace("_", " ").title(),
            value=0.0,
            format="%.3f",
            key=f"input_{setting}_{reset_id}",
        )


# ==========================================================
# 2. Sensor Measurements
# ==========================================================

section_title("2. Sensor Measurements")

sensor_inputs = {}

sensor_features = [
    feature for feature in selected_features
    if feature.startswith("sensor")
]

left, right = st.columns(2)

for idx, sensor in enumerate(sensor_features):
    target_col = left if idx < 7 else right

    with target_col:
        sensor_inputs[sensor] = st.number_input(
            sensor.replace("_", " ").title(),
            value=0.0,
            format="%.3f",
            key=f"input_{sensor}_{reset_id}",
        )


# ==========================================================
# 3. AI Inference
# ==========================================================

section_title("3. AI Inference")

predict_col, clear_col = st.columns(2)

with predict_col:
    predict_clicked = st.button(
        "Predict Remaining Useful Life",
        use_container_width=True,
    )

with clear_col:
    clear_clicked = st.button(
        "Clear Inputs",
        use_container_width=True,
    )


if clear_clicked:
    clear_prediction_page()
    st.rerun()


if predict_clicked:
    ordered_values = []

    for feature in selected_features:
        if feature.startswith("setting"):
            ordered_values.append(settings[feature])
        else:
            ordered_values.append(sensor_inputs[feature])

    predicted_rul = predict_rul(
        input_values=ordered_values,
        lstm_model=lstm_model,
        scaler=scaler,
        selected_features=selected_features,
    )

    decision = generate_decision_output(predicted_rul)

    st.session_state.prediction_completed = True
    st.session_state.prediction_rul = decision["predicted_rul"]
    st.session_state.prediction_health_status = decision["health_status"]
    st.session_state.prediction_risk_level = decision["risk_level"]
    st.session_state.prediction_recommendation = decision["recommendation"]


# ==========================================================
# 4. AI Prediction Summary
# ==========================================================

if st.session_state.prediction_completed:
    st.divider()

    section_title("4. AI Prediction Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted RUL",
        f"{st.session_state.prediction_rul:.2f} cycles",
    )

    c2.metric(
        "Health Status",
        st.session_state.prediction_health_status,
    )

    c3.metric(
        "Risk Level",
        st.session_state.prediction_risk_level,
    )


    # ======================================================
    # 5. Maintenance Recommendation
    # ======================================================

    section_title("5. Maintenance Recommendation")

    st.info(st.session_state.prediction_recommendation)

    st.markdown(
        f"""
        The trained LSTM model predicts that this engine has approximately
        **{st.session_state.prediction_rul:.2f} operational cycles** remaining.

        Based on this output, the engine is classified as
        **{st.session_state.prediction_health_status}** with a
        **{st.session_state.prediction_risk_level}** level.

        The recommendation above translates the model output into an actionable
        maintenance decision-support statement.
        """
    )


    # ======================================================
    # 6. Explainability Reference
    # ======================================================

    section_title("6. Explainability Reference")

    st.markdown(
        """
        The final predictive model is the **Improved LSTM**.  
        The explainability reference model is the **Tuned Random Forest**, which
        provides global feature-importance insights for the decision-support layer.
        """
    )

    st.dataframe(
        feature_importance.head(5),
        use_container_width=True,
    )

else:
    st.divider()
    st.info(
        """
        Enter operational settings and sensor measurements, then click
        **Predict Remaining Useful Life** to generate a prediction and
        maintenance recommendation.
        """
    )


end_page()
import streamlit as st

from config import APP_TITLE
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


def clear_prediction_page():
    st.session_state.prediction_completed = False
    st.session_state.input_reset_counter += 1


st.title("AI Predictive Maintenance Decision Support")

st.markdown(
    """
    Provide the engine operational settings and sensor measurements below.
    The trained LSTM model will estimate the Remaining Useful Life (RUL)
    and generate a maintenance recommendation.
    """
)


lstm_model = load_lstm_model()
scaler = load_scaler()
selected_features = load_selected_features()
feature_importance = load_feature_importance()


if "prediction_completed" not in st.session_state:
    st.session_state.prediction_completed = False

if "input_reset_counter" not in st.session_state:
    st.session_state.input_reset_counter = 0


reset_id = st.session_state.input_reset_counter


st.subheader("Operational Settings")

settings = {}
setting_names = ["setting_1", "setting_2", "setting_3"]
setting_cols = st.columns(3)

for i, setting in enumerate(setting_names):
    with setting_cols[i]:
        settings[setting] = st.number_input(
            label=setting.replace("_", " ").title(),
            value=0.0,
            step=0.01,
            key=f"input_{setting}_{reset_id}",
        )


st.subheader("Sensor Measurements")

sensor_inputs = {}

sensor_features = [
    feature for feature in selected_features
    if feature.startswith("sensor")
]

sensor_cols = st.columns(3)

for i, sensor in enumerate(sensor_features):
    with sensor_cols[i % 3]:
        sensor_inputs[sensor] = st.number_input(
            label=sensor.replace("_", " ").title(),
            value=0.0,
            step=0.01,
            key=f"input_{sensor}_{reset_id}",
        )


st.divider()

clear_col, predict_col = st.columns([1, 2])

with clear_col:
    clear_clicked = st.button(
        "Clear All",
        use_container_width=True,
    )

with predict_col:
    predict_clicked = st.button(
        "Predict Remaining Useful Life",
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

    results = generate_decision_output(predicted_rul)

    st.session_state.prediction_completed = True
    st.session_state.prediction_rul = results["predicted_rul"]
    st.session_state.prediction_health_status = results["health_status"]
    st.session_state.prediction_risk_level = results["risk_level"]
    st.session_state.prediction_recommendation = results["recommendation"]


if st.session_state.prediction_completed:
    st.divider()

    st.subheader("Prediction Results")

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

    st.success(st.session_state.prediction_recommendation)

    st.subheader("Decision Summary")

    st.info(
        f"""
        The trained LSTM model predicts that this engine has approximately
        **{st.session_state.prediction_rul:.2f} operational cycles** remaining
        before failure.

        Based on this prediction, the engine is classified as
        **{st.session_state.prediction_health_status}** with an overall
        **{st.session_state.prediction_risk_level}**.

        Recommended action: **{st.session_state.prediction_recommendation}**
        """
    )

    st.subheader("Top Five Influential Features")

    st.dataframe(
        feature_importance.head(5),
        use_container_width=True,
    )
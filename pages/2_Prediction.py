import streamlit as st

from config import (
    APP_TITLE,
    PREDICTIVE_MODEL_NAME,
    PRIMARY_XAI_METHOD,
    SUPPLEMENTARY_XAI_METHOD,
)

from utils.layout import render_page, end_page
from utils.helpers import section_title

from utils.data_loader import (
    load_lstm_model,
    load_scaler,
    load_selected_features,
)

from utils.prediction import (
    predict_rul,
    generate_decision_output,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title=f"Prediction | {APP_TITLE}",
    layout="wide",
)

render_page(
    "AI Predictive Maintenance Decision Support",
    (
        "Predict Remaining Useful Life and translate the result "
        "into health status, risk level and maintenance guidance."
    ),
)


# ============================================================
# Load deployed model artefacts
# ============================================================

lstm_model = load_lstm_model()
scaler = load_scaler()
selected_features = list(load_selected_features())


# ============================================================
# Session-state and reset logic
# ============================================================

if "prediction_completed" not in st.session_state:
    st.session_state.prediction_completed = False

if "input_reset_counter" not in st.session_state:
    st.session_state.input_reset_counter = 0


def clear_prediction_page() -> None:
    """
    Reset the prediction output and create new widget keys so that
    all operational-setting and sensor inputs return to zero.
    """
    st.session_state.prediction_completed = False
    st.session_state.input_reset_counter += 1

    for key in [
        "prediction_rul",
        "prediction_health_status",
        "prediction_risk_level",
        "prediction_recommendation",
    ]:
        st.session_state.pop(key, None)


reset_id = st.session_state.input_reset_counter


# ============================================================
# 1. Predictive model overview
# ============================================================

section_title("1. Predictive Model Overview")

overview_col1, overview_col2, overview_col3 = st.columns(3)

overview_col1.metric(
    "Deployed Model",
    PREDICTIVE_MODEL_NAME,
)

overview_col2.metric(
    "Input Window",
    "30 operational cycles",
)

overview_col3.metric(
    "Input Variables",
    f"{len(selected_features)} features",
)

st.info(
    """
    The deployed Improved LSTM was trained using **30-cycle multivariate
    operational sequences** from the NASA C-MAPSS FD001 dataset.

    During training, the model learned degradation patterns from the
    historical behaviour of each engine rather than from a single
    operational cycle. This interface provides a simplified
    decision-support demonstration of the trained model.
    """
)


# ============================================================
# 2. Operational settings
# ============================================================

section_title("2. Operational Settings")

settings = {}

setting_names = [
    feature
    for feature in selected_features
    if feature.startswith("setting")
]

setting_columns = st.columns(
    max(1, len(setting_names))
)

for index, setting in enumerate(setting_names):
    with setting_columns[index]:
        settings[setting] = st.number_input(
            setting.replace("_", " ").title(),
            value=0.0,
            format="%.3f",
            key=f"input_{setting}_{reset_id}",
            help=(
                "Enter the current value for this operational "
                "setting."
            ),
        )


# ============================================================
# 3. Sensor measurements
# ============================================================

section_title("3. Sensor Measurements")

sensor_inputs = {}

sensor_features = [
    feature
    for feature in selected_features
    if feature.startswith("sensor")
]

left_column, right_column = st.columns(2)

split_index = (
    len(sensor_features) + 1
) // 2

for index, sensor in enumerate(sensor_features):
    target_column = (
        left_column
        if index < split_index
        else right_column
    )

    with target_column:
        sensor_inputs[sensor] = st.number_input(
            sensor.replace("_", " ").title(),
            value=0.0,
            format="%.3f",
            key=f"input_{sensor}_{reset_id}",
            help=(
                "Enter the current measurement for this "
                "selected sensor variable."
            ),
        )


# ============================================================
# 4. AI inference controls
# ============================================================

section_title("4. AI Inference")

predict_column, clear_column = st.columns(2)

with predict_column:
    predict_clicked = st.button(
        "Predict Remaining Useful Life",
        use_container_width=True,
        type="primary",
    )

with clear_column:
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
            ordered_values.append(
                settings[feature]
            )
        elif feature.startswith("sensor"):
            ordered_values.append(
                sensor_inputs[feature]
            )
        else:
            st.error(
                f"Unsupported input feature detected: {feature}"
            )
            st.stop()

    try:
        predicted_rul = predict_rul(
            input_values=ordered_values,
            lstm_model=lstm_model,
            scaler=scaler,
            selected_features=selected_features,
        )

        decision = generate_decision_output(
            predicted_rul
        )

    except (ValueError, TypeError) as error:
        st.error(
            "The prediction could not be completed because "
            f"the input data were invalid: {error}"
        )
        st.stop()

    except Exception as error:
        st.error(
            "An unexpected error occurred during model "
            f"inference: {error}"
        )
        st.stop()

    st.session_state.prediction_completed = True
    st.session_state.prediction_rul = float(
        decision["predicted_rul"]
    )
    st.session_state.prediction_health_status = (
        decision["health_status"]
    )
    st.session_state.prediction_risk_level = (
        decision["risk_level"]
    )
    st.session_state.prediction_recommendation = (
        decision["recommendation"]
    )


# ============================================================
# 5. Prediction and decision-support output
# ============================================================

if st.session_state.prediction_completed:
    st.divider()

    section_title(
        "5. Prediction and Health Assessment"
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    result_col1.metric(
        "Predicted RUL",
        (
            f"{st.session_state.prediction_rul:.2f} "
            "cycles"
        ),
    )

    result_col2.metric(
        "Health Status",
        st.session_state.prediction_health_status,
    )

    result_col3.metric(
        "Risk Level",
        st.session_state.prediction_risk_level,
    )


    # ========================================================
    # 6. Maintenance recommendation
    # ========================================================

    section_title(
        "6. Maintenance Recommendation"
    )

    recommendation = (
        st.session_state.prediction_recommendation
    )

    risk_level = (
        st.session_state.prediction_risk_level
    )

    if risk_level == "High Risk":
        st.error(recommendation)

    elif risk_level == "Medium Risk":
        st.warning(recommendation)

    else:
        st.success(recommendation)

    st.markdown(
        f"""
        The **{PREDICTIVE_MODEL_NAME}** estimates that the
        engine has approximately
        **{st.session_state.prediction_rul:.2f} operational
        cycles** remaining.

        This output corresponds to a
        **{st.session_state.prediction_health_status}**
        health classification and a
        **{st.session_state.prediction_risk_level}**
        risk level.

        The recommendation converts the numerical prediction
        into an operational decision-support statement. Final
        maintenance action should remain subject to engineering
        inspection, operational context and organisational
        maintenance procedures.
        """
    )


    # ========================================================
    # 7. Explainability reference
    # ========================================================

    section_title(
        "7. Explainability Reference"
    )

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.metric(
            "Primary XAI Method",
            PRIMARY_XAI_METHOD,
        )

        st.markdown(
            """
            TimeSHAP provides sequence-aware feature, event
            and temporal-coalition explanations directly for
            the Improved LSTM.
            """
        )

    with xai_col2:
        st.metric(
            "Supplementary XAI Method",
            SUPPLEMENTARY_XAI_METHOD,
        )

        st.markdown(
            """
            Integrated Gradients provides independent feature,
            timestep and sensor-time attribution used to
            validate the TimeSHAP findings.
            """
        )

    st.info(
        """
        The deployed application displays validated explanation
        artefacts generated during the Kaggle experimental phase.

        Open the **Explainability** module to review feature
        attribution, event attribution, coalition pruning,
        sensor-time heatmaps, deletion faithfulness and the
        critical degradation window.
        """
    )

else:
    st.divider()

    st.info(
        """
        Enter the operational settings and selected sensor
        measurements, then select **Predict Remaining Useful
        Life** to generate an RUL prediction, health assessment
        and maintenance recommendation.
        """
    )


end_page()
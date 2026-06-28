import numpy as np
import pandas as pd

from config import WINDOW_SIZE, RUL_CAP
from utils.health_status import classify_health_status, classify_risk_level
from utils.recommendations import recommend_action


def prepare_lstm_sequence(input_values, scaler, selected_features):
    input_df = pd.DataFrame([input_values], columns=selected_features)

    scaled_values = scaler.transform(input_df)

    sequence = np.repeat(
        scaled_values.reshape(1, 1, len(selected_features)),
        WINDOW_SIZE,
        axis=1
    )

    return sequence


def predict_rul(input_values, lstm_model, scaler, selected_features):
    sequence = prepare_lstm_sequence(
        input_values=input_values,
        scaler=scaler,
        selected_features=selected_features
    )

    prediction = lstm_model.predict(sequence, verbose=0).flatten()[0]
    prediction = float(np.clip(prediction, 0, RUL_CAP))

    return prediction


def generate_decision_output(predicted_rul):
    return {
        "predicted_rul": predicted_rul,
        "health_status": classify_health_status(predicted_rul),
        "risk_level": classify_risk_level(predicted_rul),
        "recommendation": recommend_action(predicted_rul)
    }
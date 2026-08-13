import numpy as np
import pandas as pd

from config import WINDOW_SIZE, RUL_CAP
from utils.health_status import (
    classify_health_status,
    classify_risk_level,
)
from utils.recommendations import recommend_action


def prepare_lstm_sequence(
    input_df: pd.DataFrame,
    scaler,
    selected_features,
):
    """
    Prepare a genuine multivariate time-series sequence for the
    deployed Improved LSTM model.

    Expected input:
    - exactly WINDOW_SIZE rows;
    - all selected model features present;
    - rows ordered from earliest to latest operational cycle.

    Output shape:
    (1, WINDOW_SIZE, number_of_features)
    """

    if input_df is None or input_df.empty:
        raise ValueError(
            "No engine sequence data were supplied."
        )

    missing_features = [
        feature
        for feature in selected_features
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )

    if len(input_df) != WINDOW_SIZE:
        raise ValueError(
            f"The Improved LSTM requires exactly "
            f"{WINDOW_SIZE} consecutive operational cycles. "
            f"The uploaded file contains {len(input_df)} rows."
        )

    sequence_df = input_df[
        selected_features
    ].copy()

    for feature in selected_features:
        sequence_df[feature] = pd.to_numeric(
            sequence_df[feature],
            errors="coerce",
        )

    if sequence_df.isnull().any().any():
        invalid_features = sequence_df.columns[
            sequence_df.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing or non-numeric values were detected in: "
            + ", ".join(invalid_features)
        )

    scaled_values = scaler.transform(
        sequence_df
    )

    sequence = scaled_values.reshape(
        1,
        WINDOW_SIZE,
        len(selected_features),
    )

    return sequence


def predict_rul(
    input_df: pd.DataFrame,
    lstm_model,
    scaler,
    selected_features,
):
    """
    Predict Remaining Useful Life from a genuine 30-cycle
    multivariate engine sequence.
    """

    sequence = prepare_lstm_sequence(
        input_df=input_df,
        scaler=scaler,
        selected_features=selected_features,
    )

    prediction = (
        lstm_model
        .predict(sequence, verbose=0)
        .flatten()[0]
    )

    prediction = float(
        np.clip(
            prediction,
            0,
            RUL_CAP,
        )
    )

    return prediction


def generate_decision_output(predicted_rul):
    """
    Translate numerical RUL into health status,
    risk level and maintenance recommendation.
    """

    return {
        "predicted_rul": predicted_rul,
        "health_status": classify_health_status(
            predicted_rul
        ),
        "risk_level": classify_risk_level(
            predicted_rul
        ),
        "recommendation": recommend_action(
            predicted_rul
        ),
    }
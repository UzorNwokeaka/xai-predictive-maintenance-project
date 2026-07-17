import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from config import (
    DASHBOARD_DATA_PATH,
    MODEL_COMPARISON_PATH,
    FEATURE_IMPORTANCE_PATH,
    TEMPORAL_EXPLAINABILITY_PATH,
    MODEL_METADATA_PATH,
    TIMESHAP_FEATURE_PATH,
    TIMESHAP_EVENT_PATH,
    TIMESHAP_PRUNING_PATH,
    IG_FEATURE_PATH,
    IG_EVENT_PATH,
    IG_SENSOR_TIME_PATH,
    XAI_FIDELITY_PATH,
    XAI_METHOD_SCORECARD_PATH,
    XAI_COMPARISON_SUMMARY_PATH,
    LSTM_MODEL_PATH,
    RF_MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
)


def _require_file(path: Path) -> Path:
    """
    Validating that an expected artefact exists before attempting to load it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required application artefact was not found: {path}"
        )
    return path


# ============================================================
# Existing application datasets
# ============================================================

@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    return pd.read_csv(_require_file(DASHBOARD_DATA_PATH))


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(_require_file(MODEL_COMPARISON_PATH))


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(_require_file(FEATURE_IMPORTANCE_PATH))


@st.cache_data
def load_temporal_explainability() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(TEMPORAL_EXPLAINABILITY_PATH)
    )


@st.cache_data
def load_model_metadata() -> dict:
    with open(
        _require_file(MODEL_METADATA_PATH),
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Direct LSTM explainability datasets
# ============================================================

@st.cache_data
def load_timeshap_feature_attribution() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(TIMESHAP_FEATURE_PATH)
    )


@st.cache_data
def load_timeshap_event_attribution() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(TIMESHAP_EVENT_PATH)
    )


@st.cache_data
def load_timeshap_pruning() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(TIMESHAP_PRUNING_PATH)
    )


@st.cache_data
def load_ig_feature_attribution() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(IG_FEATURE_PATH)
    )


@st.cache_data
def load_ig_event_attribution() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(IG_EVENT_PATH)
    )


@st.cache_data
def load_ig_sensor_time_attribution() -> np.ndarray:
    return np.load(
        _require_file(IG_SENSOR_TIME_PATH),
        allow_pickle=False,
    )


@st.cache_data
def load_xai_feature_deletion_fidelity() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(XAI_FIDELITY_PATH)
    )


@st.cache_data
def load_xai_method_scorecard() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(XAI_METHOD_SCORECARD_PATH)
    )


@st.cache_data
def load_xai_comparison_summary() -> dict:
    with open(
        _require_file(XAI_COMPARISON_SUMMARY_PATH),
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Model and preprocessing artefacts
# ============================================================

@st.cache_resource
def load_lstm_model():
    return load_model(
        _require_file(LSTM_MODEL_PATH)
    )


@st.cache_resource
def load_random_forest_model():
    return joblib.load(
        _require_file(RF_MODEL_PATH)
    )


@st.cache_resource
def load_scaler():
    return joblib.load(
        _require_file(SCALER_PATH)
    )


@st.cache_resource
def load_selected_features():
    return joblib.load(
        _require_file(FEATURES_PATH)
    )
import joblib
import pandas as pd
import json
import streamlit as st
from tensorflow.keras.models import load_model

from config import (
    DASHBOARD_DATA_PATH,
    MODEL_COMPARISON_PATH,
    FEATURE_IMPORTANCE_PATH,
    LSTM_MODEL_PATH,
    RF_MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    TEMPORAL_EXPLAINABILITY_PATH,
    MODEL_METADATA_PATH,
)


@st.cache_data
def load_dashboard_data():
    return pd.read_csv(DASHBOARD_DATA_PATH)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_resource
def load_lstm_model():
    return load_model(LSTM_MODEL_PATH)


@st.cache_resource
def load_random_forest_model():
    return joblib.load(RF_MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_selected_features():
    return joblib.load(FEATURES_PATH)

@st.cache_data
def load_temporal_explainability():
    return pd.read_csv(TEMPORAL_EXPLAINABILITY_PATH)


@st.cache_data
def load_model_metadata():
    with open(MODEL_METADATA_PATH, "r") as file:
        return json.load(file)
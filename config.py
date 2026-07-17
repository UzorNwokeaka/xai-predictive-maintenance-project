from pathlib import Path


# ============================================================
# Project directories
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# Existing application datasets
# ============================================================

DASHBOARD_DATA_PATH = DATA_DIR / "dashboard_predictions.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "final_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "global_feature_importance.csv"
TEMPORAL_EXPLAINABILITY_PATH = DATA_DIR / "temporal_explainability.csv"
MODEL_METADATA_PATH = DATA_DIR / "model_metadata.json"


# ============================================================
# Direct LSTM explainability artefacts
# ============================================================

TIMESHAP_FEATURE_PATH = (
    DATA_DIR / "lstm_timeshap_feature_attribution.csv"
)

TIMESHAP_EVENT_PATH = (
    DATA_DIR / "lstm_timeshap_event_attribution.csv"
)

TIMESHAP_PRUNING_PATH = (
    DATA_DIR / "lstm_timeshap_pruning.csv"
)

IG_FEATURE_PATH = (
    DATA_DIR / "lstm_ig_feature_attribution.csv"
)

IG_EVENT_PATH = (
    DATA_DIR / "lstm_ig_event_attribution.csv"
)

IG_SENSOR_TIME_PATH = (
    DATA_DIR / "lstm_ig_sensor_time_attribution.npy"
)

XAI_FIDELITY_PATH = (
    DATA_DIR / "xai_feature_deletion_fidelity.csv"
)

XAI_METHOD_SCORECARD_PATH = (
    DATA_DIR / "xai_method_scorecard.csv"
)

XAI_COMPARISON_SUMMARY_PATH = (
    DATA_DIR / "xai_method_comparison_summary.json"
)


# ============================================================
# Model and preprocessing artefacts
# ============================================================

LSTM_MODEL_PATH = (
    MODEL_DIR / "final_best_predictive_model_lstm.keras"
)

RF_MODEL_PATH = (
    MODEL_DIR / "final_explainable_reference_model_rf.pkl"
)

SCALER_PATH = (
    MODEL_DIR / "fd001_minmax_scaler.pkl"
)

FEATURES_PATH = (
    MODEL_DIR / "selected_features.pkl"
)


# ============================================================
# Application configuration
# ============================================================

APP_TITLE = (
    "Explainable AI Predictive Maintenance "
    "Decision Support System"
)

RUL_CAP = 125
WINDOW_SIZE = 30

PRIMARY_XAI_METHOD = "TimeSHAP"
SUPPLEMENTARY_XAI_METHOD = "Integrated Gradients"
PREDICTIVE_MODEL_NAME = "Improved LSTM"
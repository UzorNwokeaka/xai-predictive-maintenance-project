from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

DASHBOARD_DATA_PATH = DATA_DIR / "dashboard_predictions.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "final_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "global_feature_importance.csv"
TEMPORAL_EXPLAINABILITY_PATH = DATA_DIR / "temporal_explainability.csv"
MODEL_METADATA_PATH = DATA_DIR / "model_metadata.json"

LSTM_MODEL_PATH = MODEL_DIR / "final_best_predictive_model_lstm.keras"
RF_MODEL_PATH = MODEL_DIR / "final_explainable_reference_model_rf.pkl"
SCALER_PATH = MODEL_DIR / "fd001_minmax_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "selected_features.pkl"

APP_TITLE = "Explainable AI Predictive Maintenance Decision Support System"

RUL_CAP = 125
WINDOW_SIZE = 30
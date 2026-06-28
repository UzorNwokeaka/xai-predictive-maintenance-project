from pathlib import Path
import pandas as pd
import json

from config import (
    DATA_DIR,
    MODEL_DIR,
    DASHBOARD_DATA_PATH,
    MODEL_COMPARISON_PATH,
    FEATURE_IMPORTANCE_PATH,
    TEMPORAL_EXPLAINABILITY_PATH,
    MODEL_METADATA_PATH,
    LSTM_MODEL_PATH,
    RF_MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
)


def check_file_exists(path: Path) -> dict:
    return {
        "item": path.name,
        "path": str(path),
        "exists": path.exists(),
        "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() else 0,
    }


def run_file_dependency_audit() -> pd.DataFrame:
    required_files = [
        DASHBOARD_DATA_PATH,
        MODEL_COMPARISON_PATH,
        FEATURE_IMPORTANCE_PATH,
        TEMPORAL_EXPLAINABILITY_PATH,
        MODEL_METADATA_PATH,
        LSTM_MODEL_PATH,
        RF_MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH,
    ]

    return pd.DataFrame([check_file_exists(path) for path in required_files])


def run_dataset_audit() -> dict:
    dashboard_df = pd.read_csv(DASHBOARD_DATA_PATH)
    model_df = pd.read_csv(MODEL_COMPARISON_PATH)
    feature_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    temporal_df = pd.read_csv(TEMPORAL_EXPLAINABILITY_PATH)

    with open(MODEL_METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return {
        "dashboard_rows": len(dashboard_df),
        "dashboard_columns": dashboard_df.columns.tolist(),
        "model_rows": len(model_df),
        "model_columns": model_df.columns.tolist(),
        "feature_rows": len(feature_df),
        "feature_columns": feature_df.columns.tolist(),
        "temporal_rows": len(temporal_df),
        "temporal_columns": temporal_df.columns.tolist(),
        "metadata_keys": list(metadata.keys()),
        "best_model_from_metadata": metadata.get("best_model"),
        "rmse_from_metadata": metadata.get("rmse"),
    }


def run_quality_checks() -> pd.DataFrame:
    checks = []

    dashboard_df = pd.read_csv(DASHBOARD_DATA_PATH)
    model_df = pd.read_csv(MODEL_COMPARISON_PATH)
    feature_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    temporal_df = pd.read_csv(TEMPORAL_EXPLAINABILITY_PATH)

    checks.append({
        "check": "Dashboard data has 100 engines",
        "status": dashboard_df["engine_id"].nunique() == 100,
    })

    checks.append({
        "check": "Dashboard data contains predicted LSTM RUL",
        "status": "predicted_rul_lstm" in dashboard_df.columns,
    })

    checks.append({
        "check": "Dashboard data contains health status",
        "status": "health_status" in dashboard_df.columns,
    })

    checks.append({
        "check": "Model comparison contains RMSE",
        "status": "RMSE" in model_df.columns,
    })

    checks.append({
        "check": "Improved LSTM exists in model comparison",
        "status": "Improved LSTM" in model_df["Model"].values,
    })

    checks.append({
        "check": "Feature importance file is not empty",
        "status": len(feature_df) > 0,
    })

    checks.append({
        "check": "Temporal explainability file is not empty",
        "status": len(temporal_df) > 0,
    })

    checks.append({
        "check": "Temporal file contains engine_id and cycle",
        "status": all(col in temporal_df.columns for col in ["engine_id", "cycle"]),
    })

    return pd.DataFrame(checks)
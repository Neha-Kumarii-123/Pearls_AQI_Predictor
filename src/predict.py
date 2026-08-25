"""
Production inference pipeline for the Karachi AQI Predictor.

Production flow:

    Live feature pipeline
        ↓
    Latest complete 100-feature row
        ↓
    Hopsworks Model Registry
        ↓
    Day +1 XGBoost
    Day +2 Ridge
    Day +3 Ridge
        ↓
    Three AQI predictions

Important:
    This file is an orchestration layer only.

    Feature engineering remains inside feature_pipeline.py
    and feature_engineering.py.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import hopsworks
import joblib
import pandas as pd
from dotenv import load_dotenv

from feature_engineering import MODEL_FEATURES
from feature_pipeline import AirQualityFeaturePipeline


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    dotenv_path=REPO_ROOT / ".env",
    override=False,
)


# ---------------------------------------------------------------------
# Hopsworks configuration
# ---------------------------------------------------------------------

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

MODEL_NAMES = {
    "day1": "karachi_aqi_day1_xgboost",
    "day2": "karachi_aqi_day2_ridge",
    "day3": "karachi_aqi_day3_ridge",
}


# ---------------------------------------------------------------------
# Hopsworks authentication
# ---------------------------------------------------------------------

def connect_to_hopsworks():
    """
    Authenticate with Hopsworks and return the project.
    """

    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is not set."
        )

    print("--- Connecting to Hopsworks ---")

    project = hopsworks.login(
        api_key_value=api_key,
        host=HOPSWORKS_HOST,
        cert_folder=tempfile.gettempdir(),
    )

    return project


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_registered_model(
    registry,
    model_name: str,
):
    """
    Load the latest version of a registered model.

    Returns:
        model,
        metadata
    """

    print(
        f"\n--- Loading registered model: "
        f"{model_name} ---"
    )

    models = registry.get_models(model_name)

    if not models:
        raise RuntimeError(
            f"No registered versions found for {model_name}"
        )

    latest_model = max(
        models,
        key=lambda model: model.version,
    )

    print(
        f"Using {model_name} "
        f"version {latest_model.version}"
    )

    model_directory = latest_model.download()

    model_directory = Path(model_directory)

    metadata_path = (
        model_directory / "model_metadata.pkl"
    )

    if not metadata_path.exists():
        raise RuntimeError(
            f"model_metadata.pkl not found for "
            f"{model_name}: {metadata_path}"
        )

    metadata = joblib.load(
        metadata_path
    )

    artifact_path = (
        model_directory
        / f"{metadata['model_name']}.pkl"
    )

    if not artifact_path.exists():
        # Fallback: locate the PKL artifact in the bundle.
        candidates = list(
            model_directory.glob("*.pkl")
        )

        candidates = [
            path
            for path in candidates
            if path.name != "model_metadata.pkl"
        ]

        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not uniquely identify model "
                f"artifact for {model_name}. "
                f"Found: {candidates}"
            )

        artifact_path = candidates[0]

    model = joblib.load(
        artifact_path
    )

    return model, metadata, latest_model.version


# ---------------------------------------------------------------------
# Validate model metadata
# ---------------------------------------------------------------------

def validate_model_metadata(
    metadata: dict,
    expected_model_name: str,
):
    """
    Validate the registered model's production contract.
    """

    if metadata.get("model_name") != expected_model_name:
        raise RuntimeError(
            "Registered model metadata mismatch: "
            f"expected {expected_model_name}, "
            f"found {metadata.get('model_name')}"
        )

    canonical_count = metadata.get(
        "canonical_feature_count"
    )

    if canonical_count != 100:
        raise RuntimeError(
            "Expected 100 canonical features, "
            f"found {canonical_count}"
        )

    canonical_features = metadata.get(
        "canonical_model_features"
    )

    if canonical_features != list(MODEL_FEATURES):
        raise RuntimeError(
            "Registered model canonical feature "
            "contract does not match the current "
            "MODEL_FEATURES definition."
        )


# ---------------------------------------------------------------------
# Prepare model input
# ---------------------------------------------------------------------

def prepare_model_input(
    feature_row: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """
    Prepare the exact feature frame expected by a
    registered model.

    Day +1:
        100 canonical features.

    Day +2 / Day +3:
        60 selected features stored in model metadata.
    """

    canonical_features = list(
        MODEL_FEATURES
    )

    selected_features = metadata.get(
        "selected_features"
    )

    if selected_features:
        feature_columns = selected_features
    else:
        feature_columns = canonical_features

    missing = [
        column
        for column in feature_columns
        if column not in feature_row.columns
    ]

    if missing:
        raise RuntimeError(
            "Production feature row is missing "
            f"required model features: {missing}"
        )

    X = feature_row[
        feature_columns
    ].copy()

    if X.isna().any().any():
        raise RuntimeError(
            "Model input contains NaN values."
        )

    return X


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict():
    """
    Execute the complete production inference flow.
    """

    print(
        "\n=============================================="
    )
    print(
        " KARACHI AQI PRODUCTION INFERENCE"
    )
    print(
        "=============================================="
    )

    # -------------------------------------------------------------
    # 1. Run the finalized live feature pipeline
    # -------------------------------------------------------------

    print(
        "\n--- Running production feature pipeline ---"
    )

    feature_pipeline = (
        AirQualityFeaturePipeline(
            city="karachi"
        )
    )

    feature_row = feature_pipeline.run(
        write_to_feature_store=False
    )

    print(
        "\nLatest production timestamp:",
        feature_row["timestamp"].iloc[0],
    )

    print(
        "Production feature count:",
        len(MODEL_FEATURES),
    )

    # -------------------------------------------------------------
    # 2. Connect to Model Registry
    # -------------------------------------------------------------

    project = connect_to_hopsworks()

    registry = (
        project.get_model_registry()
    )

    # -------------------------------------------------------------
    # 3. Load Day +1 model
    # -------------------------------------------------------------

    day1_model, day1_metadata, day1_version = (
        load_registered_model(
            registry,
            MODEL_NAMES["day1"],
        )
    )

    validate_model_metadata(
        day1_metadata,
        MODEL_NAMES["day1"],
    )

    X_day1 = prepare_model_input(
        feature_row,
        day1_metadata,
    )

    day1_prediction = float(
        day1_model.predict(X_day1)[0]
    )

    # -------------------------------------------------------------
    # 4. Load Day +2 model
    # -------------------------------------------------------------

    day2_model, day2_metadata, day2_version = (
        load_registered_model(
            registry,
            MODEL_NAMES["day2"],
        )
    )

    validate_model_metadata(
        day2_metadata,
        MODEL_NAMES["day2"],
    )

    X_day2 = prepare_model_input(
        feature_row,
        day2_metadata,
    )

    day2_prediction = float(
        day2_model.predict(X_day2)[0]
    )

    # -------------------------------------------------------------
    # 5. Load Day +3 model
    # -------------------------------------------------------------

    day3_model, day3_metadata, day3_version = (
        load_registered_model(
            registry,
            MODEL_NAMES["day3"],
        )
    )

    validate_model_metadata(
        day3_metadata,
        MODEL_NAMES["day3"],
    )

    X_day3 = prepare_model_input(
        feature_row,
        day3_metadata,
    )

    day3_prediction = float(
        day3_model.predict(X_day3)[0]
    )

    # -------------------------------------------------------------
    # 6. Results
    # -------------------------------------------------------------

    print(
        "\n=============================================="
    )
    print(
        " AQI PREDICTIONS"
    )
    print(
        "=============================================="
    )

    print(
        f"Day +1 ({MODEL_NAMES['day1']} "
        f"v{day1_version}): "
        f"{day1_prediction:.2f}"
    )

    print(
        f"Day +2 ({MODEL_NAMES['day2']} "
        f"v{day2_version}): "
        f"{day2_prediction:.2f}"
    )

    print(
        f"Day +3 ({MODEL_NAMES['day3']} "
        f"v{day3_version}): "
        f"{day3_prediction:.2f}"
    )

    print(
        "\n=============================================="
    )
    print(
        " PRODUCTION INFERENCE SUCCESS"
    )
    print(
        "=============================================="
    )

    return {
        "timestamp": feature_row[
            "timestamp"
        ].iloc[0],
        "day1": day1_prediction,
        "day2": day2_prediction,
        "day3": day3_prediction,
        "day1_model_version": day1_version,
        "day2_model_version": day2_version,
        "day3_model_version": day3_version,
    }


# ---------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    predict()
"""
Production inference pipeline for the Karachi AQI Predictor.

Production flow:

    Hopsworks Feature Group v6
        ↓
    Latest complete feature row
        ↓
    Hopsworks Model Registry
        ↓
    Day +1 XGBoost
    Day +2 Ridge
    Day +3 Ridge
        ↓
    Three AQI predictions

Important
---------
1. Day +1, Day +2, and Day +3 models are already trained and locked.
2. This file does NOT retrain, modify, or register any model.
3. Feature engineering and feature generation are handled by the
   production feature pipeline, not by this inference script.
4. v6 is the canonical production Feature Group.
5. This file only reads the latest available Karachi feature row
   from v6 for inference.
6. The latest feature row must contain all 100 canonical MODEL_FEATURES.
7. The exact model feature contract stored in model metadata is
   validated before prediction.
8. Day +1 uses the canonical feature set, while Day +2 and Day +3
   use their registered selected feature sets.
9. This script loads the latest registered model version for each
   forecasting horizon and generates three AQI predictions.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import hopsworks
import joblib
import pandas as pd
from dotenv import load_dotenv

from src.feature_engineering import MODEL_FEATURES
from src.feature_pipeline import FeaturePipelineError


# =====================================================================
# ENVIRONMENT
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    dotenv_path=REPO_ROOT / ".env",
    override=False,
)


# =====================================================================
# HOPSWORKS CONFIGURATION
# =====================================================================

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 6

MODEL_NAMES = {
    "day1": "karachi_aqi_day1_xgboost",
    "day2": "karachi_aqi_day2_ridge",
    "day3": "karachi_aqi_day3_ridge",
}

# ---------------------------------------------------------------------
# APPLICATION-LEVEL CACHES
# ---------------------------------------------------------------------
# Keep the heavy Hopsworks and model initialization work in memory for the
# lifetime of the FastAPI process. The feature row itself remains fresh per
# request and is never cached as a final prediction.
_HOPSWORKS_PROJECT = None
_MODEL_CACHE = {}

# Guards Hopsworks login so the background cache warm-up thread and an
# incoming request can't both try to authenticate at the same time.
_HOPSWORKS_CONNECT_LOCK = threading.Lock()


# =====================================================================
# HOPSWORKS AUTHENTICATION
# =====================================================================

def connect_to_hopsworks():
    """
    Authenticate with Hopsworks and return the project.

    The project connection is reused across requests in the same FastAPI
    process instead of re-authenticating on every /predict or /current call.

    A lock guards the actual login call so that if two callers (e.g. the
    startup warm-up thread and an incoming request) reach this function at
    nearly the same time, only one of them performs the Hopsworks login.
    """

    global _HOPSWORKS_PROJECT

    if _HOPSWORKS_PROJECT is not None:
        print("\n--- Reusing cached Hopsworks project ---")
        return _HOPSWORKS_PROJECT

    with _HOPSWORKS_CONNECT_LOCK:

        # Re-check inside the lock: another thread may have already
        # connected while this one was waiting.
        if _HOPSWORKS_PROJECT is not None:
            print("\n--- Reusing cached Hopsworks project ---")
            return _HOPSWORKS_PROJECT

        api_key = os.getenv("HOPSWORKS_API_KEY")

        if not api_key:
            raise RuntimeError(
                "HOPSWORKS_API_KEY is not set."
            )

        print("\n--- Connecting to Hopsworks ---")

        _HOPSWORKS_PROJECT = hopsworks.login(
            api_key_value=api_key,
            host=HOPSWORKS_HOST,
            cert_folder=tempfile.gettempdir(),
        )

        print("--- Hopsworks connection successful ---")

    return _HOPSWORKS_PROJECT


# =====================================================================
# LOAD LATEST V6 FEATURE ROW
# =====================================================================

def get_latest_v6_row(
    project,
) -> pd.DataFrame:
    """
    Read the latest complete production feature row from v6.

    This function is used when the feature pipeline reports that
    no new hourly observations are available.

    Returns
    -------
    pd.DataFrame
        Exactly one latest v6 row.
    """

    print(
        "\n--- Reading latest existing row from "
        "Hopsworks v6 ---"
    )

    fs = project.get_feature_store()

    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    # -------------------------------------------------------------
    # Read a sufficiently recent window.
    #
    # The production feature pipeline already guarantees that
    # v6 contains the required historical context.
    #
    # 200 hours gives us enough room to find the latest row.
    # -------------------------------------------------------------

    now = pd.Timestamp.now(
        tz="UTC"
    )

    start_time = (
        now
        - pd.Timedelta(hours=200)
    )

    try:

        dataframe = feature_group.read(
            start_time=start_time.to_pydatetime(),
            end_time=now.to_pydatetime(),
            dataframe_type="pandas",
            online=True
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to read latest v6 feature row: "
            f"{exc}"
        ) from exc

    if dataframe is None or dataframe.empty:

        raise RuntimeError(
            "Hopsworks v6 contains no recent feature rows."
        )

    if "timestamp" not in dataframe.columns:

        raise RuntimeError(
            "v6 data does not contain a timestamp column."
        )

    if "city" not in dataframe.columns:

        raise RuntimeError(
            "v6 data does not contain a city column."
        )

    # -------------------------------------------------------------
    # Normalize timestamp.
    # -------------------------------------------------------------

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["timestamp"]
    )

    # -------------------------------------------------------------
    # Karachi only.
    # -------------------------------------------------------------

    dataframe = dataframe[
        dataframe["city"].astype(str).str.lower()
        == "karachi"
    ].copy()

    if dataframe.empty:

        raise RuntimeError(
            "No Karachi rows found in v6."
        )

    # -------------------------------------------------------------
    # Sort and select the latest row.
    # -------------------------------------------------------------

    dataframe = (
        dataframe
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    latest_row = dataframe.iloc[
        [-1]
    ].copy()

    print(
        "Latest existing v6 timestamp:",
        latest_row["timestamp"].iloc[0],
    )

    return latest_row


# =====================================================================
# LOAD REGISTERED MODEL
# =====================================================================


def load_registered_model(
    registry,
    model_name: str,
):
    """
    Load the latest version of a registered model.

    Reuse the cached model object when the registry version has not changed.
    This avoids the expensive download + joblib.load sequence on every
    prediction request while still respecting the active Hopsworks registry.

    Returns
    -------
    model
    metadata
    version
    """

    global _MODEL_CACHE

    models = registry.get_models(
        model_name
    )

    if not models:
        raise RuntimeError(
            f"No registered versions found for "
            f"{model_name}"
        )

    latest_model = max(
        models,
        key=lambda model: model.version,
    )

    cached_entry = _MODEL_CACHE.get(model_name)
    if cached_entry is not None:
        cached_version = cached_entry.get("version")
        if cached_version == latest_model.version:
            print(
                f"Using cached {model_name} "
                f"version {cached_version}"
            )
            return (
                cached_entry["model"],
                cached_entry["metadata"],
                cached_version,
            )

    print(
        f"\n--- Loading registered model: "
        f"{model_name} ---"
    )

    print(
        f"Using {model_name} "
        f"version {latest_model.version}"
    )

    model_directory = latest_model.download()

    model_directory = Path(
        model_directory
    )

    print(
        "Downloaded model directory:",
        model_directory,
    )

    metadata_path = (
        model_directory
        / "model_metadata.pkl"
    )

    if not metadata_path.exists():

        raise RuntimeError(
            f"model_metadata.pkl not found for "
            f"{model_name}: {metadata_path}"
        )

    metadata = joblib.load(
        metadata_path
    )

    if not isinstance(metadata, dict):

        raise RuntimeError(
            f"Invalid model metadata for "
            f"{model_name}. Expected dict, "
            f"found {type(metadata).__name__}."
        )

    expected_model_name = metadata.get(
        "model_name"
    )

    if not expected_model_name:

        raise RuntimeError(
            f"model_metadata.pkl for {model_name} "
            "does not contain 'model_name'."
        )

    artifact_path = (
        model_directory
        / f"{expected_model_name}.pkl"
    )

    if artifact_path.exists():

        print(
            "Selected model artifact:",
            artifact_path.name,
        )

    else:

        candidates = [
            path
            for path in model_directory.glob("*.pkl")
            if path.name != "model_metadata.pkl"
        ]

        model_candidates = [
            path
            for path in candidates
            if model_name.lower() in path.stem.lower()
            and "feature" not in path.stem.lower()
            and "features" not in path.stem.lower()
            and "metadata" not in path.stem.lower()
        ]

        if len(model_candidates) == 1:

            artifact_path = model_candidates[0]

            print(
                "Selected model artifact:",
                artifact_path.name,
            )

        elif len(model_candidates) > 1:

            raise RuntimeError(
                f"Multiple possible trained model artifacts "
                f"found for {model_name}: "
                f"{model_candidates}"
            )

        else:

            raise RuntimeError(
                f"Could not identify trained model artifact "
                f"for {model_name}.\n"
                f"Expected: {expected_model_name}.pkl\n"
                f"Available artifacts: {candidates}"
            )

    try:

        model = joblib.load(
            artifact_path
        )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to load trained model artifact "
            f"{artifact_path}: {exc}"
        ) from exc

    if not hasattr(model, "predict"):

        raise RuntimeError(
            f"Selected artifact does not appear to be a "
            f"trained prediction model: "
            f"{artifact_path.name}"
        )

    print(
        "Trained model loaded successfully."
    )

    _MODEL_CACHE[model_name] = {
        "model": model,
        "metadata": metadata,
        "version": latest_model.version,
    }

    return (
        model,
        metadata,
        latest_model.version,
    )


# =====================================================================
# VALIDATE MODEL METADATA
# =====================================================================

def validate_model_metadata(
    metadata: dict,
    expected_model_name: str,
):
    """
    Validate the registered model's production contract.
    """

    # -------------------------------------------------------------
    # Model name.
    # -------------------------------------------------------------

    registered_model_name = metadata.get(
        "model_name"
    )

    if registered_model_name != expected_model_name:

        raise RuntimeError(
            "Registered model metadata mismatch: "
            f"expected {expected_model_name}, "
            f"found {registered_model_name}"
        )

    # -------------------------------------------------------------
    # Canonical feature count.
    # -------------------------------------------------------------

    canonical_count = metadata.get(
        "canonical_feature_count"
    )

    if canonical_count != 100:

        raise RuntimeError(
            "Expected 100 canonical features, "
            f"found {canonical_count}"
        )

    # -------------------------------------------------------------
    # Canonical feature contract.
    # -------------------------------------------------------------

    canonical_features = metadata.get(
        "canonical_model_features"
    )

    if canonical_features is None:

        raise RuntimeError(
            "Registered model metadata does not "
            "contain canonical_model_features."
        )

    if list(canonical_features) != list(
        MODEL_FEATURES
    ):

        raise RuntimeError(
            "Registered model canonical feature "
            "contract does not match the current "
            "MODEL_FEATURES definition."
        )


# =====================================================================
# PREPARE MODEL INPUT
# =====================================================================

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

    # -------------------------------------------------------------
    # Use selected features when the model stores them.
    #
    # Day +2 and Day +3 use their final selected 60 features.
    # Day +1 uses the canonical 100 features.
    # -------------------------------------------------------------

    selected_features = metadata.get(
        "selected_features"
    )

    if selected_features:

        feature_columns = list(
            selected_features
        )

    else:

        feature_columns = canonical_features

    # -------------------------------------------------------------
    # Verify every required feature exists.
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Preserve exact feature order.
    # -------------------------------------------------------------

    X = feature_row[
        feature_columns
    ].copy()

    # -------------------------------------------------------------
    # NaN protection.
    # -------------------------------------------------------------

    if X.isna().any().any():

        raise RuntimeError(
            "Model input contains NaN values."
        )

    # -------------------------------------------------------------
    # Exactly one prediction row.
    # -------------------------------------------------------------

    if len(X) != 1:

        raise RuntimeError(
            "Inference expects exactly one feature row, "
            f"received {len(X)}."
        )

    return X



# =====================================================================
# MAIN PREDICTION FUNCTION
# =====================================================================

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
    # 1. Read latest production feature row from v6.
    # -------------------------------------------------------------

    project = connect_to_hopsworks()

    feature_row = get_latest_v6_row(
        project
    )
    # -------------------------------------------------------------
    # 3. Final feature-row validation.
    # -------------------------------------------------------------

    if feature_row is None or feature_row.empty:

        raise FeaturePipelineError(
            "No production feature row is available "
            "for inference."
        )

    if len(feature_row) != 1:

        raise FeaturePipelineError(
            "Inference requires exactly one latest "
            f"feature row, received {len(feature_row)}."
        )

    # -------------------------------------------------------------
    # Ensure all canonical features exist.
    # -------------------------------------------------------------

    missing_canonical = [
        column
        for column in MODEL_FEATURES
        if column not in feature_row.columns
    ]

    if missing_canonical:

        raise FeaturePipelineError(
            "Latest production row is missing canonical "
            f"MODEL_FEATURES: {missing_canonical}"
        )

    # -------------------------------------------------------------
    # Ensure no NaN in canonical features.
    # -------------------------------------------------------------

    if feature_row[
        list(MODEL_FEATURES)
    ].isna().any().any():

        raise FeaturePipelineError(
            "Latest production feature row contains "
            "NaN values in MODEL_FEATURES."
        )

    latest_timestamp = (
        feature_row["timestamp"].iloc[0]
        
    )
    print("Latest actual AQI:", feature_row["target_aqi"].iloc[0])

    print(
        "\nLatest production timestamp:",
        latest_timestamp,
    )

    print(
        "Production feature count:",
        len(MODEL_FEATURES),
    )

    print(
        "Production feature validation: PASSED"
    )

    # -------------------------------------------------------------
    # 4. Connect to Model Registry.
    # -------------------------------------------------------------



    registry = (
        project.get_model_registry()
    )

    # -------------------------------------------------------------
    # 5. Load Day +1 model.
    # -------------------------------------------------------------

    (
        day1_model,
        day1_metadata,
        day1_version,
    ) = load_registered_model(
        registry,
        MODEL_NAMES["day1"],
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
        day1_model.predict(
            X_day1
        )[0]
    )

    # -------------------------------------------------------------
    # 6. Load Day +2 model.
    # -------------------------------------------------------------

    (
        day2_model,
        day2_metadata,
        day2_version,
    ) = load_registered_model(
        registry,
        MODEL_NAMES["day2"],
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
        day2_model.predict(
            X_day2
        )[0]
    )

    # -------------------------------------------------------------
    # 7. Load Day +3 model.
    # -------------------------------------------------------------

    (
        day3_model,
        day3_metadata,
        day3_version,
    ) = load_registered_model(
        registry,
        MODEL_NAMES["day3"],
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
        day3_model.predict(
            X_day3
        )[0]
    )

    # -------------------------------------------------------------
    # 8. Results.
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
        f"Prediction base timestamp: "
        f"{latest_timestamp}"
    )

    print(
        f"Day +1 "
        f"({MODEL_NAMES['day1']} "
        f"v{day1_version}): "
        f"{day1_prediction:.2f}"
    )

    print(
        f"Day +2 "
        f"({MODEL_NAMES['day2']} "
        f"v{day2_version}): "
        f"{day2_prediction:.2f}"
    )

    print(
        f"Day +3 "
        f"({MODEL_NAMES['day3']} "
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
        "timestamp": latest_timestamp,
        "current_aqi": float(
        feature_row["target_aqi"].iloc[0]
        ),
        "day1": day1_prediction,
        "day2": day2_prediction,
        "day3": day3_prediction,
        "day1_model_version": day1_version,
        "day2_model_version": day2_version,
        "day3_model_version": day3_version,
    }


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================

if __name__ == "__main__":

    try:

        result = predict()

        print(
            "\nReturned prediction result:"
        )

        print(result)

    except Exception as exc:

        print(
            "\n=============================================="
        )

        print(
            " PRODUCTION INFERENCE FAILED"
        )

        print(
            "=============================================="
        )

        print(
            f"\nError: {exc}"
        )

        raise
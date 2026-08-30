"""
Automated production prediction pipeline.

Flow:
    Hopsworks v6
        ↓
    Production inference
        ↓
    Day +1 / Day +2 / Day +3
        ↓
    Hopsworks Prediction Feature Group

This script does NOT train or modify models.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from src.predict import predict


# =====================================================================
# CONFIGURATION
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parent

load_dotenv(
    dotenv_path=REPO_ROOT / ".env",
    override=False,
)

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

PREDICTION_FG_NAME = "karachi_aqi_predictions"
PREDICTION_FG_VERSION = 1


# =====================================================================
# HOPSWORKS CONNECTION
# =====================================================================

def connect_to_hopsworks():
    """Connect to Hopsworks."""

    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is not set."
        )

    print("\n--- Connecting to Hopsworks ---")

    project = hopsworks.login(
        api_key_value=api_key,
        host=HOPSWORKS_HOST,
        cert_folder=tempfile.gettempdir(),
    )

    print("--- Hopsworks connection successful ---")

    return project


# =====================================================================
# CREATE / GET PREDICTION FEATURE GROUP
# =====================================================================

def get_prediction_feature_group(project):
    """
    Get the prediction Feature Group.

    If it does not exist, create it once.
    """

    fs = project.get_feature_store()

    # -------------------------------------------------------------
    # Try to get existing Feature Group.
    # -------------------------------------------------------------

    try:

        prediction_fg = fs.get_feature_group(
            name=PREDICTION_FG_NAME,
            version=PREDICTION_FG_VERSION,
        )

        if prediction_fg is not None:

            print(
                f"\n--- Using existing prediction Feature Group "
                f"{PREDICTION_FG_NAME} v{PREDICTION_FG_VERSION} ---"
            )

            return prediction_fg

    except Exception:

        print(
            "\n--- Prediction Feature Group does not exist yet ---"
        )

    # -------------------------------------------------------------
    # Create Feature Group.
    # -------------------------------------------------------------

    print(
        "\n--- Creating prediction Feature Group ---"
    )

    prediction_fg = fs.create_feature_group(
        name=PREDICTION_FG_NAME,
        version=PREDICTION_FG_VERSION,
        description=(
            "Automatically generated Karachi AQI "
            "predictions for Day +1, Day +2 and Day +3."
        ),
        primary_key=[
            "timestamp"
        ],
        event_time="timestamp",
        online_enabled=False,
    )

    print(
        f"--- Prediction Feature Group created: "
        f"{PREDICTION_FG_NAME} v{PREDICTION_FG_VERSION} ---"
    )

    return prediction_fg


# =====================================================================
# CHECK EXISTING PREDICTION
# =====================================================================

def prediction_already_exists(
    prediction_fg,
    timestamp,
):
    """
    Check whether a prediction already exists
    for the given timestamp.
    """

    try:

        dataframe = prediction_fg.read(
            dataframe_type="pandas"
        )

    except Exception as exc:

        print(
            "Could not read existing predictions. "
            "Assuming no duplicate exists."
        )

        print(
            f"Reason: {exc}"
        )

        return False

    if dataframe is None or dataframe.empty:

        return False

    if "timestamp" not in dataframe.columns:

        return False

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="coerce",
    )

    timestamp = pd.Timestamp(
        timestamp
    )

    return bool(
        (
            dataframe["timestamp"]
            == timestamp
        ).any()
    )


# =====================================================================
# STORE PREDICTION
# =====================================================================

def store_prediction(
    prediction_fg,
    result,
):
    """Store generated prediction."""

    timestamp = pd.Timestamp(
        result["timestamp"]
    )

    print(
        "\nPrediction timestamp:",
        timestamp,
    )

    # -------------------------------------------------------------
    # Prevent duplicate predictions.
    # -------------------------------------------------------------

    if prediction_already_exists(
        prediction_fg,
        timestamp,
    ):

        print(
            "Prediction already exists for this timestamp."
        )

        return

    # -------------------------------------------------------------
    # Prepare prediction row.
    # -------------------------------------------------------------

    prediction_dataframe = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "current_aqi": float(
                    result["current_aqi"]
                ),
                "day1": float(
                    result["day1"]
                ),
                "day2": float(
                    result["day2"]
                ),
                "day3": float(
                    result["day3"]
                ),
                "day1_model_version": int(
                    result["day1_model_version"]
                ),
                "day2_model_version": int(
                    result["day2_model_version"]
                ),
                "day3_model_version": int(
                    result["day3_model_version"]
                ),
            }
        ]
    )

    # -------------------------------------------------------------
    # Insert prediction.
    # -------------------------------------------------------------

    print(
        "\n--- Storing prediction in Hopsworks ---"
    )

    prediction_fg.insert(
        prediction_dataframe
    )

    print(
        "--- Prediction stored successfully ---"
    )

    print(
        prediction_dataframe.to_string(
            index=False
        )
    )


# =====================================================================
# MAIN AUTOMATION
# =====================================================================

def run_prediction_automation():
    """
    Generate and store the latest AQI prediction.
    """

    print(
        "\n=============================================="
    )
    print(
        " KARACHI AQI AUTOMATED PREDICTION"
    )
    print(
        "=============================================="
    )

    # -------------------------------------------------------------
    # 1. Connect to Hopsworks.
    # -------------------------------------------------------------

    project = connect_to_hopsworks()

    # -------------------------------------------------------------
    # 2. Get or create prediction Feature Group.
    # -------------------------------------------------------------

    prediction_fg = get_prediction_feature_group(
        project
    )

    # -------------------------------------------------------------
    # 3. Generate production prediction.
    # -------------------------------------------------------------

    print(
        "\n--- Generating production prediction ---"
    )

    result = predict()

    # -------------------------------------------------------------
    # 4. Store prediction.
    # -------------------------------------------------------------

    store_prediction(
        prediction_fg,
        result,
    )

    print(
        "\n=============================================="
    )
    print(
        " AUTOMATED PREDICTION SUCCESS"
    )
    print(
        "=============================================="
    )


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================

if __name__ == "__main__":

    try:

        run_prediction_automation()

    except Exception as exc:

        print(
            "\n=============================================="
        )
        print(
            " AUTOMATED PREDICTION FAILED"
        )
        print(
            "=============================================="
        )

        print(
            f"\nError: {exc}"
        )

        raise
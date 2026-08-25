import os
import sys
from pathlib import Path

import hopsworks
import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------
# Project root / src import
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from feature_engineering import (  # noqa: E402
    MODEL_FEATURES,
    REQUIRED_RAW_COLUMNS,
    MAX_LOOKBACK_HOURS,
    build_rich_features,
    validate_feature_frame,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 5


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\n==============================================")
    print(" V5 RAW → 100 FEATURE COMPATIBILITY TEST")
    print("==============================================")

    # -----------------------------------------------------
    # 1. Load environment
    # -----------------------------------------------------

    load_dotenv(
        dotenv_path=ROOT_DIR / ".env"
    )

    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is not set."
        )

    # -----------------------------------------------------
    # 2. Connect to Hopsworks
    # -----------------------------------------------------

    print("\n--- Connecting to Hopsworks ---")

    project = hopsworks.login(
        api_key_value=api_key,
        host="eu-west.cloud.hopsworks.ai",
    )

    feature_store = project.get_feature_store()

    print("Hopsworks connection: PASSED")

    # -----------------------------------------------------
    # 3. Retrieve v5
    # -----------------------------------------------------

    print(
        f"\n--- Reading {FEATURE_GROUP_NAME} "
        f"version {FEATURE_GROUP_VERSION} ---"
    )

    feature_group = feature_store.get_feature_group(
        FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    print("Feature Group retrieval: PASSED")

    # -----------------------------------------------------
    # 4. Read raw historical data
    # -----------------------------------------------------

    print("\n--- Reading v5 data ---")

    df = feature_group.select_all().read(
        read_options={
            "use_arrow_flight": False
        }
    )

    print(
        "Raw v5 shape:",
        df.shape
    )

    print(
        "\nRaw v5 columns:"
    )

    print(
        list(df.columns)
    )

    # -----------------------------------------------------
    # 5. Verify raw schema
    # -----------------------------------------------------

    print(
        "\n--- Checking Required Raw Columns ---"
    )

    missing_raw = [
        column
        for column in REQUIRED_RAW_COLUMNS
        if column not in df.columns
    ]

    if missing_raw:

        raise RuntimeError(
            "v5 is missing required raw columns: "
            + ", ".join(missing_raw)
        )

    print(
        "All required raw columns are present."
    )

    # -----------------------------------------------------
    # 6. Remove Hopsworks metadata columns if necessary
    # -----------------------------------------------------

    # city and timestamp are expected.
    # The shared feature engineering function only
    # requires the columns listed in REQUIRED_RAW_COLUMNS.

    df = df[
        list(REQUIRED_RAW_COLUMNS)
    ].copy()

    # -----------------------------------------------------
    # 7. Verify raw missing values
    # -----------------------------------------------------

    print(
        "\n--- Checking Raw Missing Values ---"
    )

    missing_counts = (
        df[list(REQUIRED_RAW_COLUMNS)]
        .isna()
        .sum()
    )

    print(missing_counts)

    if missing_counts.sum() > 0:

        raise RuntimeError(
            "v5 contains missing raw values."
        )

    print(
        "Raw missing-value check: PASSED"
    )

    # -----------------------------------------------------
    # 8. Verify raw history
    # -----------------------------------------------------

    print(
        "\n--- Checking Historical Context ---"
    )

    print(
        "Rows available:",
        len(df)
    )

    print(
        "Maximum feature lookback:",
        MAX_LOOKBACK_HOURS,
    )

    if len(df) <= MAX_LOOKBACK_HOURS:

        raise RuntimeError(
            "v5 does not contain enough historical "
            "observations for feature engineering."
        )

    print(
        "Historical context check: PASSED"
    )

    # -----------------------------------------------------
    # 9. Run SHARED feature engineering
    # -----------------------------------------------------

    print("\n==============================================")
    print(" RUNNING SHARED FEATURE ENGINEERING")
    print("==============================================")

    features = build_rich_features(
        df
    )

    print(
        "\nGenerated feature frame shape:",
        features.shape
    )

    # -----------------------------------------------------
    # 10. Verify exactly 100 canonical features
    # -----------------------------------------------------

    print(
        "\n--- Checking MODEL_FEATURES ---"
    )

    print(
        "Expected model features:",
        len(MODEL_FEATURES)
    )

    actual_model_features = [
        column
        for column in features.columns
        if column in MODEL_FEATURES
    ]

    print(
        "Generated model features:",
        len(actual_model_features)
    )

    if len(MODEL_FEATURES) != 100:

        raise RuntimeError(
            "MODEL_FEATURES definition is not 100."
        )

    if actual_model_features != list(
        MODEL_FEATURES
    ):

        raise RuntimeError(
            "Generated feature order does not "
            "match MODEL_FEATURES."
        )

    print(
        "100-feature schema check: PASSED"
    )

    # -----------------------------------------------------
    # 11. Remove warm-up rows
    # -----------------------------------------------------

    print(
        "\n--- Checking Feature Warm-up ---"
    )

    complete_features = features.dropna(
        subset=list(MODEL_FEATURES)
    ).copy()

    print(
        "Complete rows after warm-up:",
        len(complete_features)
    )

    if complete_features.empty:

        raise RuntimeError(
            "No complete feature rows were produced."
        )

    # -----------------------------------------------------
    # 12. Select latest complete row
    # -----------------------------------------------------

    latest = complete_features.iloc[
        [-1]
    ].copy()

    print(
        "Latest complete timestamp:",
        latest["timestamp"].iloc[0]
    )

    # -----------------------------------------------------
    # 13. Validate canonical feature frame
    # -----------------------------------------------------

    print(
        "\n--- Validating Canonical Feature Frame ---"
    )

    validate_feature_frame(
        latest,
        require_complete=True,
    )

    print(
        "Canonical feature validation: PASSED"
    )

    # -----------------------------------------------------
    # 14. Final NaN check
    # -----------------------------------------------------

    nan_count = int(
        latest[
            list(MODEL_FEATURES)
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Latest 100-feature NaN count:",
        nan_count
    )

    if nan_count != 0:

        raise RuntimeError(
            "Latest 100-feature row contains NaN values."
        )

    # -----------------------------------------------------
    # SUCCESS
    # -----------------------------------------------------

    print("\n==============================================")
    print(" V5 → 100 FEATURES TEST PASSED")
    print("==============================================")

    print(
        "\nHopsworks v5 raw historical data can be"
    )

    print(
        "successfully transformed by the shared"
    )

    print(
        "feature_engineering.py into the canonical"
    )

    print(
        "100 MODEL_FEATURES used by the project."
    )

    print("\n==============================================")


if __name__ == "__main__":
    main()
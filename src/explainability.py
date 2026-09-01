"""
SHAP Explainability Pipeline for Karachi AQI Predictor.

Purpose
-------
Explain why the trained production models generated their AQI predictions.

Models:
    Day +1 -> XGBoost
    Day +2 -> Ridge
    Day +3 -> Ridge

This file:
    1. Reads recent production features from Hopsworks v6.
    2. Loads the latest registered production model.
    3. Uses the exact feature contract stored in model metadata.
    4. Calculates SHAP values for one latest feature row.
    5. Returns the most influential features.

Important
---------
This file does NOT:
    - retrain models
    - modify models
    - register models
    - modify the prediction pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.feature_engineering import MODEL_FEATURES

from src.predict import (
    MODEL_NAMES,
    connect_to_hopsworks,
    load_registered_model,
    validate_model_metadata,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 6

# Number of recent rows used as SHAP background data.
# This is NOT used as training data.
BACKGROUND_ROWS = 100

# Number of features returned to the dashboard.
TOP_FEATURES = 10


# =====================================================================
# READ RECENT V6 FEATURES
# =====================================================================

def get_recent_v6_features(
    project,
    hours: int = 200,
) -> pd.DataFrame:
    """
    Read recent Karachi production features from Hopsworks v6.

    These rows are used only as background data for SHAP.
    """

    print(
        "\n--- Reading recent v6 features for SHAP ---"
    )

    fs = project.get_feature_store()

    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    start_time = (
        now
        - pd.Timedelta(hours=hours)
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
            f"Failed to read v6 features for SHAP: {exc}"
        ) from exc

    if dataframe is None or dataframe.empty:

        raise RuntimeError(
            "No recent v6 feature rows available for SHAP."
        )

    dataframe = dataframe.copy()

    # -------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------

    if "timestamp" not in dataframe.columns:

        raise RuntimeError(
            "v6 data does not contain timestamp."
        )

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
    # Karachi only
    # -------------------------------------------------------------

    if "city" in dataframe.columns:

        dataframe = dataframe[
            dataframe["city"]
            .astype(str)
            .str.lower()
            == "karachi"
        ].copy()

    if dataframe.empty:

        raise RuntimeError(
            "No Karachi feature rows found in v6."
        )

    # -------------------------------------------------------------
    # Sort chronologically
    # -------------------------------------------------------------

    dataframe = (
        dataframe
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Validate canonical features
    # -------------------------------------------------------------

    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in dataframe.columns
    ]

    if missing:

        raise RuntimeError(
            "v6 is missing canonical MODEL_FEATURES: "
            f"{missing}"
        )

    # -------------------------------------------------------------
    # Remove rows containing NaN in canonical features
    # -------------------------------------------------------------

    dataframe = dataframe.dropna(
        subset=list(MODEL_FEATURES)
    )

    if dataframe.empty:

        raise RuntimeError(
            "No complete v6 rows available for SHAP."
        )

    return dataframe


# =====================================================================
# PREPARE FEATURES
# =====================================================================

def prepare_shap_data(
    dataframe: pd.DataFrame,
    metadata: dict,
):
    """
    Prepare:

        background data
        latest observation

    using the exact feature order expected by the model.
    """

    selected_features = metadata.get(
        "selected_features"
    )

    if selected_features:

        feature_columns = list(
            selected_features
        )

    else:

        feature_columns = list(
            MODEL_FEATURES
        )

    # -------------------------------------------------------------
    # Verify columns
    # -------------------------------------------------------------

    missing = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]

    if missing:

        raise RuntimeError(
            "SHAP input is missing required model features: "
            f"{missing}"
        )

    # -------------------------------------------------------------
    # Exact feature order
    # -------------------------------------------------------------

    feature_data = dataframe[
        feature_columns
    ].copy()

    # -------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------

    for column in feature_columns:

        feature_data[column] = pd.to_numeric(
            feature_data[column],
            errors="coerce",
        )

    # -------------------------------------------------------------
    # Remove invalid rows
    # -------------------------------------------------------------

    feature_data = feature_data.dropna()

    if feature_data.empty:

        raise RuntimeError(
            "No valid numeric rows available for SHAP."
        )

    # -------------------------------------------------------------
    # Background data
    # -------------------------------------------------------------

    background = feature_data.tail(
        min(
            BACKGROUND_ROWS,
            len(feature_data),
        )
    ).copy()

    # -------------------------------------------------------------
    # Latest row
    # -------------------------------------------------------------

    latest_row = feature_data.iloc[
        [-1]
    ].copy()

    return (
        background,
        latest_row,
        feature_columns,
    )


# =====================================================================
# CALCULATE SHAP VALUES
# =====================================================================

def calculate_shap_values(
    model,
    background: pd.DataFrame,
    latest_row: pd.DataFrame,
    model_name: str,
):
    """
    Calculate SHAP values.

    XGBoost:
        TreeExplainer

    Ridge / linear models:
        LinearExplainer

    Fallback:
        Generic SHAP Explainer
    """

    print(
        f"\n--- Calculating SHAP values for {model_name} ---"
    )

    # -------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------

    if "xgboost" in model_name.lower():

        try:

            explainer = shap.TreeExplainer(
                model
            )

            shap_result = explainer(
                latest_row
            )

            values = shap_result.values

            base_value = shap_result.base_values

            if values.ndim == 2:

                values = values[0]

            if np.ndim(base_value) > 0:

                base_value = np.asarray(
                    base_value
                ).reshape(-1)[0]

            base_value = float(
                base_value
            )

            return (
                np.asarray(values),
                base_value,
            )

        except Exception as exc:

            print(
                "TreeExplainer failed. "
                f"Trying generic SHAP: {exc}"
            )

    # -------------------------------------------------------------
    # Linear / Ridge
    # -------------------------------------------------------------

    if hasattr(model, "coef_"):

        try:

            explainer = shap.LinearExplainer(
                model,
                background,
            )

            shap_result = explainer(
                latest_row
            )

            values = shap_result.values

            if values.ndim == 2:

                values = values[0]

            base_value = shap_result.base_values

            if np.ndim(base_value) > 0:

                base_value = np.asarray(
                    base_value
                ).reshape(-1)[0]

            base_value = float(
                base_value
            )

            return (
                np.asarray(values),
                base_value,
            )

        except Exception as exc:

            print(
                "LinearExplainer failed. "
                f"Trying generic SHAP: {exc}"
            )

    # -------------------------------------------------------------
    # Generic fallback
    # -------------------------------------------------------------

    try:

        prediction_function = (
            lambda data:
            model.predict(data)
        )

        explainer = shap.Explainer(
            prediction_function,
            background,
        )

        shap_result = explainer(
            latest_row
        )

        values = shap_result.values

        if values.ndim == 2:

            values = values[0]

        base_value = shap_result.base_values

        if np.ndim(base_value) > 0:

            base_value = np.asarray(
                base_value
            ).reshape(-1)[0]

        base_value = float(
            base_value
        )

        return (
            np.asarray(values),
            base_value,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to calculate SHAP values "
            f"for {model_name}: {exc}"
        ) from exc


# =====================================================================
# FORMAT SHAP RESULTS
# =====================================================================

def format_shap_results(
    feature_names,
    shap_values,
    latest_row,
    top_n: int = TOP_FEATURES,
):
    """
    Convert SHAP values into dashboard-friendly JSON data.
    """

    values = np.asarray(
        shap_values,
        dtype=float,
    )

    if len(values) != len(feature_names):

        raise RuntimeError(
            "SHAP value count does not match "
            "feature count."
        )

    feature_values = (
        latest_row.iloc[0]
        .to_dict()
    )

    rows = []

    for index, feature in enumerate(
        feature_names
    ):

        shap_value = float(
            values[index]
        )

        raw_value = feature_values.get(
            feature
        )

        try:

            feature_value = float(
                raw_value
            )

        except (TypeError, ValueError):

            feature_value = None

        rows.append(
            {
                "feature": feature,
                "feature_value": feature_value,
                "shap_value": round(
                    shap_value,
                    4,
                ),
                "absolute_shap_value": round(
                    abs(shap_value),
                    4,
                ),
                "impact": (
                    "increases_prediction"
                    if shap_value > 0
                    else "decreases_prediction"
                    if shap_value < 0
                    else "neutral"
                ),
            }
        )

    # -------------------------------------------------------------
    # Most influential features first
    # -------------------------------------------------------------

    rows.sort(
        key=lambda item:
        item["absolute_shap_value"],
        reverse=True,
    )

    return rows[:top_n]


# =====================================================================
# EXPLAIN ONE MODEL
# =====================================================================

def explain_model(
    project,
    registry,
    model_name: str,
):
    """
    Generate SHAP explanation for one production model.
    """

    # -------------------------------------------------------------
    # Load latest registered model
    # -------------------------------------------------------------

    (
        model,
        metadata,
        model_version,
    ) = load_registered_model(
        registry,
        model_name,
    )

    # -------------------------------------------------------------
    # Validate production contract
    # -------------------------------------------------------------

    validate_model_metadata(
        metadata,
        model_name,
    )

    # -------------------------------------------------------------
    # Read recent production features
    # -------------------------------------------------------------

    dataframe = get_recent_v6_features(
        project
    )

    # -------------------------------------------------------------
    # Prepare background + latest row
    # -------------------------------------------------------------

    (
        background,
        latest_row,
        feature_columns,
    ) = prepare_shap_data(
        dataframe,
        metadata,
    )

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------

    prediction = float(
        model.predict(
            latest_row
        )[0]
    )

    # -------------------------------------------------------------
    # SHAP
    # -------------------------------------------------------------

    (
        shap_values,
        base_value,
    ) = calculate_shap_values(
        model=model,
        background=background,
        latest_row=latest_row,
        model_name=model_name,
    )

    # -------------------------------------------------------------
    # Format
    # -------------------------------------------------------------

    top_features = format_shap_results(
        feature_names=feature_columns,
        shap_values=shap_values,
        latest_row=latest_row,
    )

    return {
        "model_name": model_name,
        "model_version": int(
            model_version
        ),
        "prediction": round(
            prediction,
            2,
        ),
        "base_value": round(
            base_value,
            4,
        ),
        "timestamp": dataframe["timestamp"].iloc[-1],
        "features": top_features,
    }


# =====================================================================
# MAIN EXPLANATION FUNCTION
# =====================================================================

def explain_predictions():
    """
    Generate SHAP explanations for Day +1, Day +2 and Day +3.

    The latest production v6 features are read ONCE and reused
    for all three models.
    """

    print(
        "\n=============================================="
    )
    print(
        " KARACHI AQI SHAP EXPLAINABILITY"
    )
    print(
        "=============================================="
    )

    # -------------------------------------------------------------
    # Hopsworks
    # -------------------------------------------------------------

    project = connect_to_hopsworks()

    registry = (
        project.get_model_registry()
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    # Read v6 features ONLY ONCE.
    # -------------------------------------------------------------

    dataframe = get_recent_v6_features(
        project
    )

    print(
        "--- Reusing same v6 feature data for all models ---"
    )

    # -------------------------------------------------------------
    # Explain all three models using the SAME dataframe
    # -------------------------------------------------------------

    def explain_with_dataframe(model_name):

        (
            model,
            metadata,
            model_version,
        ) = load_registered_model(
            registry,
            model_name,
        )

        validate_model_metadata(
            metadata,
            model_name,
        )

        (
            background,
            latest_row,
            feature_columns,
        ) = prepare_shap_data(
            dataframe,
            metadata,
        )

        prediction = float(
            model.predict(
                latest_row
            )[0]
        )

        (
            shap_values,
            base_value,
        ) = calculate_shap_values(
            model=model,
            background=background,
            latest_row=latest_row,
            model_name=model_name,
        )

        top_features = format_shap_results(
            feature_names=feature_columns,
            shap_values=shap_values,
            latest_row=latest_row,
        )

        return {
            "model_name": model_name,
            "model_version": int(
                model_version
            ),
            "prediction": round(
                prediction,
                2,
            ),
            "base_value": round(
                base_value,
                4,
            ),
            "timestamp": dataframe[
                "timestamp"
            ].iloc[-1],
            "features": top_features,
        }

    # -------------------------------------------------------------
    # Generate explanations
    # -------------------------------------------------------------

    day1 = explain_with_dataframe(
        MODEL_NAMES["day1"]
    )

    day2 = explain_with_dataframe(
        MODEL_NAMES["day2"]
    )

    day3 = explain_with_dataframe(
        MODEL_NAMES["day3"]
    )

    # -------------------------------------------------------------
    # Final response
    # -------------------------------------------------------------

    result = {
        "timestamp": day1["timestamp"],
        "day1": day1,
        "day2": day2,
        "day3": day3,
    }

    print(
        "\n--- SHAP explanation generated successfully ---"
    )

    return result

# =====================================================================
# STANDALONE EXECUTION
# =====================================================================

if __name__ == "__main__":

    try:

        result = explain_predictions()

        print(
            "\n=============================================="
        )
        print(
            " SHAP RESULTS"
        )
        print(
            "=============================================="
        )

        for day_key in [
            "day1",
            "day2",
            "day3",
        ]:

            explanation = result[
                day_key
            ]

            print(
                f"\n{day_key.upper()} "
                f"Prediction: "
                f"{explanation['prediction']}"
            )

            print(
                "Top influential features:"
            )

            for feature in explanation[
                "features"
            ]:

                print(
                    f"  "
                    f"{feature['feature']}: "
                    f"{feature['shap_value']:+.4f} "
                    f"({feature['impact']})"
                )

    except Exception as exc:

        print(
            "\n=============================================="
        )
        print(
            " SHAP EXPLAINABILITY FAILED"
        )
        print(
            "=============================================="
        )

        print(
            f"\nError: {exc}"
        )

        raise
"""
Day +1 training pipeline for Karachi AQI Predictor — v6.

Purpose:
    Train the Day +1 XGBoost model using the canonical computed
    100 MODEL_FEATURES stored directly in Hopsworks Feature Group v6.

Important:
    - v6 contains COMPUTED features.
    - Feature engineering is NOT performed again here.
    - v6 is read directly from Hopsworks.
    - The 100 MODEL_FEATURES must exactly match the canonical
      feature contract from feature_engineering.py.
    - target_day1 is created from target_aqi shifted 24 hours forward.
"""

from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks
import xgboost as xgb

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from feature_engineering import MODEL_FEATURES


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------
# Hopsworks configuration
# ---------------------------------------------------------------------

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 6


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODEL_FILE = "karachi_aqi_day1_xgboost_v6.pkl"
METRICS_FILE = "karachi_aqi_day1_metrics_v6.json"

FORECAST_HOURS = 24


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("\n==============================================")
    print(" KARACHI AQI — DAY +1 TRAINING — V6")
    print("==============================================")


    # -------------------------------------------------------------
    # 1. Connect to Hopsworks
    # -------------------------------------------------------------

    print("\n--- Connecting to Hopsworks Feature Store ---")

    project = hopsworks.login()

    fs = project.get_feature_store()


    # -------------------------------------------------------------
    # 2. Read COMPUTED features from v6
    # -------------------------------------------------------------

    print("\n--- Fetching Computed Feature Group v6 ---")

    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    df = feature_group.read(
        online=False,
        read_options={
            "use_arrow_flight": False
        },
    )

    print(
        "\nV6 dataset shape:",
        df.shape,
    )

    print(
        "V6 columns:",
        list(df.columns),
    )

    print(
        "V6 total missing values:",
        int(df.isna().sum().sum()),
    )


    # -------------------------------------------------------------
    # 3. Basic dataset validation
    # -------------------------------------------------------------

    print("\n--- Validating V6 Dataset ---")

    if df.empty:
        raise RuntimeError(
            "V6 Feature Group returned an empty dataset."
        )

    if df.isna().sum().sum() != 0:
        missing = (
            df.isna()
            .sum()
        )

        missing = missing[
            missing > 0
        ]

        raise RuntimeError(
            "V6 dataset contains missing values:\n"
            f"{missing}"
        )

    required_columns = [
        "city",
        "timestamp",
        "target_aqi",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:
        raise RuntimeError(
            "V6 is missing required columns: "
            + ", ".join(missing_required)
        )


    # -------------------------------------------------------------
    # 4. Validate canonical MODEL_FEATURES
    # -------------------------------------------------------------

    print("\n--- Validating Canonical MODEL_FEATURES ---")

    print(
        "Expected MODEL_FEATURES:",
        len(MODEL_FEATURES),
    )

    if len(MODEL_FEATURES) != 100:
        raise RuntimeError(
            "Expected exactly 100 MODEL_FEATURES."
        )

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "V6 is missing MODEL_FEATURES:\n"
            + "\n".join(missing_features)
        )

    actual_features = [
        column
        for column in df.columns
        if column in MODEL_FEATURES
    ]

    if actual_features != list(MODEL_FEATURES):
        raise RuntimeError(
            "V6 MODEL_FEATURES order does not match "
            "the canonical MODEL_FEATURES order."
        )

    print(
        "100-feature schema check: PASSED"
    )


    # -------------------------------------------------------------
    # 5. Sort by timestamp
    # -------------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )
    time_diff = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    if not (time_diff == pd.Timedelta(hours=1)).all():
        raise RuntimeError(
            "V6 timestamps are not a continuous hourly series."
        )

    print(
        "Hourly timestamp continuity check: PASSED"
    )


    print(
        "\nFirst V6 timestamp:",
        df["timestamp"].iloc[0],
    )

    print(
        "Last V6 timestamp:",
        df["timestamp"].iloc[-1],
    )


    # -------------------------------------------------------------
    # 6. Create Day +1 target
    #
    # Current row:
    #
    #       t
    #
    # Target:
    #
    #       t + 24 hours
    #
    # Therefore:
    #
    # target_day1 = target_aqi.shift(-24)
    # -------------------------------------------------------------

    print("\n--- Creating Day +1 Target ---")

    df["target_day1"] = (
        df["target_aqi"]
        .shift(-FORECAST_HOURS)
    )


    # -------------------------------------------------------------
    # 7. Remove rows where future target is unavailable
    # -------------------------------------------------------------

    training_df = df.dropna(
        subset=[
            *MODEL_FEATURES,
            "target_day1",
        ]
    ).copy()

    print(
        "\nUsable Day +1 training rows:",
        len(training_df),
    )

    print(
        "Rows removed:",
        len(df) - len(training_df),
    )


    if training_df.empty:
        raise RuntimeError(
            "No usable Day +1 training rows."
        )


    # -------------------------------------------------------------
    # 8. Build X and y
    # -------------------------------------------------------------

    X = training_df[
        list(MODEL_FEATURES)
    ]

    y = training_df[
        "target_day1"
    ]


    # -------------------------------------------------------------
    # 9. Final X validation
    # -------------------------------------------------------------

    print("\n--- Final Training Matrix Validation ---")

    print(
        "X shape:",
        X.shape,
    )

    print(
        "y shape:",
        y.shape,
    )

    print(
        "X feature count:",
        len(X.columns),
    )

    print(
        "target_day1 in X:",
        "target_day1" in X.columns,
    )

    print(
        "target_aqi in X:",
        "target_aqi" in X.columns,
    )

    print(
        "X columns exactly match MODEL_FEATURES:",
        list(X.columns) == list(MODEL_FEATURES),
    )

    if len(X.columns) != 100:
        raise RuntimeError(
            "X does not contain exactly 100 features."
        )

    if list(X.columns) != list(MODEL_FEATURES):
        raise RuntimeError(
            "X columns do not exactly match "
            "canonical MODEL_FEATURES."
        )

    if "target_day1" in X.columns:
        raise RuntimeError(
            "Target leakage detected: "
            "target_day1 is present in X."
        )

    if "target_aqi" in X.columns:
            raise RuntimeError(
            "target_aqi must not be directly present in X. "
            "Historical AQI information is represented through "
            "the canonical target_aqi lag/rolling features."
        )


    # -------------------------------------------------------------
    # 10. Time-ordered train/test split
    #
    # IMPORTANT:
    #     shuffle=False
    #
    # This preserves chronological order for time-series data.
    # -------------------------------------------------------------

    print("\n--- Creating Time-Ordered Train/Test Split ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    print(
        "Training rows:",
        len(X_train),
    )

    print(
        "Testing rows:",
        len(X_test),
    )

    print(
        "Training period:",
        training_df.loc[
            X_train.index[0],
            "timestamp",
        ],
        "→",
        training_df.loc[
            X_train.index[-1],
            "timestamp",
        ],
    )

    print(
        "Testing period:",
        training_df.loc[
            X_test.index[0],
            "timestamp",
        ],
        "→",
        training_df.loc[
            X_test.index[-1],
            "timestamp",
        ],
    )


    # -------------------------------------------------------------
    # 11. Train XGBoost
    # -------------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        " TRAINING OPTIMIZED XGBOOST — DAY +1"
    )

    print(
        "=============================================="
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
    )

    xgb_model.fit(
        X_train,
        y_train,
    )


    # -------------------------------------------------------------
    # 12. Predictions
    # -------------------------------------------------------------

    preds = xgb_model.predict(
        X_test
    )


    # -------------------------------------------------------------
    # 13. XGBoost metrics
    # -------------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        preds,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds,
        )
    )

    r2 = r2_score(
        y_test,
        preds,
    )

    print(
        "\n📊 [Day +1 — V6 XGBoost]"
    )

    print(
        f"  - MAE : {mae:.4f}"
    )

    print(
        f"  - RMSE: {rmse:.4f}"
    )

    print(
        f"  - R²  : {r2:.4f}"
    )


    # -------------------------------------------------------------
    # 14. Persistence baseline
    #
    # Predict tomorrow's AQI using the current AQI.
    # -------------------------------------------------------------

    baseline_preds = training_df.loc[
        X_test.index,
        "target_aqi",
    ]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_preds,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_preds,
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_preds,
    )

    print(
        "\n📊 [Day +1 — Persistence Baseline]"
    )

    print(
        f"  - MAE : {baseline_mae:.4f}"
    )

    print(
        f"  - RMSE: {baseline_rmse:.4f}"
    )

    print(
        f"  - R²  : {baseline_r2:.4f}"
    )


    # -------------------------------------------------------------
    # 15. Compare model against baseline
    # -------------------------------------------------------------

    print(
        "\n--- Model vs Persistence Baseline ---"
    )

    print(
        f"MAE improvement : "
        f"{baseline_mae - mae:.4f}"
    )

    print(
        f"RMSE improvement: "
        f"{baseline_rmse - rmse:.4f}"
    )

    print(
        f"R² improvement  : "
        f"{r2 - baseline_r2:.4f}"
    )
        # -------------------------------------------------------------
    # 16. Save metrics dynamically
    # -------------------------------------------------------------

    metrics = {
        "model_mae": float(mae),
        "model_rmse": float(rmse),
        "model_r2": float(r2),
        "baseline_mae": float(baseline_mae),
        "baseline_rmse": float(baseline_rmse),
        "baseline_r2": float(baseline_r2),
        "mae_improvement": float(baseline_mae - mae),
        "rmse_improvement": float(baseline_rmse - rmse),
        "r2_improvement": float(r2 - baseline_r2),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        "\nMetrics saved dynamically as:",
        METRICS_FILE,
    )

    # -------------------------------------------------------------
    # 17. Save model
    # -------------------------------------------------------------

    joblib.dump(
        xgb_model,
        MODEL_FILE,
    )

    print(
        "\nModel saved locally as:",
        MODEL_FILE,
    )


    # -------------------------------------------------------------
    # 18. Final summary
    # -------------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        " DAY +1 V6 TRAINING COMPLETE"
    )

    print(
        "=============================================="
    )

    print(
        f"Feature Group : "
        f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
    )

    print(
        f"Dataset rows  : {len(df)}"
    )

    print(
        f"Training rows : {len(X_train)}"
    )

    print(
        f"Testing rows  : {len(X_test)}"
    )

    print(
        f"Features      : {len(MODEL_FEATURES)}"
    )

    print(
        f"MAE           : {mae:.4f}"
    )

    print(
        f"RMSE          : {rmse:.4f}"
    )

    print(
        f"R²            : {r2:.4f}"
    )

    print(
        f"Model         : {MODEL_FILE}"
    )

    print(
        "Source        : Hopsworks computed Feature Group v6"
    )

    print(
        "Feature engineering: already performed during v6 backfill"
    )

    print(
        "=============================================="
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
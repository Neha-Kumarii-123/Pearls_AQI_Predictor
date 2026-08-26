import json

from dotenv import load_dotenv

import hopsworks
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from feature_engineering import MODEL_FEATURES


load_dotenv()


FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 6

MODEL_FILE = "karachi_aqi_day2_ridge_v6.pkl"
FEATURES_FILE = "karachi_aqi_day2_features_v6.pkl"
METRICS_FILE = "karachi_aqi_day2_metrics_v6.json"

FORECAST_HOURS = 48
SELECTED_FEATURE_COUNT = 60
TEST_SIZE = 0.20


def main():

    print("\n==============================================")
    print(" KARACHI AQI — DAY +2 TRAINING — V6")
    print("==============================================")

    # ---------------------------------------------------------
    # 1. Connect to Hopsworks
    # ---------------------------------------------------------

    print("\n--- Connecting to Hopsworks Feature Store ---")

    project = hopsworks.login()
    fs = project.get_feature_store()

    # ---------------------------------------------------------
    # 2. Read COMPUTED features from V6
    # ---------------------------------------------------------

    print("\n--- Fetching Computed Feature Group v6 ---")

    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    df = feature_group.read(
        online=False,
        read_options={
            "arrow_flight_config": {
                "timeout": 900
            }
        },
    )

    print(
        "\nV6 dataset shape:",
        df.shape,
    )

    print(
        "V6 total missing values:",
        int(df.isna().sum().sum()),
    )

    if df.empty:
        raise RuntimeError(
            "V6 Feature Group returned no rows."
        )

    if df.isna().sum().sum() != 0:
        raise RuntimeError(
            "V6 dataset contains missing values."
        )

    # ---------------------------------------------------------
    # 3. Validate required columns
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Validate Karachi dataset
    # ---------------------------------------------------------

    cities = (
        df["city"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    if len(cities) != 1 or cities[0] != "karachi":
        raise RuntimeError(
            f"Expected Karachi-only V6 data, found cities: {cities}"
        )

    print("City validation: PASSED — Karachi")

    # ---------------------------------------------------------
    # 4. Validate canonical MODEL_FEATURES
    # ---------------------------------------------------------

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
            "V6 is missing MODEL_FEATURES: "
            + ", ".join(missing_features)
        )

    actual_features = [
        column
        for column in df.columns
        if column in MODEL_FEATURES
    ]

    if actual_features != list(MODEL_FEATURES):
        raise RuntimeError(
            "V6 feature columns do not match "
            "canonical MODEL_FEATURES order."
        )

    print("100-feature schema check: PASSED")

    # ---------------------------------------------------------
    # 5. Sort chronologically
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        "\nFirst V6 timestamp:",
        df["timestamp"].iloc[0],
    )

    print(
        "Last V6 timestamp:",
        df["timestamp"].iloc[-1],
    )

    # ---------------------------------------------------------
    # Validate hourly continuity
    # ---------------------------------------------------------

    time_diff = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    if not (
        time_diff == pd.Timedelta(hours=1)
    ).all():
        raise RuntimeError(
            "V6 timestamps are not a continuous hourly series."
        )

    print(
        "Hourly timestamp continuity check: PASSED"
    )

    # ---------------------------------------------------------
    # 6. Create Day +2 target
    #
    # Current row t → AQI at t + 48 hours
    # ---------------------------------------------------------

    print("\n--- Creating Day +2 Target ---")

    df["target_day2"] = (
        df["target_aqi"]
        .shift(-FORECAST_HOURS)
    )

    # ---------------------------------------------------------
    # 7. Remove rows where target_day2 is unavailable
    #
    # V6 is already warm-up cleaned.
    #
    # Therefore, for Day +2:
    #
    # 17376 rows
    # -   48 rows without future target
    # = 17328 usable rows
    # ---------------------------------------------------------

    eval_df = df.dropna(
        subset=[
            *MODEL_FEATURES,
            "target_day2",
        ]
    ).copy()

    print(
        "\nUsable Day +2 training rows:",
        len(eval_df),
    )

    actual_removed_rows = (
        len(df) - len(eval_df)
    )

    print(
        "Rows removed:",
        actual_removed_rows,
    )

    if actual_removed_rows != FORECAST_HOURS:
        raise RuntimeError(
            f"Expected exactly {FORECAST_HOURS} rows to be removed "
            f"for Day +2 target creation, but "
            f"{actual_removed_rows} were removed."
        )

    print(
        f"Target horizon validation: PASSED — "
        f"{FORECAST_HOURS}-hour target / "
        f"{actual_removed_rows} rows removed"
    )

    # ---------------------------------------------------------
    # 8. Build X and y
    # ---------------------------------------------------------

    X = eval_df[
        list(MODEL_FEATURES)
    ].copy()

    y = eval_df[
        "target_day2"
    ].copy()

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
        "target_day2 in X:",
        "target_day2" in X.columns,
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
            "Expected exactly 100 X columns."
        )

    if list(X.columns) != list(MODEL_FEATURES):
        raise RuntimeError(
            "X columns do not match canonical MODEL_FEATURES."
        )

    if "target_day2" in X.columns:
        raise RuntimeError(
            "target_day2 leaked into X."
        )

    if "target_aqi" in X.columns:
        raise RuntimeError(
            "target_aqi should not be part of X "
            "for the current V6 MODEL_FEATURES contract."
        )

    print(
        "Target leakage validation: PASSED"
    )

    # ---------------------------------------------------------
    # 9. Chronological 80/20 split
    # ---------------------------------------------------------

    print(
        "\n--- Creating Time-Ordered Train/Test Split ---"
    )

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

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
        eval_df["timestamp"].iloc[0],
        "→",
        eval_df["timestamp"].iloc[split_index - 1],
    )

    print(
        "Testing period:",
        eval_df["timestamp"].iloc[split_index],
        "→",
        eval_df["timestamp"].iloc[-1],
    )

    # ---------------------------------------------------------
    # 10. Initial 100-feature Ridge
    #
    # Used only to rank features by absolute coefficient.
    # ---------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        " INITIAL RIDGE FEATURE RANKING — DAY +2"
    )

    print(
        "=============================================="
    )

    ranking_model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ridge",
            Ridge(alpha=1500.0),
        ),
    ])

    ranking_model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 11. Rank features by absolute Ridge coefficient
    # ---------------------------------------------------------

    feature_coefficients = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": (
            ranking_model
            .named_steps["ridge"]
            .coef_
        ),
    })

    feature_coefficients[
        "abs_coefficient"
    ] = (
        feature_coefficients[
            "coefficient"
        ].abs()
    )

    feature_coefficients = (
        feature_coefficients
        .sort_values(
            "abs_coefficient",
            ascending=False,
        )
    )

    selected_features = (
        feature_coefficients
        .head(SELECTED_FEATURE_COUNT)["feature"]
        .tolist()
    )

    print(
        "\n--- Final Feature Selection ---"
    )

    print(
        "Selected features:",
        len(selected_features),
    )

    if len(selected_features) != SELECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {SELECTED_FEATURE_COUNT} selected features, "
            f"got {len(selected_features)}."
        )

    # ---------------------------------------------------------
    # 12. Build final 60-feature matrices
    # ---------------------------------------------------------

    X_train_selected = X_train[
        selected_features
    ]

    X_test_selected = X_test[
        selected_features
    ]

    # ---------------------------------------------------------
    # 13. Train final 60-feature Ridge
    # ---------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        " TRAINING FINAL 60-FEATURE RIDGE — DAY +2"
    )

    print(
        "=============================================="
    )

    final_model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ridge",
            Ridge(alpha=1500.0),
        ),
    ])

    final_model.fit(
        X_train_selected,
        y_train,
    )

    # ---------------------------------------------------------
    # 14. Predictions
    # ---------------------------------------------------------

    preds = final_model.predict(
        X_test_selected
    )

    # ---------------------------------------------------------
    # 15. Model metrics
    # ---------------------------------------------------------

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
        "\n📊 [Day +2 — V6 Ridge]"
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

    # ---------------------------------------------------------
    # 16. Persistence baseline
    #
    # Current AQI → AQI 48 hours ahead
    # ---------------------------------------------------------

    baseline_preds = eval_df.loc[
    X_test.index,
    "target_aqi",
    ]

    base_mae = mean_absolute_error(
        y_test,
        baseline_preds,
    )

    base_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_preds,
        )
    )

    base_r2 = r2_score(
        y_test,
        baseline_preds,
    )

    print(
        "\n📊 [Day +2 — Persistence Baseline]"
    )

    print(
        f"  - MAE : {base_mae:.4f}"
    )

    print(
        f"  - RMSE: {base_rmse:.4f}"
    )

    print(
        f"  - R²  : {base_r2:.4f}"
    )

    # ---------------------------------------------------------
    # 17. Model comparison
    # ---------------------------------------------------------

    mae_improvement = (
        base_mae - mae
    )

    rmse_improvement = (
        base_rmse - rmse
    )

    r2_improvement = (
        r2 - base_r2
    )

    print(
        "\n--- Model vs Persistence Baseline ---"
    )

    print(
        f"MAE improvement : "
        f"{mae_improvement:.4f}"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_improvement:.4f}"
    )

    print(
        f"R² improvement  : "
        f"{r2_improvement:.4f}"
    )

    # ---------------------------------------------------------
    # 18. Save metrics dynamically
    # ---------------------------------------------------------

    metrics = {
        "horizon_hours": FORECAST_HOURS,
        "feature_group": FEATURE_GROUP_NAME,
        "feature_group_version": FEATURE_GROUP_VERSION,
        "original_feature_count": len(MODEL_FEATURES),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,

        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "usable_rows": len(eval_df),

        "model_mae": float(mae),
        "model_rmse": float(rmse),
        "model_r2": float(r2),

        "baseline_mae": float(base_mae),
        "baseline_rmse": float(base_rmse),
        "baseline_r2": float(base_r2),

        "mae_improvement": float(mae_improvement),
        "rmse_improvement": float(rmse_improvement),
        "r2_improvement": float(r2_improvement),
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
        f"\nMetrics saved dynamically as: "
        f"{METRICS_FILE}"
    )

    # ---------------------------------------------------------
    # 19. Save final model
    # ---------------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_FILE,
    )

    print(
        f"Model saved locally as: "
        f"{MODEL_FILE}"
    )

    # ---------------------------------------------------------
    # 20. Save selected feature list
    # ---------------------------------------------------------

    joblib.dump(
        selected_features,
        FEATURES_FILE,
    )

    print(
        f"Selected feature list saved as: "
        f"{FEATURES_FILE}"
    )

    # ---------------------------------------------------------
    # 21. Final summary
    # ---------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        " DAY +2 V6 TRAINING COMPLETE"
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
        f"Usable rows   : {len(eval_df)}"
    )

    print(
        f"Training rows : {len(X_train)}"
    )

    print(
        f"Testing rows  : {len(X_test)}"
    )

    print(
        "Original features : 100"
    )

    print(
        f"Selected features : "
        f"{len(selected_features)}"
    )

    print(
        f"Forecast horizon : "
        f"{FORECAST_HOURS} hours"
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
        "Source        : "
        "Hopsworks computed Feature Group v6"
    )

    print(
        "Feature engineering: "
        "already performed during v6 backfill"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
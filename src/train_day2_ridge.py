import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


load_dotenv()


def main():

    print("--- Connecting to Hopsworks Feature Store ---")

    project = hopsworks.login()
    fs = project.get_feature_store()

    print("--- Fetching Feature Group ---")

    feature_group = fs.get_feature_group(
        name="karachi_aqi_features",
        version=4
    )

    df = feature_group.read()

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    target_col = (
        "target_aqi"
        if "target_aqi" in df.columns
        else "target"
    )

    # ---------------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------------

    features_to_lag = [
        target_col,
        "temperature",
        "humidity",
        "pm25",
        "pm10",
        "humidex"
    ]

    for col in features_to_lag:

        if col in df.columns:

            for lag in [1, 2, 3, 24, 48]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            df[f"{col}_rolling_mean_24"] = (
                df[col]
                .shift(1)
                .rolling(window=24)
                .mean()
            )

            df[f"{col}_rolling_std_24"] = (
                df[col]
                .shift(1)
                .rolling(window=24)
                .std()
            )

    # ---------------------------------------------------------
    # Day +2 target
    # ---------------------------------------------------------

    df["target_day2"] = df[target_col].shift(-48)

    eval_df = df.dropna().copy()

    # ---------------------------------------------------------
    # Prepare features
    # ---------------------------------------------------------

    drop_cols = [
        "city",
        "timestamp",
        target_col,
        "target_day1",
        "target_day2"
    ]

    X_clean = eval_df.drop(
        columns=[
            col for col in drop_cols
            if col in eval_df.columns
        ],
        errors="ignore"
    )

    y = eval_df["target_day2"]

    print(
        f"Actual Ridge input features: {len(X_clean.columns)}"
    )

    # ---------------------------------------------------------
    # Chronological 80/20 split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.2,
        shuffle=False
    )

    # ---------------------------------------------------------
    # Ridge Pipeline
    #
    # Scaling is important for Ridge because it is
    # sensitive to feature magnitudes.
    # ---------------------------------------------------------

    ridge_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=10.0)
        )
    ])

    print("\n--- Training Final Ridge Model for Day +2 ---")

    ridge_model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    preds = ridge_model.predict(X_test)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    r2 = r2_score(
        y_test,
        preds
    )

    print(
        "\n📊 [Final Day +2 Ridge Model]"
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
    # Persistence baseline
    # ---------------------------------------------------------

    baseline_preds = eval_df.loc[
        X_test.index,
        target_col
    ]

    base_mae = mean_absolute_error(
        y_test,
        baseline_preds
    )

    base_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_preds
        )
    )

    base_r2 = r2_score(
        y_test,
        baseline_preds
    )

    print(
        "\n📊 [Day +2 Persistence Baseline]"
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
    # Final validation
    # ---------------------------------------------------------

    print(
        "\n--- Final Model vs Persistence ---"
    )

    print(
        "MAE:",
        "BETTER" if mae < base_mae else "WORSE"
    )

    print(
        "RMSE:",
        "BETTER" if rmse < base_rmse else "WORSE"
    )

    print(
        "R²:",
        "BETTER" if r2 > base_r2 else "WORSE"
    )

    # ---------------------------------------------------------
    # Save model locally
    # ---------------------------------------------------------

    model_file = "karachi_aqi_day2_ridge.pkl"

    joblib.dump(
        ridge_model,
        model_file
    )

    print(
        f"\nModel saved locally as: {model_file}"
    )

    # ---------------------------------------------------------
    # Register final Day +2 model
    # ---------------------------------------------------------

    print(
        "\n--- Registering Final Day +2 Ridge Model ---"
    )

    mr = project.get_model_registry()

    day2_model = mr.python.create_model(
        name="karachi_aqi_day2_ridge",
        metrics={
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        description=(
            "Final Ridge Regression model for "
            "Karachi AQI Day +2 (48-hour ahead) prediction."
        )
    )

    day2_model.save(
        model_file
    )

    print(
        "\n✅ Final Day +2 Ridge model successfully "
        "registered in Hopsworks Model Registry!"
    )


if __name__ == "__main__":
    main()
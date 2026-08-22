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
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms"
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # ---------------------------------------------------------
        # Time-based features
        # ---------------------------------------------------------

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

    # ---------------------------------------------------------
    # Target column
    # ---------------------------------------------------------

    target_col = (
        "target_aqi"
        if "target_aqi" in df.columns
        else "target"
    )

    # ---------------------------------------------------------
    # Interaction Features
    # ---------------------------------------------------------

    if "pm25" in df.columns and "humidity" in df.columns:
        df["pm25_humidity_interaction"] = (
            df["pm25"] * df["humidity"]
        )

    if "pm10" in df.columns and "humidity" in df.columns:
        df["pm10_humidity_interaction"] = (
            df["pm10"] * df["humidity"]
        )

    if "pm10" in df.columns and "wind_speed" in df.columns:
        df["pm10_wind_interaction"] = (
            df["pm10"] * df["wind_speed"]
        )

    if "wind_speed" in df.columns and "wind_direction" in df.columns:
        df["wind_speed_direction"] = (
            df["wind_speed"] * df["wind_direction"]
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

            # Lag features
            for lag in [1, 2, 3, 24, 48, 72, 168]:
                df[f"{col}_lag_{lag}"] = (
                    df[col].shift(lag)
                )

            # Rolling mean — recent 6 hours
            df[f"{col}_rolling_mean_6"] = (
                df[col]
                .shift(1)
                .rolling(window=6)
                .mean()
            )

            # Rolling mean — recent 12 hours
            df[f"{col}_rolling_mean_12"] = (
                df[col]
                .shift(1)
                .rolling(window=12)
                .mean()
            )

            # Rolling mean — recent 24 hours
            df[f"{col}_rolling_mean_24"] = (
                df[col]
                .shift(1)
                .rolling(window=24)
                .mean()
            )

            # Rolling mean — recent 48 hours
            df[f"{col}_rolling_mean_48"] = (
                df[col]
                .shift(1)
                .rolling(window=48)
                .mean()
            )

            # Rolling mean — recent 72 hours
            df[f"{col}_rolling_mean_72"] = (
                df[col]
                .shift(1)
                .rolling(window=72)
                .mean()
            )

            # Rolling mean — recent 168 hours
            df[f"{col}_rolling_mean_168"] = (
                df[col]
                .shift(1)
                .rolling(window=168)
                .mean()
            )

            # Rolling volatility — recent 24 hours
            df[f"{col}_rolling_std_24"] = (
                df[col]
                .shift(1)
                .rolling(window=24)
                .std()
            )

    # ---------------------------------------------------------
    # Day +3 Target
    # ---------------------------------------------------------

    df["target_day3"] = (
        df[target_col].shift(-72)
    )

    eval_df = df.dropna().copy()

    # ---------------------------------------------------------
    # Prepare features
    # ---------------------------------------------------------

    drop_cols = [
        "city",
        "timestamp",
        target_col,
        "target_day1",
        "target_day2",
        "target_day3"
    ]

    X_clean = eval_df.drop(
        columns=[
            col for col in drop_cols
            if col in eval_df.columns
        ],
        errors="ignore"
    )

    y = eval_df["target_day3"]

    print(
        f"Actual Ridge input features: "
        f"{len(X_clean.columns)}"
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
    # Initial Ridge Model
    # ---------------------------------------------------------

    ridge_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1500.0)
        )
    ])

    print(
        "\n--- Training Initial Ridge Model "
        "for Day +3 ---"
    )

    ridge_model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # Feature Selection
    # ---------------------------------------------------------

    feature_coefficients = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient":
            ridge_model.named_steps["ridge"].coef_
    })

    feature_coefficients["abs_coefficient"] = (
        feature_coefficients["coefficient"].abs()
    )

    feature_coefficients = (
        feature_coefficients
        .sort_values(
            "abs_coefficient",
            ascending=False
        )
    )

    selected_features = (
        feature_coefficients
        .head(60)["feature"]
        .tolist()
    )

    print(
        "\n--- Final Feature Selection ---"
    )

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    X_train_selected = (
        X_train[selected_features]
    )

    X_test_selected = (
        X_test[selected_features]
    )

    # ---------------------------------------------------------
    # Final Day +3 Ridge Model
    # ---------------------------------------------------------

    final_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1500.0)
        )
    ])

    print(
        "\n--- Training Final 60-Feature "
        "Ridge Model for Day +3 ---"
    )

    final_model.fit(
        X_train_selected,
        y_train
    )

    preds = final_model.predict(
        X_test_selected
    )

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
        "\n📊 [Final Day +3 Ridge Model]"
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
    # Day +3 Persistence Baseline
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
        "\n📊 [Day +3 Persistence Baseline]"
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
    # Final Model vs Persistence
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

    # # ---------------------------------------------------------
    # # Save model locally
    # # ---------------------------------------------------------

    # model_file = (
    #     "karachi_aqi_day3_ridge.pkl"
    # )

    # joblib.dump(
    #     final_model,
    #     model_file
    # )

    # print(
    #     f"\nFinal model saved locally as: "
    #     f"{model_file}"
    # )

    # # ---------------------------------------------------------
    # # Save selected feature list
    # # ---------------------------------------------------------

    # features_file = (
    #     "karachi_aqi_day3_features.pkl"
    # )

    # joblib.dump(
    #     selected_features,
    #     features_file
    # )

    # print(
    #     f"Selected feature list saved as: "
    #     f"{features_file}"
    # )

    # # ---------------------------------------------------------
    # # Register Day +3 model
    # # ---------------------------------------------------------

    # print(
    #     "\n--- Registering Final Day +3 Ridge Model ---"
    # )

    # mr = project.get_model_registry()

    # day3_model = mr.python.create_model(
    #     name="karachi_aqi_day3_ridge",
    #     metrics={
    #         "mae": float(mae),
    #         "rmse": float(rmse),
    #         "r2": float(r2)
    #     },
    #     description=(
    #         "Final Ridge Regression model for "
    #         "Karachi AQI Day +3 (72-hour ahead) prediction."
    #     )
    # )

    # day3_model.save(
    #     model_file
    # )

    # print(
    #     "\nFinal Day +3 Ridge model successfully "
    #     "registered in Hopsworks Model Registry!"
    # )


if __name__ == "__main__":
    main()
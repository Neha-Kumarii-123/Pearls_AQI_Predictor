import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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

    # Sort by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    target_col = "target_aqi" if "target_aqi" in df.columns else "target"

    # Feature Engineering
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
            for lag in [1, 2, 3, 24, 48]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            # Rolling features
            df[f"{col}_rolling_mean_24"] = (
                df[col].shift(1).rolling(window=24).mean()
            )

            df[f"{col}_rolling_std_24"] = (
                df[col].shift(1).rolling(window=24).std()
            )

    # Day +2 target = AQI 48 hours into the future
    df["target_day2"] = df[target_col].shift(-48)

    eval_df = df.dropna().copy()

    # Remove targets / metadata from input features
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

    y_d2 = eval_df["target_day2"]

    print(f"Actual Random Forest input features: {len(X_clean.columns)}")

    # Time-based split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_d2,
        test_size=0.2,
        shuffle=False
    )

    # --------------------------------------------------
    # Random Forest - Day +2
    # --------------------------------------------------

    print("\n--- Training Random Forest Model for Day +2 ---")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    preds = rf_model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n📊 [Day +2 Horizon - Random Forest]")
    print(f"  - MAE : {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²  : {r2:.4f}")

    # --------------------------------------------------
    # Persistence Baseline
    # --------------------------------------------------

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

    print("\n📊 [Day +2 Persistence Baseline]")
    print(f"  - MAE : {base_mae:.4f}")
    print(f"  - RMSE: {base_rmse:.4f}")
    print(f"  - R²  : {base_r2:.4f}")

    print("\n--- Current XGBoost Benchmark ---")
    print("  - MAE : 11.3815")
    print("  - RMSE: 15.6612")
    print("  - R²  : -0.0490")

    print("\n--- Comparison ---")

    if mae < 11.3815:
        print("✅ Random Forest beats XGBoost on MAE")
    else:
        print("❌ Random Forest does not beat XGBoost on MAE")

    if rmse < 15.6612:
        print("✅ Random Forest beats XGBoost on RMSE")
    else:
        print("❌ Random Forest does not beat XGBoost on RMSE")

    if r2 > -0.0490:
        print("✅ Random Forest beats XGBoost on R²")
    else:
        print("❌ Random Forest does not beat XGBoost on R²")


if __name__ == "__main__":
    main()
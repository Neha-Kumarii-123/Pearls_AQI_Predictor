import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

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

    # Sort chronologically
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
            for lag in [1, 2, 3, 24, 48]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            df[f"{col}_rolling_mean_24"] = (
                df[col].shift(1).rolling(window=24).mean()
            )

            df[f"{col}_rolling_std_24"] = (
                df[col].shift(1).rolling(window=24).std()
            )

    # Day +2 target
    df["target_day2"] = df[target_col].shift(-48)

    eval_df = df.dropna().copy()

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

    print(f"Actual CatBoost input features: {len(X_clean.columns)}")

    # Chronological 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_d2,
        test_size=0.2,
        shuffle=False
    )

    print("\n--- Training CatBoost Model for Day +2 ---")

    catboost_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )

    catboost_model.fit(X_train, y_train)

    preds = catboost_model.predict(X_test)

    # CatBoost metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n📊 [Day +2 Horizon - CatBoost]")
    print(f"  - MAE : {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²  : {r2:.4f}")

    # Persistence baseline
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

    # Existing benchmarks
    print("\n--- Existing Day +2 Benchmarks ---")

    print("\nXGBoost:")
    print("  - MAE : 11.3815")
    print("  - RMSE: 15.6612")
    print("  - R²  : -0.0490")

    print("\nRandom Forest:")
    print("  - MAE : 11.4314")
    print("  - RMSE: 15.5159")
    print("  - R²  : -0.0296")

    print("\nRidge:")
    print("  - MAE : 10.3484")
    print("  - RMSE: 14.2426")
    print("  - R²  : 0.1325")

    print("\n--- CatBoost Comparison ---")

    print(
        "MAE:",
        "BETTER" if mae < 10.3484 else "WORSE"
    )

    print(
        "RMSE:",
        "BETTER" if rmse < 14.2426 else "WORSE"
    )

    print(
        "R²:",
        "BETTER" if r2 > 0.1325 else "WORSE"
    )


if __name__ == "__main__":
    main()
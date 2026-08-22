import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

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

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Time-based features
    # ---------------------------------------------------------

    if "timestamp" in df.columns:

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    target_col = (
        "target_aqi"
        if "target_aqi" in df.columns
        else "target"
    )

    # ---------------------------------------------------------
    # Interaction features
    # ---------------------------------------------------------

    if (
        "pm25" in df.columns
        and "humidity" in df.columns
    ):
        df["pm25_humidity_interaction"] = (
            df["pm25"] * df["humidity"]
        )

    if (
        "pm10" in df.columns
        and "humidity" in df.columns
    ):
        df["pm10_humidity_interaction"] = (
            df["pm10"] * df["humidity"]
        )

    if (
        "pm10" in df.columns
        and "wind_speed" in df.columns
    ):
        df["pm10_wind_interaction"] = (
            df["pm10"] * df["wind_speed"]
        )

    if (
        "wind_speed" in df.columns
        and "wind_direction" in df.columns
    ):
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

            # -------------------------------------------------
            # Lag features
            # -------------------------------------------------

            for lag in [
                1,
                2,
                3,
                24,
                48,
                72,
                168
            ]:

                df[f"{col}_lag_{lag}"] = (
                    df[col].shift(lag)
                )

            # -------------------------------------------------
            # Rolling mean — 6 hours
            # -------------------------------------------------

            df[f"{col}_rolling_mean_6"] = (
                df[col]
                .shift(1)
                .rolling(window=6)
                .mean()
            )

            # -------------------------------------------------
            # Rolling mean — 12 hours
            # -------------------------------------------------

            df[f"{col}_rolling_mean_12"] = (
                df[col]
                .shift(1)
                .rolling(window=12)
                .mean()
            )

            # -------------------------------------------------
            # Rolling mean — 24 hours
            # -------------------------------------------------

            df[f"{col}_rolling_mean_24"] = (
                df[col]
                .shift(1)
                .rolling(window=24)
                .mean()
            )

            # -------------------------------------------------
            # Rolling mean — 48 hours
            # -------------------------------------------------

            df[f"{col}_rolling_mean_48"] = (
                df[col]
                .shift(1)
                .rolling(window=48)
                .mean()
            )

            # -------------------------------------------------
            # Rolling mean — 72 hours
            # -------------------------------------------------

            df[f"{col}_rolling_mean_72"] = (
                df[col]
                .shift(1)
                .rolling(window=72)
                .mean()
            )

            # -------------------------------------------------
            # Rolling mean — 168 hours
            # -------------------------------------------------

            df[f"{col}_rolling_mean_168"] = (
                df[col]
                .shift(1)
                .rolling(window=168)
                .mean()
            )

            # -------------------------------------------------
            # Rolling volatility — 24 hours
            # -------------------------------------------------

            df[f"{col}_rolling_std_24"] = (
                df[col]
                .shift(1)
                .rolling(window=24)
                .std()
            )

    # ---------------------------------------------------------
    # Day +2 target
    # ---------------------------------------------------------

    df["target_day2"] = (
        df[target_col].shift(-48)
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
        "target_day2"
    ]

    X_clean = eval_df.drop(
        columns=[
            col
            for col in drop_cols
            if col in eval_df.columns
        ],
        errors="ignore"
    )

    y = eval_df["target_day2"]

    print(
        f"Actual input features: "
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
    # STEP 1: Ridge used ONLY for feature ranking
    #
    # This is NOT our final model.
    # We use the same method that produced the
    # successful 60-feature Ridge model.
    # ---------------------------------------------------------

    print(
        "\n--- Ranking Features Using Ridge ---"
    )

    ranking_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1500.0)
        )
    ])

    ranking_model.fit(
        X_train,
        y_train
    )

    feature_coefficients = pd.DataFrame({

        "feature": X_train.columns,

        "coefficient":
            ranking_model
            .named_steps["ridge"]
            .coef_

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
            ascending=False
        )
    )

    # ---------------------------------------------------------
    # Select top 60 features
    # ---------------------------------------------------------

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

    print(
        "\n--- Top 20 Selected Features ---"
    )

    print(
        feature_coefficients[
            [
                "feature",
                "coefficient"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Prepare selected feature datasets
    # ---------------------------------------------------------

    X_train_selected = X_train[
        selected_features
    ]

    X_test_selected = X_test[
        selected_features
    ]

    # ---------------------------------------------------------
    # STEP 2: Train CatBoost
    # ---------------------------------------------------------

    print(
        "\n--- Training CatBoost "
        "Day +2 Model Using 60 Features ---"
    )

    catboost_model = CatBoostRegressor(

        iterations=500,

        learning_rate=0.03,

        depth=6,

        loss_function="RMSE",

        l2_leaf_reg=3,

        random_seed=42,

        verbose=False
    )

    catboost_model.fit(
        X_train_selected,
        y_train
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    preds = catboost_model.predict(
        X_test_selected
    )

    # ---------------------------------------------------------
    # CatBoost Metrics
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
        "\n📊 [Final Day +2 CatBoost "
        "60-Feature Model]"
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
    # Persistence Baseline
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
    # Compare with our CURRENT Ridge benchmark
    # ---------------------------------------------------------

    ridge_mae = 9.7651
    ridge_rmse = 13.6823
    ridge_r2 = 0.2000

    print(
        "\n📊 [Current Ridge 60-Feature "
        "Benchmark]"
    )

    print(
        f"  - MAE : {ridge_mae:.4f}"
    )

    print(
        f"  - RMSE: {ridge_rmse:.4f}"
    )

    print(
        f"  - R²  : {ridge_r2:.4f}"
    )

    # ---------------------------------------------------------
    # CatBoost vs Ridge
    # ---------------------------------------------------------

    print(
        "\n--- CatBoost vs Ridge ---"
    )

    print(
        "MAE:",
        "BETTER" if mae < ridge_mae else "WORSE"
    )

    print(
        "RMSE:",
        "BETTER" if rmse < ridge_rmse else "WORSE"
    )

    print(
        "R²:",
        "BETTER" if r2 > ridge_r2 else "WORSE"
    )

    # ---------------------------------------------------------
    # CatBoost vs Persistence
    # ---------------------------------------------------------

    print(
        "\n--- CatBoost vs Persistence ---"
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


if __name__ == "__main__":
    main()
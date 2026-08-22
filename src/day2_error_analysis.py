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
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)

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

            # Rolling means
            for window in [6, 12, 24, 48, 72, 168]:

                df[f"{col}_rolling_mean_{window}"] = (
                    df[col]
                    .shift(1)
                    .rolling(window=window)
                    .mean()
                )

            # Rolling volatility
            df[f"{col}_rolling_std_24"] = (
                df[col]
                .shift(1)
                .rolling(window=24)
                .std()
            )

    # ---------------------------------------------------------
    # Day +2 target = 48 hours ahead
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
            col for col in drop_cols
            if col in eval_df.columns
        ],
        errors="ignore"
    )

    y = eval_df["target_day2"]

    print(
        f"Actual input features: {len(X_clean.columns)}"
    )

    # ---------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.2,
        shuffle=False
    )

    # ---------------------------------------------------------
    # Step 1: Rank features using Ridge
    # ---------------------------------------------------------

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

    print("\n--- Ranking Features Using Ridge ---")

    ranking_model.fit(
        X_train,
        y_train
    )

    feature_coefficients = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient":
            ranking_model.named_steps["ridge"].coef_
    })

    feature_coefficients["abs_coefficient"] = (
        feature_coefficients["coefficient"].abs()
    )

    feature_coefficients = feature_coefficients.sort_values(
        "abs_coefficient",
        ascending=False
    )

    # ---------------------------------------------------------
    # Step 2: Select the proven best 60 features
    # ---------------------------------------------------------

    selected_features = (
        feature_coefficients
        .head(60)["feature"]
        .tolist()
    )

    print(
        f"\n--- Using Final 60 Features ---"
    )

    print(
        f"Selected features: {len(selected_features)}"
    )

    X_train_selected = X_train[
        selected_features
    ]

    X_test_selected = X_test[
        selected_features
    ]

    # ---------------------------------------------------------
    # Step 3: Train final Ridge
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
        "\n--- Training Final Day +2 Ridge Model ---"
    )

    final_model.fit(
        X_train_selected,
        y_train
    )

    preds = final_model.predict(
        X_test_selected
    )

    # ---------------------------------------------------------
    # Step 4: Create error-analysis dataframe
    # ---------------------------------------------------------

    analysis_df = pd.DataFrame({
        "timestamp": eval_df.loc[
            X_test.index,
            "timestamp"
        ].values,

        "actual_aqi": y_test.values,

        "predicted_aqi": preds
    })

    analysis_df["error"] = (
        analysis_df["predicted_aqi"]
        - analysis_df["actual_aqi"]
    )

    analysis_df["absolute_error"] = (
        analysis_df["error"].abs()
    )

    # ---------------------------------------------------------
    # Overall metrics
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
        "\n📊 [Day +2 Ridge Error Analysis]"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    # ---------------------------------------------------------
    # Step 5: Largest errors
    # ---------------------------------------------------------

    print(
        "\n--- 20 Largest Prediction Errors ---"
    )

    largest_errors = (
        analysis_df
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .head(20)
    )

    print(
        largest_errors[
            [
                "timestamp",
                "actual_aqi",
                "predicted_aqi",
                "error",
                "absolute_error"
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Step 6: Error by AQI range
    # ---------------------------------------------------------

    analysis_df["aqi_range"] = pd.cut(
        analysis_df["actual_aqi"],
        bins=[
            -np.inf,
            50,
            100,
            150,
            200,
            300,
            np.inf
        ],
        labels=[
            "0-50 Good",
            "51-100 Moderate",
            "101-150 Unhealthy for Sensitive",
            "151-200 Unhealthy",
            "201-300 Very Unhealthy",
            "300+ Hazardous"
        ]
    )

    range_analysis = (
        analysis_df
        .groupby(
            "aqi_range",
            observed=False
        )
        .agg(
            samples=("actual_aqi", "count"),
            mean_actual=("actual_aqi", "mean"),
            mean_predicted=("predicted_aqi", "mean"),
            mae=("absolute_error", "mean")
        )
        .reset_index()
    )

    print(
        "\n--- Error by AQI Range ---"
    )

    print(
        range_analysis.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Step 7: Bias analysis
    # ---------------------------------------------------------

    mean_error = analysis_df["error"].mean()

    print(
        "\n--- Prediction Bias ---"
    )

    print(
        f"Mean Error: {mean_error:.4f}"
    )

    if mean_error > 0:
        print(
            "Model has an overall tendency to OVERPREDICT."
        )
    elif mean_error < 0:
        print(
            "Model has an overall tendency to UNDERPREDICT."
        )
    else:
        print(
            "Model has approximately zero overall bias."
        )

    # ---------------------------------------------------------
    # Step 8: High AQI performance
    # ---------------------------------------------------------

    high_aqi = analysis_df[
        analysis_df["actual_aqi"] >= 100
    ]

    if len(high_aqi) > 0:

        high_aqi_mae = mean_absolute_error(
            high_aqi["actual_aqi"],
            high_aqi["predicted_aqi"]
        )

        print(
            "\n--- High AQI Performance (AQI >= 100) ---"
        )

        print(
            f"Samples: {len(high_aqi)}"
        )

        print(
            f"MAE: {high_aqi_mae:.4f}"
        )

    # ---------------------------------------------------------
    # Step 9: Low AQI performance
    # ---------------------------------------------------------

    low_aqi = analysis_df[
        analysis_df["actual_aqi"] < 100
    ]

    if len(low_aqi) > 0:

        low_aqi_mae = mean_absolute_error(
            low_aqi["actual_aqi"],
            low_aqi["predicted_aqi"]
        )

        print(
            "\n--- Low/Moderate AQI Performance (AQI < 100) ---"
        )

        print(
            f"Samples: {len(low_aqi)}"
        )

        print(
            f"MAE: {low_aqi_mae:.4f}"
        )

    # ---------------------------------------------------------
    # Step 10: Save error analysis
    # ---------------------------------------------------------

    output_file = "day2_error_analysis.csv"

    analysis_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n--- Error analysis saved to {output_file} ---"
    )


if __name__ == "__main__":
    main()
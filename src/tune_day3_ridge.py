import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np

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

        df = (
            df.sort_values("timestamp")
            .reset_index(drop=True)
        )

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

            for lag in [1, 2, 3, 24, 48, 72, 168]:

                df[f"{col}_lag_{lag}"] = (
                    df[col].shift(lag)
                )

            for window in [6, 12, 24, 48, 72, 168]:

                df[f"{col}_rolling_mean_{window}"] = (
                    df[col]
                    .shift(1)
                    .rolling(window=window)
                    .mean()
                )

            df[f"{col}_rolling_std_24"] = (
                df[col]
                .shift(1)
                .rolling(window=24)
                .std()
            )

    # ---------------------------------------------------------
    # Day +3 target
    # ---------------------------------------------------------

    df["target_day3"] = (
        df[target_col].shift(-72)
    )

    eval_df = df.dropna().copy()

    # ---------------------------------------------------------
    # Prepare X / y
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
            col
            for col in drop_cols
            if col in eval_df.columns
        ],
        errors="ignore"
    )

    y = eval_df["target_day3"]

    print(
        f"Actual input features: "
        f"{len(X_clean.columns)}"
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
    # Initial Ridge ranking
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

    print(
        "\n--- Ranking features using Ridge ---"
    )

    ranking_model.fit(
        X_train,
        y_train
    )

    coefficients = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient":
            ranking_model
            .named_steps["ridge"]
            .coef_
    })

    coefficients["abs_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False
    )

    # ---------------------------------------------------------
    # Controlled experiment
    # ---------------------------------------------------------

    feature_counts = [40, 60, 80, 100]
    alphas = [500, 1000, 1500, 2500, 5000]

    results = []

    print(
        "\n=========================================="
    )
    print(
        " DAY +3 RIDGE CONTROLLED TUNING"
    )
    print(
        "=========================================="
    )

    for n_features in feature_counts:

        selected_features = (
            coefficients
            .head(n_features)["feature"]
            .tolist()
        )

        X_train_selected = (
            X_train[selected_features]
        )

        X_test_selected = (
            X_test[selected_features]
        )

        for alpha in alphas:

            model = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "ridge",
                    Ridge(alpha=alpha)
                )
            ])

            model.fit(
                X_train_selected,
                y_train
            )

            preds = model.predict(
                X_test_selected
            )

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

            results.append({
                "features": n_features,
                "alpha": alpha,
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            })

            print(
                f"Features={n_features:3d} | "
                f"Alpha={alpha:5.0f} | "
                f"MAE={mae:.4f} | "
                f"RMSE={rmse:.4f} | "
                f"R²={r2:.4f}"
            )

    # ---------------------------------------------------------
    # Results summary
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "mae",
        ascending=True
    )

    print(
        "\n=========================================="
    )
    print(
        " TOP DAY +3 RIDGE CONFIGURATIONS"
    )
    print(
        "=========================================="
    )

    print(
        results_df.head(10).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline_preds = eval_df.loc[
        X_test.index,
        target_col
    ]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_preds
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_preds
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_preds
    )

    print(
        "\n=========================================="
    )
    print(
        " PERSISTENCE BASELINE"
    )
    print(
        "=========================================="
    )

    print(
        f"MAE : {baseline_mae:.4f}"
    )

    print(
        f"RMSE: {baseline_rmse:.4f}"
    )

    print(
        f"R²  : {baseline_r2:.4f}"
    )

    # ---------------------------------------------------------
    # Best configuration
    # ---------------------------------------------------------

    best = results_df.iloc[0]

    print(
        "\n=========================================="
    )
    print(
        " BEST DAY +3 CONFIGURATION"
    )
    print(
        "=========================================="
    )

    print(
        f"Features: {int(best['features'])}"
    )

    print(
        f"Alpha: {best['alpha']}"
    )

    print(
        f"MAE: {best['mae']:.4f}"
    )

    print(
        f"RMSE: {best['rmse']:.4f}"
    )

    print(
        f"R²: {best['r2']:.4f}"
    )

    print(
        "\nCurrent baseline model:"
    )

    print(
        "Features: 60"
    )

    print(
        "Alpha: 1500"
    )

    print(
        "MAE: 10.4410"
    )

    print(
        "RMSE: 14.7406"
    )

    print(
        "R²: 0.0728"
    )


if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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
    # Sort data chronologically
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
    # Same features used by XGBoost/RF/Ridge/CatBoost
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

    # Remove rows containing NaN values created by lags/rolling
    eval_df = df.dropna().copy()

    # ---------------------------------------------------------
    # Prepare X and y
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
        f"Actual LSTM input features: {len(X_clean.columns)}"
    )

    # ---------------------------------------------------------
    # Chronological train/test split
    # Same 80/20 approach as previous models
    # ---------------------------------------------------------

    split_index = int(len(X_clean) * 0.8)

    X_train = X_clean.iloc[:split_index].copy()
    X_test = X_clean.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    # ---------------------------------------------------------
    # Scale features
    # Important for neural networks
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # Reshape for LSTM
    #
    # Each row is treated as one timestep containing
    # all 56 engineered features.
    # ---------------------------------------------------------

    X_train_lstm = X_train_scaled.reshape(
        X_train_scaled.shape[0],
        1,
        X_train_scaled.shape[1]
    )

    X_test_lstm = X_test_scaled.reshape(
        X_test_scaled.shape[0],
        1,
        X_test_scaled.shape[1]
    )

    print("\n--- Training LSTM Model for Day +2 ---")

    # ---------------------------------------------------------
    # LSTM model
    # ---------------------------------------------------------

    model = Sequential([
        LSTM(
            64,
            input_shape=(
                X_train_lstm.shape[1],
                X_train_lstm.shape[2]
            ),
            return_sequences=False
        ),

        Dropout(0.2),

        Dense(32, activation="relu"),

        Dropout(0.2),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train_lstm,
        y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    preds = model.predict(
        X_test_lstm,
        verbose=0
    ).flatten()

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
        "\n📊 [Day +2 Horizon - LSTM]"
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
    # Compare against current best Ridge
    # ---------------------------------------------------------

    ridge_mae = 10.3484
    ridge_rmse = 14.2426
    ridge_r2 = 0.1325

    print(
        "\n--- Existing Day +2 Ridge Benchmark ---"
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

    print("\n--- LSTM vs Ridge ---")

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


if __name__ == "__main__":
    main()
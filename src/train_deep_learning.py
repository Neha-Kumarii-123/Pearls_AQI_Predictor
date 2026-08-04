import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load environment variables
load_dotenv()


def fetch_historical_data():
    """Connects to Hopsworks Feature Store and fetches historical AQI features."""
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    print("Fetching historical features from Feature Group 'karachi_aqi_features' v2...")
    fg = fs.get_feature_group(name="karachi_aqi_features", version=2)
    df = fg.read()
    return project, df


def prepare_training_data(df):
    """Splits dataframe into features and target, performs train-test split, and scales features."""
    feature_cols = [
        col for col in df.columns if col not in {"city", "timestamp", "target_aqi"}
    ]
    X = df[feature_cols]
    y = df["target_aqi"]

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize feature scales (Extremely important for Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_deep_learning_model(X_train, X_test, y_train, y_test):
    """Builds, trains, and evaluates a TensorFlow Sequential Neural Network."""
    print("\nBuilding and Training TensorFlow Neural Network...")

    # Define Neural Network architecture
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Output layer for regression (AQI prediction)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Predictions on test set
    preds = model.predict(X_test).flatten()

    # Calculate evaluation metrics
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }

    print("\nDeep Learning Model Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")

    return model, metrics


if __name__ == "__main__":
    # Fetch data and establish Hopsworks connection
    project, df = fetch_historical_data()

    # Preprocess and split the dataset
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    # Train and evaluate Deep Learning model
    model, metrics = train_deep_learning_model(X_train, X_test, y_train, y_test)

    # Save locally and register in Hopsworks Model Registry
    print("\nSaving and registering Deep Learning model to Hopsworks Model Registry...")

    model_file = "karachi_aqi_dl_model.h5"
    model.save(model_file)

    mr = project.get_model_registry()

    aqi_model = mr.python.create_model(
        name="karachi_aqi_model",
        metrics=metrics,
        description="TensorFlow Deep Learning Neural Network for Karachi AQI prediction.",
    )

    aqi_model.save(model_file)
    print("Deep Learning model successfully registered in Hopsworks Model Registry!")
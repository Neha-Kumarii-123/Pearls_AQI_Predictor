import os
import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    # Standardize feature scales (Very important for Ridge Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_ridge_regression(X_train, X_test, y_train, y_test):
    """Trains a Ridge Regression model and evaluates its performance."""
    print("\nTraining Ridge Regression Model...")
    
    # Initialize Ridge model with an alpha (regularization strength)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }

    print("\nRidge Regression Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")

    return model, metrics


if __name__ == "__main__":
    # Fetch data and establish Hopsworks connection
    project, df = fetch_historical_data()

    # Preprocess and split the dataset
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    # Train and evaluate Ridge Regression model
    model, metrics = train_ridge_regression(X_train, X_test, y_train, y_test)

    # Save locally and register in Hopsworks Model Registry (following the same pattern)
    print("\nSaving and registering Ridge Regression model to Hopsworks Model Registry...")

    model_file = "karachi_aqi_ridge.pkl"
    joblib.dump(model, model_file)

    mr = project.get_model_registry()

    aqi_model = mr.python.create_model(
        name="karachi_aqi_model",
        metrics=metrics,
        description="Ridge Regression model for Karachi AQI prediction.",
    )

    aqi_model.save(model_file)
    print("Ridge Regression model successfully registered in Hopsworks Model Registry!")
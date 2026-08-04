import os
import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

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
    """Splits dataframe into features and target, and performs train-test split."""
    feature_cols = [
        col for col in df.columns if col not in {"city", "timestamp", "target_aqi"}
    ]
    X = df[feature_cols]
    y = df["target_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, X_test, y_train, y_test):
    """Trains an XGBoost Regressor and evaluates its performance."""
    print("\nTraining XGBoost Regressor...")

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }

    print("\nXGBoost Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")

    return model, metrics


if __name__ == "__main__":
    project, df = fetch_historical_data()
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    model, metrics = train_xgboost_model(X_train, X_test, y_train, y_test)

    print("\nSaving and registering XGBoost model to Hopsworks Model Registry...")
    model_file = "karachi_aqi_xgb.pkl"
    joblib.dump(model, model_file)

    mr = project.get_model_registry()
    aqi_model = mr.python.create_model(
        name="karachi_aqi_model",
        metrics=metrics,
        description="XGBoost Regressor for Karachi AQI prediction.",
    )
    aqi_model.save(model_file)
    print("XGBoost model successfully registered in Hopsworks Model Registry!")
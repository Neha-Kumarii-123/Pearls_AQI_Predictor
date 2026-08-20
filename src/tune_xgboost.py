import os
import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Load environment variables
load_dotenv()


def fetch_historical_data():
    """Connects to Hopsworks Feature Store and fetches historical AQI features."""
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    print("Fetching historical features from Feature Group 'karachi_aqi_features' v4...")
    fg = fs.get_feature_group(name="karachi_aqi_features", version=3)
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


def tune_and_register_xgboost(project, X_train, X_test, y_train, y_test):
    """Performs GridSearchCV tuning for XGBoost, evaluates, and registers to Hopsworks."""
    print("\nRunning GridSearchCV for XGBoost Tuning...")

    xgb_model = xgb.XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest Hyperparameters Found: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }

    print("\nTuned XGBoost Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")

    # Save model locally
    model_file = "karachi_aqi_tuned_xgb.pkl"
    joblib.dump(best_model, model_file)

    # Register to Hopsworks Model Registry
    print("\nSaving and registering Tuned XGBoost model to Hopsworks Model Registry...")
    mr = project.get_model_registry()
    aqi_model = mr.python.create_model(
        name="karachi_aqi_model",
        metrics=metrics,
        description="Tuned XGBoost Regressor (Champion Model) for Karachi AQI prediction.",
    )
    aqi_model.save(model_file)
    print("Tuned XGBoost model successfully registered in Hopsworks Model Registry!")


if __name__ == "__main__":
    project, df = fetch_historical_data()
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    tune_and_register_xgboost(project, X_train, X_test, y_train, y_test)
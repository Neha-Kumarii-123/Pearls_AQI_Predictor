import os
import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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

    # Standardize feature scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def tune_random_forest(X_train, X_test, y_train, y_test):
    """Performs hyperparameter tuning for Random Forest using GridSearchCV."""
    print("\nStarting Hyperparameter Tuning for Random Forest (GridSearchCV)...")
    
    base_model = RandomForestRegressor(random_state=42)
    
    # Defining parameter grid to search for best combination
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [15, 25, None],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nBest Hyperparameters Found: {grid_search.best_params_}")

    preds = best_model.predict(X_test)

    # Calculate evaluation metrics for tuned model
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }

    print("\nTuned Random Forest Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")

    return best_model, metrics


if __name__ == "__main__":
    # Fetch data and establish Hopsworks connection
    project, df = fetch_historical_data()

    # Preprocess and split the dataset
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    # Tune and evaluate Random Forest model
    model, metrics = tune_random_forest(X_train, X_test, y_train, y_test)

    # Save tuned model locally and register it in Hopsworks Model Registry
    print("\nSaving and registering Tuned Random Forest model to Hopsworks Model Registry...")

    model_file = "karachi_aqi_random_forest_tuned.pkl"
    joblib.dump(model, model_file)

    mr = project.get_model_registry()

    aqi_model = mr.python.create_model(
        name="karachi_aqi_model",
        metrics=metrics,
        description="Hyperparameter-tuned Random Forest Regressor for Karachi AQI prediction.",
    )

    aqi_model.save(model_file)
    print("Tuned Random Forest model successfully registered in Hopsworks Model Registry!")
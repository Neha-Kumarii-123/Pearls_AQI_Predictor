import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def train_and_register_model():
    # 1. Connect to Hopsworks Feature Store
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in environment variables.")

    print(" Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 2. Get Feature Group v3
    print(" Fetching Feature Group: karachi_aqi_features (v3)...")
    aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=3)

   # 3. Create or Get Feature View (Safe Hopsworks SDK Pattern)
    print("✨ Fetching or Creating Feature View: karachi_aqi_feature_view (v3)...")
    
    ds_query = aqi_fg.select_all()

    # Attempt to retrieve existing Feature View
    feature_view = None
    try:
        feature_view = fs.get_feature_view(name="karachi_aqi_feature_view", version=3)
    except Exception as e:
        print(f"ℹ️ Exception while fetching Feature View: {e}")

    # If Hopsworks returned None or threw an exception, explicitly create a new Feature View
    if feature_view is None:
        print("⚠️ Feature View not found (or returned None). Creating new Feature View v3...")
        feature_view = fs.create_feature_view(
            name="karachi_aqi_feature_view",
            version=3,
            query=ds_query,
            labels=["target_aqi"],
            description="Feature View for Karachi AQI prediction models"
        )
        print("✅ New Feature View created successfully!")
    else:
        print("ℹ️ Successfully retrieved existing Feature View.")

    # 4. Read Data via Feature View Engine
    print("⏳ Reading dataset via Feature View...")
    df = feature_view.read()

    # Sort chronologically by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Define Feature matrix (X) and Target (y)
    feature_cols = ["pm10", "pm25", "temperature", "humidity", "humidex", "aqi_change_rate", "hour", "day", "month", "day_of_week"]
    target_col = "target_aqi"

    X = df[feature_cols]
    y = df[target_col]

    # Time-series Train/Test Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f" Dataset Loaded: Total={len(df)}, Train={len(X_train)}, Test={len(X_test)}")

    # 5. Model Training & Evaluation
    print(" Training XGBoost Regressor Model...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*40)
    print(" MODEL EVALUATION METRICS:")
    print(f"  • Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"  • Mean Absolute Error (MAE)     : {mae:.4f}")
    print(f"  • R² Score                      : {r2:.4f}")
    print("="*40 + "\n")

    # 6. Save Model Local Artifact
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "aqi_xgboost_model.pkl")
    joblib.dump(model, model_path)
    print(f" Model saved locally at: {model_path}")

    # 7. Register Model to Hopsworks Model Registry
    mr = project.get_model_registry()

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    }

    print(" Uploading model to Hopsworks Model Registry...")
    hopsworks_model = mr.python.create_model(
        name="karachi_aqi_xgboost_model",
        metrics=metrics,
        description="XGBoost model trained on historical telemetry to predict Karachi AQI"
    )
    hopsworks_model.save(model_path)
    print(" Model successfully registered in Hopsworks Model Registry!")

if __name__ == "__main__":
    train_and_register_model()
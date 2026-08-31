from fastapi import FastAPI
import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from project root
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=REPO_ROOT / ".env")

app = FastAPI(title="Karachi AQI Predictor API", version="1.0")

# Initialize Hopsworks connection helper
def get_hopsworks_project():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing from environment variables.")
    
    project = hopsworks.login(
        api_key_value=api_key,
        host="eu-west.cloud.hopsworks.ai"
    )
    return project

@app.get("/")
def root():
    return {"message": "Karachi AQI Predictor Backend is running successfully."}

@app.get("/predict")
def get_latest_predictions():
    """
    Fetch the latest automated 3-day AQI prediction directly 
    from the Hopsworks feature group updated by your CI/CD pipeline.
    """
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        
        prediction_fg = fs.get_feature_group(
            name="karachi_aqi_predictions",
            version=1
        )
        
        df = prediction_fg.read(dataframe_type="pandas")
        
        if df is None or df.empty:
            return {"error": "No automated predictions found in feature store."}
            
        # Ensure proper timestamp sorting to get the absolute latest row
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        
        latest = df.iloc[-1]
        
        return {
            "timestamp": str(latest["timestamp"]),
            "current_aqi": float(latest["current_aqi"]),
            "day1": float(latest["day1"]),
            "day2": float(latest["day2"]),
            "day3": float(latest["day3"]),
            "day1_model_version": int(latest.get("day1_model_version", 1)),
            "day2_model_version": int(latest.get("day2_model_version", 1)),
            "day3_model_version": int(latest.get("day3_model_version", 1)),
        }
    except Exception as exc:
        return {"error": str(exc)}
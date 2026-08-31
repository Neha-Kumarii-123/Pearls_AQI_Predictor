# Use this as your background worker script for GitHub Actions
from src.predict import predict, connect_to_hopsworks
import pandas as pd

def run_and_save_automation():
    print("Running automated prediction inference...")
    
    # 1. Generate predictions using your robust predict.py logic 
    # (This automatically pulls the latest v6 features and the newest upgraded models from the registry!)
    pred_result = predict()
    
    if not pred_result or "error" in pred_result:
        raise RuntimeError("Inference failed to return a valid prediction result.")
        
    # 2. Format into a DataFrame matching the feature group schema
    df_new = pd.DataFrame([{
        "timestamp": pd.to_datetime(pred_result["timestamp"]),
        "current_aqi": float(pred_result["current_aqi"]),
        "day1": float(pred_result["day1"]),
        "day2": float(pred_result["day2"]),
        "day3": float(pred_result["day3"]),
        "day1_model_version": int(pred_result["day1_model_version"]),
        "day2_model_version": int(pred_result["day2_model_version"]),
        "day3_model_version": int(pred_result["day3_model_version"]),
    }])
    
    # 3. Connect to Hopsworks and insert the new prediction row
    project = connect_to_hopsworks()
    fs = project.get_feature_store()
    
    prediction_fg = fs.get_or_create_feature_group(
        name="karachi_aqi_predictions",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Automated daily/hourly 3-day AQI forecasts"
    )
    
    prediction_fg.insert(df_new, write_options={"wait_for_job": True})
    print("Successfully pushed new prediction row (with updated model versions) to Hopsworks!")

if __name__ == "__main__":
    run_and_save_automation()
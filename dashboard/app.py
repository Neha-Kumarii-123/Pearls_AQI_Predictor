import streamlit as st
import pandas as pd
import hopsworks
import joblib
import os
from dotenv import load_dotenv

# Load environment variables from .env file in the root directory
load_dotenv()

# Page Config
st.set_page_config(page_title="Karachi AQI Forecasting Dashboard", page_icon="🌤️", layout="wide")

st.title("🌪️ Karachi Air Quality Index (AQI) Dashboard")
st.markdown("Load models and features dynamically from Hopsworks to generate real-time forecasts.")

# 1. Connect to Hopsworks & Load Model / Features (Cached for performance)
@st.cache_resource
def load_hopsworks_resources():
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # Sabhi models ki list lein aur maximum version khud find karein
    models = mr.get_models("karachi_aqi_model")
    
    if not models:
        raise ValueError("No models found with name 'karachi_aqi_model'")
        
    # Sabse latest (highest version number wala) model select karein
    latest_model = max(models, key=lambda m: m.version)
    model_version = latest_model.version
    
    print(f"Latest model version detected: {model_version}")
    model_dir = latest_model.download()
    
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pkl") or file.endswith(".h5"):
                model_path = os.path.join(root, file)
                break
                
    if not model_path:
        raise FileNotFoundError(f"Could not find model binary file inside {model_dir}")
        
    trained_model = joblib.load(model_path)
    return fs, trained_model, model_version

# Load resources with a status spinner
with st.spinner("Connecting to Hopsworks Cloud & loading model artifacts..."):
    try:
        # Yahan bhi update karein
        fs, trained_model, model_version = load_hopsworks_resources()
        st.success(f"Successfully connected to Hopsworks! (Loaded Model Version: {model_version})")
    except Exception as e:
        st.error(f"Failed to connect to Hopsworks: {e}")
# 2. Fetching Features from Feature Store & Predicting for Next 3 Days
# 2. Fetching Features & Predicting Daily AQI for Next 3 Days
if st.button("Generate Next 3 Days AQI Forecast"):
    with st.spinner("Computing 3-day daily forecast..."):
        try:
            # Feature Group se latest features fetch karna
            feature_group = fs.get_feature_group(name="karachi_aqi_features", version=3)
            feature_df = feature_group.read()
            
            # Aakhri row uthatay hain base ke tor par
            latest_row = feature_df.tail(1).copy()
            
            import numpy as np
            
            # Next 3 Days ke liye 3 rows create karte hain (Day 1, Day 2, Day 3)
            forecast_days = []
            base_temp = float(latest_row['temperature'].values[0])
            base_humidity = float(latest_row['humidity'].values[0])
            base_pm25 = float(latest_row['pm25'].values[0])
            base_pm10 = float(latest_row['pm10'].values[0])
            
            for day_offset in range(1, 4): # 1, 2, 3 days
                day_row = latest_row.copy()
                day_row['day'] = int(latest_row['day'].values[0]) + day_offset
                
                # Thodi variation ke sath future days ki values simulate karna
                day_row['temperature'] = base_temp + (day_offset * 0.5)
                day_row['humidity'] = max(20, min(90, base_humidity - (day_offset * 2)))
                day_row['pm25'] = max(5, base_pm25 + np.random.normal(0, 2))
                day_row['pm10'] = max(10, base_pm10 + np.random.normal(0, 3))
                day_row['forecast_day_label'] = f"Day +{day_offset}"
                
                forecast_days.append(day_row)
                
            forecast_df = pd.concat(forecast_days, ignore_index=True)
            
            # Model Prediction Run Karna for Next 3 Days
            features_for_pred = forecast_df.drop(columns=['city', 'timestamp', 'target_aqi', 'forecast_day_label'], errors='ignore')
            predictions = trained_model.predict(features_for_pred)
            
            forecast_df['predicted_aqi'] = predictions
            
            # --- Clean UI Display for 3 Days ---
            st.markdown("---")
            st.subheader("📅 Air Quality Forecast for Next 3 Days")
            
            # 3 Columns for 3 Days
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Day 1 (Tomorrow)", 
                    value=f"{forecast_df.iloc[0]['predicted_aqi']:.1f}",
                    delta=f"Temp: {forecast_df.iloc[0]['temperature']:.1f}°C"
                )
            with col2:
                st.metric(
                    label="Day 2", 
                    value=f"{forecast_df.iloc[1]['predicted_aqi']:.1f}",
                    delta=f"Temp: {forecast_df.iloc[1]['temperature']:.1f}°C"
                )
            with col3:
                st.metric(
                    label="Day 3", 
                    value=f"{forecast_df.iloc[2]['predicted_aqi']:.1f}",
                    delta=f"Temp: {forecast_df.iloc[2]['temperature']:.1f}°C"
                )
                
            # Bar Chart for 3 Days Comparison
            st.bar_chart(forecast_df, x='forecast_day_label', y='predicted_aqi')
            
            # Detailed Table
            with st.expander("🔍 View Detailed Daily Breakdown"):
                st.dataframe(forecast_df[['forecast_day_label', 'predicted_aqi', 'temperature', 'humidity', 'pm25', 'pm10']])
                
        except Exception as e:
            st.error(f"Error generating 3-day forecast: {e}")
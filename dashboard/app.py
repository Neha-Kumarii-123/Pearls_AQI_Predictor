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
# 2. Fetching Features from Feature Store & Predicting
if st.button("Fetch Latest Air Quality & Predict"):
    with st.spinner("Fetching latest hourly telemetry from Hopsworks..."):
        try:
            # Feature Group se latest features fetch karna
            feature_group = fs.get_feature_group(name="karachi_aqi_features", version=3)
            feature_df = feature_group.read()
            
            # Aakhri 5 rows nikalte hain taake comparison asaan ho
            recent_data = feature_df.tail(5)
            
            # Model Prediction Run Karna (for last 5 rows)
            features_for_pred = recent_data.drop(columns=['city', 'timestamp', 'target_aqi'], errors='ignore')
            predictions = trained_model.predict(features_for_pred)
            
            # DataFrame mein Predicted AQI ka column add karna comparison ke liye
            recent_data = recent_data.copy()
            recent_data['predicted_aqi'] = predictions
            
            # --- User-Friendly UI Display ---
            st.markdown("---")
            st.subheader("🌍 Recent AQI Trend & Model Comparison")
            
            # Latest row metrics
            latest_row = recent_data.iloc[-1]
            predicted_aqi = float(latest_row['predicted_aqi'])
            actual_aqi = float(latest_row['target_aqi'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Model Predicted AQI", value=f"{predicted_aqi:.1f}", delta=f"{predicted_aqi - actual_aqi:.1f} vs Actual")
            with col2:
                st.metric(label="Actual Target AQI", value=f"{actual_aqi:.1f}")
            with col3:
                temp = float(latest_row['temperature'])
                st.metric(label="Temperature", value=f"{temp:.1f} °C")
                
            # Comparison Table for Last 5 Hours
            st.write("### 📈 Last 5 Hours: Actual vs Predicted Comparison")
            st.dataframe(recent_data[['timestamp', 'hour', 'day', 'target_aqi', 'predicted_aqi', 'temperature', 'humidity']])
            
        except Exception as e:
            st.error(f"Error fetching data or running prediction: {e}")
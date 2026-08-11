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
    
    # Model download karein (version check kar lein, maslan 9 hai ya koi aur)
    model = mr.get_model("karachi_aqi_model", version=None)
    model_dir = model.download()
    
    # Automatically directory ke andar .pkl file dhoond lein taake path error na aaye
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pkl") or file.endswith(".h5"):
                model_path = os.path.join(root, file)
                break
                
    if not model_path:
        raise FileNotFoundError(f"Could not find model binary file inside {model_dir}")
        
    trained_model = joblib.load(model_path)
    return fs, trained_model

# Load resources with a status spinner
with st.spinner("Connecting to Hopsworks Cloud & loading model artifacts..."):
    try:
        fs, trained_model = load_hopsworks_resources()
        st.success("Successfully connected to Hopsworks and loaded model!")
    except Exception as e:
        st.error(f"Failed to connect to Hopsworks: {e}")

# 2. Fetching Features from Feature Store for Prediction
if st.button("Fetch Latest Features & Predict"):
    st.info("Querying Feature Store for recent telemetry...")
    
    try:
        # Feature Group se latest features fetch karna
        feature_group = fs.get_feature_group(name="karachi_aqi_features", version=3)
        feature_df = feature_group.read()
        
        # Latest row/data uthana prediction ke liye
        latest_data = feature_df.tail(1)
        
        st.write("### Latest Fetched Features from Store:")
        st.dataframe(latest_data)
        
        st.success("Features successfully loaded and ready for inference!")
        
    except Exception as e:
        st.error(f"Error fetching features: {e}")
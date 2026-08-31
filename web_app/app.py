import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="Karachi AQI Intelligence Dashboard",
    page_icon="🌍",
    layout="wide"
)

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/predict"
@st.cache_data(ttl=600)  # Cache dashboard data for 10 minutes to keep it snappy
def fetch_predictions():
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# 2. Header Section
st.title("Karachi Air Quality Index (AQI) Predictor")
st.markdown("Real-time 3-day forecasting powered by MLOps pipelines, Hopsworks Feature Store, and FastAPI.")

# 3. Fetch Data
data = fetch_predictions()

if "error" in data:
    st.error(f"Could not connect to FastAPI backend: {data['error']}")
    st.info("Please ensure your FastAPI backend is running locally via uvicorn.")
else:
    st.success("Successfully connected to FastAPI backend service!")
    st.write("Raw Payload Received:", data)
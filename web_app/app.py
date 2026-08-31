import streamlit as st
import requests
from datetime import datetime

# 1. Page Config
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for exact SaaS header card look matching your reference screenshot
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .top-header-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        margin-bottom: 2rem;
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .logo-icon {
        background-color: #dcfce7;
        color: #166534;
        padding: 10px;
        border-radius: 12px;
        font-size: 1.2rem;
    }
    .header-right {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .badge-pill {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/predict"

@st.cache_data(ttl=600)
def fetch_predictions():
    try:
        response = requests.get(API_URL, timeout=150)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Fetch Data for Header Timestamp
data = fetch_predictions()
timestamp_str = "Updating..."
if "timestamp" in data:
    try:
        dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        timestamp_str = dt.strftime("%b %d, %Y at %I:%M %p")
    except:
        timestamp_str = str(data["timestamp"])[:16]

# Render the Exact Header Card from Reference
st.markdown(f"""
    <div class="top-header-card">
        <div class="header-left">
            <div class="logo-icon">🌿</div>
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; color: #0f172a; font-weight: 700;">Pearls AQI Predictor</h3>
                <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Advanced air quality monitoring</p>
            </div>
        </div>
        <div class="header-right">
            <div class="badge-pill">📍 Karachi</div>
            <div class="badge-pill">🕒 {timestamp_str}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

if "error" in data:
    st.error(f"Backend Connection Error: {data['error']}")
else:
    st.success("Header loaded successfully! Check if this header matches the style you wanted.")
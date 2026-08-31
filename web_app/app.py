import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

# 1. Page Configuration
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/predict"

@st.cache_data(ttl=600)
def fetch_predictions():
    try:
        response = requests.get(API_URL, timeout=200)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Helper function for AQI category & styling
def get_aqi_category(val):
    if val <= 50:
        return "Good", "🟢"
    elif val <= 100:
        return "Moderate", "🟡"
    elif val <= 150:
        return "Sensitive Groups", "🟠"
    elif val <= 200:
        return "Unhealthy", "🔴"
    else:
        return "Very Unhealthy", "🟣"

# 2. Header Section
st.markdown("### 10PEARLS SHINE INTERNSHIP · AI FORECASTING")
st.title("Karachi AQI Predictor")
st.markdown("Predicting Karachi's Air Quality Index for the next 3 days using an automated machine learning pipeline.")
st.markdown("---")

# 3. Fetch Data
data = fetch_predictions()

if "error" in data:
    st.error(f"Could not connect to FastAPI backend: {data['error']}")
    st.info("Please ensure your FastAPI backend is running locally via uvicorn.")
else:
    timestamp = data.get("timestamp", "N/A")
    current_aqi = data.get("current_aqi", 0.0)
    day1 = data.get("day1", 0.0)
    day2 = data.get("day2", 0.0)
    day3 = data.get("day3", 0.0)

    curr_cat, curr_icon = get_aqi_category(current_aqi)
    d1_cat, d1_icon = get_aqi_category(day1)
    d2_cat, d2_icon = get_aqi_category(day2)
    d3_cat, d3_icon = get_aqi_category(day3)

    # 4. Overview Section
    st.subheader("OVERVIEW")
    st.markdown("#### Current air quality")
    st.caption("Latest observed AQI and automated 3-day forecasting status.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Current AQI", value=f"{current_aqi:.1f}", delta=f"{curr_icon} {curr_cat}")
    with col_b:
        st.metric(label="Latest observation", value=str(timestamp)[:10])
    with col_c:
        st.metric(label="Peak 3-day forecast", value=f"{max(day1, day2, day3):.1f}")

    st.markdown("---")

    # 5. Automated Forecast 3 Days Cards
    st.subheader("AUTOMATED FORECAST")
    st.markdown("### Next 3 days")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown(f"**DAY +1**")
            st.markdown(f"### {d1_icon} {day1:.1f}")
            st.markdown(f"**{d1_cat}**")
            st.caption(f"Forecast for ~ {(datetime.now() + timedelta(days=1)).strftime('%d %b')}")

    with col2:
        with st.container(border=True):
            st.markdown(f"**DAY +2**")
            st.markdown(f"### {d2_icon} {day2:.1f}")
            st.markdown(f"**{d2_cat}**")
            st.caption(f"Forecast for ~ {(datetime.now() + timedelta(days=2)).strftime('%d %b')}")

    with col3:
        with st.container(border=True):
            st.markdown(f"**DAY +3**")
            st.markdown(f"### {d3_icon} {day3:.1f}")
            st.markdown(f"**{d3_cat}**")
            st.caption(f"Forecast for ~ {(datetime.now() + timedelta(days=3)).strftime('%d %b')}")

    st.markdown("---")

    # 6. Forecast Trend Chart
    st.subheader("FORECAST TREND")
    st.markdown("### Expected AQI movement")

    chart_data = pd.DataFrame({
        "Timeline": ["Day +1", "Day +2", "Day +3"],
        "AQI Value": [day1, day2, day3]
    })
    
    st.line_chart(chart_data.set_index("Timeline"), use_container_width=True, color="#d4af37")

    # 7. Health Guidance Section
    st.markdown("---")
    st.subheader("HEALTH ALERTS")
    st.markdown("### AQI health guidance")
    st.caption("Guidance based on the predicted AQI level.")

    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        with st.container(border=True):
            st.markdown(f"**Day +1: {d1_cat}**")
            st.write("Air quality is acceptable. Sensitive individuals should monitor conditions." if day1 <= 100 else "Consider limiting prolonged outdoor exertion.")
    with hcol2:
        with st.container(border=True):
            st.markdown(f"**Day +2: {d2_cat}**")
            st.write("Air quality is acceptable. Sensitive individuals should monitor conditions." if day2 <= 100 else "Consider limiting prolonged outdoor exertion.")
    with hcol3:
        with st.container(border=True):
            st.markdown(f"**Day +3: {d3_cat}**")
            st.write("Air quality is acceptable. Sensitive individuals should monitor conditions." if day3 <= 100 else "Consider limiting prolonged outdoor exertion.")
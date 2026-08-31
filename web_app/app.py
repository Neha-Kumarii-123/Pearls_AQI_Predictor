import streamlit as st
import requests
import pandas as pd
plotly_available = True
try:
    import plotly.express as px
except ImportError:
    plotly_available = False
from datetime import datetime, timedelta

# 1. Page Configuration
st.set_page_config(
    page_title="Karachi AQI Intelligence Dashboard",
    page_icon="🌍",
    layout="wide"
)

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/predict"

@st.cache_data(ttl=3600)
def fetch_predictions():
    try:
        response = requests.get(API_URL, timeout=100)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Helper function for AQI category & styling
def get_aqi_category(val):
    if val <= 50:
        return "Good", "🟢", "#2ecc71"
    elif val <= 100:
        return "Moderate", "🟡", "#f1c40f"
    elif val <= 150:
        return "Sensitive Groups", "🟠", "#e67e22"
    elif val <= 200:
        return "Unhealthy", "🔴", "#e74c3c"
    else:
        return "Very Unhealthy", "🟣", "#8e44ad"

# 2. Header Section
st.markdown("### 10PEARLS SHINE INTERNSHIP · ENTERPRISE MLOPS")
st.title("Karachi AQI Intelligence Dashboard")
st.markdown("Real-time automated 3-day air quality forecasting powered by Hopsworks Feature Store, FastAPI, and Streamlit.")
st.markdown("---")

# 3. Fetch Data
data = fetch_predictions()

if "error" in data:
    st.error(f"Could not connect to FastAPI backend: {data['error']}")
    st.info("Please ensure your FastAPI backend is running locally via uvicorn (`uvicorn web_app.backend_api:app --reload --host 127.0.0.1 --port 8000`).")
else:
    timestamp = data.get("timestamp", "N/A")
    current_aqi = data.get("current_aqi", 0.0)
    day1 = data.get("day1", 0.0)
    day2 = data.get("day2", 0.0)
    day3 = data.get("day3", 0.0)
    
    v1 = data.get("day1_model_version", 1)
    v2 = data.get("day2_model_version", 1)
    v3 = data.get("day3_model_version", 1)

    curr_cat, curr_icon, curr_color = get_aqi_category(current_aqi)
    d1_cat, d1_icon, _ = get_aqi_category(day1)
    d2_cat, d2_icon, _ = get_aqi_category(day2)
    d3_cat, d3_icon, _ = get_aqi_category(day3)

    # 4. Multi-Tab SaaS Navigation Structure
    tab1, tab2, tab3 = st.tabs(["📊 Overview & Forecast", "📈 Interactive Trend Analysis", "🔍 MLOps Registry & Health"])

    with tab1:
        st.subheader("Overview & Current Status")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(label="Current Baseline AQI", value=f"{current_aqi:.1f}", delta=f"{curr_icon} {curr_cat}")
        with col_b:
            st.metric(label="Last Pipeline Sync", value=str(timestamp)[:16])
        with col_c:
            st.metric(label="Peak 3-Day Forecast", value=f"{max(day1, day2, day3):.1f}")

        st.markdown("---")
        st.subheader("Automated 3-Day Forecast Cards")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.markdown(f"**DAY +1**")
                st.markdown(f"### {d1_icon} {day1:.1f}")
                st.markdown(f"**{d1_cat}**")
                st.caption(f"Target: {(datetime.now() + timedelta(days=1)).strftime('%d %b %Y')}")
        with col2:
            with st.container(border=True):
                st.markdown(f"**DAY +2**")
                st.markdown(f"### {d2_icon} {day2:.1f}")
                st.markdown(f"**{d2_cat}**")
                st.caption(f"Target: {(datetime.now() + timedelta(days=2)).strftime('%d %b %Y')}")
        with col3:
            with st.container(border=True):
                st.markdown(f"**DAY +3**")
                st.markdown(f"### {d3_icon} {day3:.1f}")
                st.markdown(f"**{d3_cat}**")
                st.caption(f"Target: {(datetime.now() + timedelta(days=3)).strftime('%d %b %Y')}")

    with tab2:
        st.subheader("Interactive AQI Trajectory")
        st.markdown("Hover over data points to inspect exact predicted indices across the 72-hour forecast window.")

        chart_df = pd.DataFrame({
            "Timeline": ["Current Baseline", "Day +1 Forecast", "Day +2 Forecast", "Day +3 Forecast"],
            "AQI Value": [current_aqi, day1, day2, day3],
            "Category": [curr_cat, d1_cat, d2_cat, d3_cat]
        })

        if plotly_available:
            fig = px.line(
                chart_df, 
                x="Timeline", 
                y="AQI Value", 
                markers=True,
                text="AQI Value",
                color_discrete_sequence=["#2980b9"]
            )
            fig.update_traces(textposition="top center", marker=dict(size=10))
            fig.update_layout(
                xaxis_title="Forecasting Horizon",
                yaxis_title="Predicted AQI",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(chart_df.set_index("Timeline"), use_container_width=True)

    with tab3:
        st.subheader("MLOps Feature Store & Model Versioning")
        st.markdown("Transparency report tracking the active machine learning models serving predictions from the Hopsworks registry.")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            with st.container(border=True):
                st.markdown("**Day +1 Model**")
                st.metric(label="Registered Version", value=f"v{v1}")
                st.caption("Algorithm: Ridge Regression / Optimized Pipeline")
        with col_m2:
            with st.container(border=True):
                st.markdown("**Day +2 Model**")
                st.metric(label="Registered Version", value=f"v{v2}")
                st.caption("Algorithm: Ridge Regression / Optimized Pipeline")
        with col_m3:
            with st.container(border=True):
                st.markdown("**Day +3 Model**")
                st.metric(label="Registered Version", value=f"v{v3}")
                st.caption("Algorithm: Ridge Regression / Optimized Pipeline")

        st.markdown("---")
        st.subheader("Health Guidelines & Actionable Mitigation")
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            with st.container(border=True):
                st.markdown(f"**Day +1: {d1_cat}**")
                st.write("Air quality is acceptable. Sensitive groups should exercise caution during long outdoor exertion.")
        with hcol2:
            with st.container(border=True):
                st.markdown(f"**Day +2: {d2_cat}**")
                st.write("Air quality is acceptable. Sensitive groups should exercise caution during long outdoor exertion.")
        with hcol3:
            with st.container(border=True):
                st.markdown(f"**Day +3: {d3_cat}**")
                st.write("Air quality is acceptable. Sensitive groups should exercise caution during long outdoor exertion.")
"""
Pearls AQI Predictor — Streamlit Dashboard
Connects live to the FastAPI backend's /predict endpoint.

Run with:
    streamlit run streamlit_app.py

Configure the backend URL either by editing DEFAULT_API_URL below,
setting the AQI_API_URL environment variable, or using the "API
Settings" expander in the sidebar at runtime.
"""

import os
import base64
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent

# EDA plot files: (relative path from this file, card title, generic caption).
# Captions describe what each chart shows in general terms only — no invented
# statistics, since we don't have the underlying summary numbers.
EDA_PLOTS = [
    ("assets/eda/aqi_historical_trend.png", "Historical AQI Trend",
     "Trailing daily AQI readings across the historical training window."),
    ("assets/eda/aqi_by_hour.png", "AQI by Hour of Day",
     "Average AQI grouped by hour, showing diurnal variation."),
    ("assets/eda/aqi_by_day_of_week.png", "AQI by Day of Week",
     "Average AQI compared across weekdays vs. weekends."),
    ("assets/eda/aqi_distribution.png", "AQI Distribution",
     "Distribution of AQI values across the historical dataset."),
    ("assets/eda/aqi_vs_pollutants.png", "AQI vs. Pollutant Features",
     "Relationship between AQI and individual pollutant concentrations."),
    ("assets/eda/high_aqi_analysis.png", "High-AQI Event Analysis",
     "Conditions and features associated with high-AQI episodes."),
]

# ============================================================
# CONFIG
# ============================================================
DEFAULT_API_URL = os.environ.get("AQI_API_URL", "http://localhost:8000")

# Static, stable metadata about which algorithm powers each horizon.
# (This does not change request-to-request, so it isn't part of the
# API response — only the *version numbers* are, and those are read
# live from /predict below.)
MODEL_TYPES = {
    "day1": "XGBoost",
    "day2": "Ridge Regression",
    "day3": "Ridge Regression",
}

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "last_timestamp" not in st.session_state:
    st.session_state.last_timestamp = None
if "prev_current_aqi" not in st.session_state:
    st.session_state.prev_current_aqi = None
if "fetch_nonce" not in st.session_state:
    st.session_state.fetch_nonce = 0
if "page" not in st.session_state:
    st.session_state.page = "global"

# ============================================================
# AQI CATEGORY HELPERS
# ============================================================
def get_aqi_category(aqi: float):
    """Returns (short_label, full_risk_label, color) for an AQI value."""
    if aqi is None:
        return "Unknown", "No Data", "#64748b"
    if aqi <= 50:
        return "Good", "GOOD AIR QUALITY", "#34d399"
    if aqi <= 100:
        return "Moderate", "MODERATE HEALTH RISK", "#f5b731"
    if aqi <= 150:
        return "Unhealthy (SG)", "UNHEALTHY FOR SENSITIVE GROUPS", "#fb923c"
    if aqi <= 200:
        return "Unhealthy", "UNHEALTHY", "#f87171"
    if aqi <= 300:
        return "Very Unhealthy", "VERY UNHEALTHY", "#c084fc"
    return "Hazardous", "HAZARDOUS", "#b91c1c"


def progress_pct(aqi: float) -> int:
    """Fill percentage for the small forecast progress bars, capped at 100."""
    if aqi is None:
        return 0
    return max(4, min(100, round((aqi / 200) * 100)))


# ============================================================
# API CALL
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_predictions(api_url: str, _nonce: int):
    """
    Calls the FastAPI /predict endpoint. `_nonce` is bumped by the
    Refresh button to force a fresh call (the backend itself also
    caches for 1 hour, so this mostly avoids re-hitting it needlessly
    on unrelated Streamlit reruns like a theme toggle).
    """
    resp = requests.get(f"{api_url}/predict", timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    return data


def format_sync_time(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta_min = int((now - ts).total_seconds() // 60)
        if delta_min < 1:
            return "Just now"
        if delta_min < 60:
            return f"{delta_min}m ago"
        return f"{delta_min // 60}h ago"
    except Exception:
        return "Unknown"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_shap_explanations(api_url: str, _nonce: int):
    """
    Calls the FastAPI /explain endpoint. Expected shape:
    {
      "day1": {"prediction": .., "base_value": .., "features": [
          {"feature": .., "shap_value": .., "impact": ..}, ...
      ]},
      "day2": {...}, "day3": {...}
    }
    """
    resp = requests.get(f"{api_url}/explain", timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    return data


# ============================================================
# THEME / CSS
# ============================================================
def inject_css(theme: str):
    if theme == "dark":
        bg_main = "#0b1120"
        bg_sidebar = "#0a0f1e"
        bg_card = "#111a2e"
        border = "#1e293b"
        text_primary = "#f8fafc"
        text_secondary = "#94a3b8"
        chart_grid = "#1e293b"
        glass_bg = "rgba(23, 31, 51, 0.55)"
        glass_border = "rgba(78, 222, 163, 0.25)"
        glass_shadow = "rgba(0,0,0,0.28)"
    else:
        bg_main = "#f1f5f9"
        bg_sidebar = "#ffffff"
        bg_card = "#ffffff"
        border = "#e2e8f0"
        text_primary = "#0f172a"
        text_secondary = "#64748b"
        chart_grid = "#e2e8f0"
        glass_bg = "rgba(255, 255, 255, 0.85)"
        glass_border = "rgba(16, 185, 129, 0.35)"
        glass_shadow = "rgba(15, 23, 42, 0.10)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_main};
            color: {text_primary};
            margin-top: -2.5rem;
        }}
        header[data-testid="stHeader"] {{
            display: none;
        }}
        #MainMenu {{
            visibility: hidden;
        }}
        [data-testid="collapsedControl"] {{
            top: 0.6rem !important;
            left: 2rem !important;
        }}
        [data-testid="stSidebarCollapseButton"] {{
            margin-top: 0.6rem !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
            border-right: 1px solid {border};
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_primary};
        }}
        .block-container {{
            padding-top: 3rem;
            padding-bottom: 2rem;
        }}

        .aqi-card {{
            background-color: {bg_card};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 18px;
        }}
        .aqi-card-title {{
            font-size: 17px;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 2px;
        }}
        .aqi-card-subtitle {{
            font-size: 13px;
            color: {text_secondary};
            margin-bottom: 14px;
        }}
        .aqi-big-number {{
            font-size: 52px;
            font-weight: 800;
            color: {text_primary};
            line-height: 1;
            margin: 6px 0 14px 0;
        }}
        .aqi-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }}
        .aqi-divider {{
            border-top: 1px solid {border};
            margin: 16px 0 10px 0;
        }}
        .aqi-footer-row {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: {text_secondary};
        }}
        .mini-card {{
            background-color: {bg_main if theme == "dark" else "#f8fafc"};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 16px;
            text-align: left;
        }}
        .mini-label {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.8px;
            color: {text_secondary};
            text-transform: uppercase;
        }}
        .mini-number {{
            font-size: 30px;
            font-weight: 800;
            color: {text_primary};
            margin: 4px 0 10px 0;
        }}
        .mini-bar-track {{
            background-color: {border};
            border-radius: 999px;
            height: 6px;
            width: 100%;
            margin-bottom: 8px;
        }}
        .mini-bar-fill {{
            height: 6px;
            border-radius: 999px;
        }}
        .mini-risk-label {{
            font-size: 13px;
            font-weight: 600;
        }}
        .model-box {{
            border: 1px solid {border};
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }}
        .model-name {{
            font-weight: 700;
            font-size: 15px;
            color: {text_primary};
        }}
        .model-sub {{
            font-size: 12.5px;
            color: {text_secondary};
            margin-top: 4px;
        }}
        .version-badge {{
            float: right;
            background-color: rgba(52,211,153,0.15);
            color: #34d399;
            border: 1px solid rgba(52,211,153,0.35);
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 700;
        }}
        .top-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid {border};
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 600;
            color: {text_primary};
            margin-left: 8px;
        }}
        /* ---- Why This Prediction? (SHAP Insights) ---- */
        .shap-card {{
            display: flex;
            gap: 32px;
            flex-wrap: wrap;
        }}
        .shap-left {{
            flex: 0 0 200px;
        }}
        .shap-forecast-label {{
            font-size: 13px;
            color: {text_secondary};
            margin-bottom: 6px;
        }}
        .shap-pred-value {{
            font-size: 40px;
            font-weight: 800;
            color: {text_primary};
            line-height: 1;
        }}
        .shap-delta {{
            display: block;
            font-size: 13px;
            font-weight: 700;
            margin-top: 8px;
        }}
        .shap-delta.positive {{ color: #f87171; }}
        .shap-delta.negative {{ color: #34d399; }}
        .shap-base-label {{
            font-size: 13px;
            color: {text_secondary};
            margin-top: 14px;
        }}
        .shap-right {{
            flex: 1;
            min-width: 260px;
            border-left: 1px solid {border};
            padding-left: 28px;
        }}
        .shap-features-title {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            color: {text_secondary};
            margin-bottom: 16px;
        }}
        .shap-feature-row {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 14px;
        }}
        .shap-feature-name {{
            flex: 0 0 90px;
            font-size: 13px;
            color: {text_primary};
        }}
        .shap-track {{
            flex: 1;
            height: 8px;
            background-color: {border};
            border-radius: 999px;
            overflow: hidden;
        }}
        .shap-fill {{
            height: 100%;
            border-radius: 999px;
        }}
        .shap-fill.positive {{ background-color: #34d399; }}
        .shap-fill.negative {{ background-color: #ef8a99; }}
        .shap-feature-value {{
            flex: 0 0 46px;
            text-align: right;
            font-size: 13px;
            font-weight: 700;
        }}
        .shap-feature-value.positive {{ color: #34d399; }}
        .shap-feature-value.negative {{ color: #ef8a99; }}
        .live-dot {{
            height: 8px;
            width: 8px;
            background-color: #34d399;
            border-radius: 50%;
            display: inline-block;
        }}
        .nav-item-active {{
            background-color: #10b981;
            color: white !important;
            border-radius: 10px;
            padding: 10px 14px;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .nav-item {{
            color: {text_secondary} !important;
            padding: 10px 14px;
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .footer-text {{
            color: {text_secondary};
            font-size: 12.5px;
            padding-top: 14px;
            border-top: 1px solid {border};
            margin-top: 10px;
        }}
        div.stButton > button {{
            background-color: #10b981;
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.5rem 1rem;
        }}
        div.stButton > button:hover {{
            background-color: #0ea371;
            color: white;
        }}

        /* Sidebar nav buttons (Global Monitoring / EDA Insights when not active)
           render as plain left-aligned rows instead of green pills, matching
           the static nav-item look. The Refresh Forecast button stays green
           via kind="primary". */
        section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {{
            background-color: transparent;
            color: {text_secondary};
            border: none;
            text-align: left;
            justify-content: flex-start;
            font-weight: 600;
            padding: 10px 14px;
            border-radius: 10px;
            width: 100%;
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {{
            background-color: rgba(255,255,255,0.06);
            color: {text_primary};
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
            background-color: #10b981;
            color: white;
            width: 100%;
        }}

        /* ---- EDA Insights page: Obsidian Emerald glass-morphism (theme-aware) ---- */
        .eda-breadcrumb {{
            font-size: 12.5px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            color: {text_secondary};
        }}
        .eda-breadcrumb .active {{
            color: #10b981;
        }}
        .eda-title {{
            font-size: 32px;
            font-weight: 800;
            color: {text_primary};
            margin: 10px 0 8px 0;
        }}
        .eda-subtitle {{
            font-size: 14.5px;
            color: {text_secondary};
            line-height: 1.55;
            max-width: 820px;
            margin-bottom: 22px;
        }}
        .glass-card {{
            background: {glass_bg};
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid {glass_border};
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 22px;
            box-shadow: 0 8px 32px {glass_shadow};
        }}
        .glass-card-title {{
            font-size: 15px;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 4px;
        }}
        .glass-card-caption {{
            font-size: 12.5px;
            color: {text_secondary};
            line-height: 1.4;
            margin-bottom: 10px;
        }}
        .glass-missing {{
            padding: 48px 16px;
            text-align: center;
            color: {text_secondary};
            font-size: 12.5px;
            border: 1px dashed {border};
            border-radius: 10px;
        }}
        .glass-card img {{
            width: 100%;
            border-radius: 10px;
            display: block;
            border: 1px solid {border};
        }}

        @keyframes pulse {{
            0% {{ opacity: 0.4; }}
            50% {{ opacity: 0.9; }}
            100% {{ opacity: 0.4; }}
        }}
        .skeleton-card {{
            animation: pulse 1.5s ease-in-out infinite;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return dict(
        bg_main=bg_main, bg_card=bg_card, border=border,
        text_primary=text_primary, text_secondary=text_secondary,
        chart_grid=chart_grid,
    )


colors = inject_css(st.session_state.theme)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        "<div style='font-size:22px;font-weight:800;color:#34d399;'>Pearls AQI</div>"
        "<div style='font-size:13px;color:#94a3b8;margin-bottom:24px;'>Environmental AI</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.page == "global":
        st.markdown("<div class='nav-item-active'>🌐&nbsp;&nbsp;Global Monitoring</div>", unsafe_allow_html=True)
    else:
        if st.button("🌐  Global Monitoring", key="nav_global", use_container_width=True):
            st.session_state.page = "global"
            st.rerun()

    if st.session_state.page == "eda":
        st.markdown("<div class='nav-item-active'>📊&nbsp;&nbsp;EDA Insights</div>", unsafe_allow_html=True)
    else:
        if st.button("📊  EDA Insights", key="nav_eda", use_container_width=True):
            st.session_state.page = "eda"
            st.rerun()

    st.markdown("<div class='nav-item'>🧩&nbsp;&nbsp;ML Pipelines</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>🗄️&nbsp;&nbsp;Model Registry</div>", unsafe_allow_html=True)

    if st.session_state.page == "global":
        with st.expander("⚙️ API Settings"):
            new_url = st.text_input("FastAPI base URL", value=st.session_state.api_url)
            if new_url != st.session_state.api_url:
                st.session_state.api_url = new_url
                fetch_predictions.clear()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Refresh Forecast", use_container_width=True, type="primary"):
            st.session_state.fetch_nonce += 1
            fetch_predictions.clear()
            fetch_shap_explanations.clear()
            st.rerun()

if st.session_state.page == "eda":
    # ============================================================
    # EDA INSIGHTS PAGE
    # ============================================================
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown(
            "<div class='eda-breadcrumb'>DASHBOARD &nbsp;›&nbsp; "
            "<span class='active'>EDA INSIGHTS</span></div>"
            "<div class='eda-title'>Exploratory Data Analysis &amp; Feature Trends</div>"
            "<div class='eda-subtitle'>Temporal distributions, pollutant correlations, and "
            "historical patterns underlying the Karachi AQI forecasting model.</div>",
            unsafe_allow_html=True,
        )
    with top_r:
        st.markdown("<div style='padding-top:10px;'></div>", unsafe_allow_html=True)
        if st.button("← Back to Global View", key="back_to_global", use_container_width=True):
            st.session_state.page = "global"
            st.rerun()

    eda_cols = st.columns(2)
    for i, (rel_path, title, caption) in enumerate(EDA_PLOTS):
        img_path = APP_DIR / rel_path
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            img_html = f'<img src="data:image/png;base64,{b64}" alt="{title}" />'
        else:
            img_html = f"<div class='glass-missing'>Image not found:<br>{rel_path}</div>"
        with eda_cols[i % 2]:
            st.markdown(
                f"<div class='glass-card'>"
                f"<div class='glass-card-title'>{title}</div>"
                f"<div class='glass-card-caption'>{caption}</div>"
                f"{img_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="footer-text" style="display:flex;justify-content:space-between;">
            <span>© {datetime.now().year} Pearls AQI Predictor • Advanced Air Quality Monitoring & 3-Day Forecasting</span>
            <span>Privacy Policy &nbsp;•&nbsp; Terms of Service &nbsp;•&nbsp; System Status</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ============================================================
# TOP BAR (Global Monitoring)
# ============================================================
top_left, top_right = st.columns([3, 2])
with top_left:
    st.markdown(
        "<div style='font-size:26px;font-weight:800;'>Pearls AQI Predictor</div>"
        "<div style='font-size:13.5px;color:#94a3b8;'>Advanced Air Quality Monitoring & 3-Day Forecasting</div>",
        unsafe_allow_html=True,
    )
with top_right:
    theme_icon = "🌙" if st.session_state.theme == "dark" else "☀️"
    b1, b2, b3, b4 = st.columns([1.3, 1, 0.6, 0.6])
    with b1:
        st.markdown(
            "<div style='text-align:right;padding-top:8px;'>"
            "<span class='top-badge'><span class='live-dot'></span> LIVE</span>"
            "<span class='top-badge'>KARACHI</span></div>",
            unsafe_allow_html=True,
        )
    with b3:
        if st.button(theme_icon, key="theme_toggle"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

# ============================================================
# FETCH DATA
# ============================================================
data = None
error_msg = None

loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown(
        """
        <div class="aqi-card skeleton-card">
            <div class="aqi-card-title">📡 Generating Forecast</div>
            <div class="aqi-card-subtitle">Running inference on live model pipeline — Karachi Urban Core</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

try:
    data = fetch_predictions(st.session_state.api_url, st.session_state.fetch_nonce)
except requests.exceptions.ConnectionError:
    error_msg = (
        f"Couldn't reach the API at **{st.session_state.api_url}**. "
        "Make sure your FastAPI backend is running, e.g.:\n\n"
        "`uvicorn backend_api:app --reload`"
    )
except requests.exceptions.Timeout:
    error_msg = "The API request timed out. The model may still be warming up — try Refresh Forecast in a moment."
except Exception as exc:
    error_msg = f"API returned an error: {exc}"
finally:
    loading_placeholder.empty()

if error_msg:
    st.error(error_msg)
    st.stop()

# ---- track prev value across genuinely new fetches (new timestamp) ----
current_aqi = float(data["current_aqi"])
day1, day2, day3 = float(data["day1"]), float(data["day2"]), float(data["day3"])
timestamp = data["timestamp"]

if st.session_state.last_timestamp != timestamp:
    st.session_state.prev_current_aqi = st.session_state.get("_last_seen_aqi")
    st.session_state.last_timestamp = timestamp
    st.session_state["_last_seen_aqi"] = current_aqi

delta = None
if st.session_state.prev_current_aqi is not None:
    delta = round(current_aqi - st.session_state.prev_current_aqi, 1)

sync_label = format_sync_time(timestamp)

# ============================================================
# ROW 1 — CURRENT BASELINE + 3-DAY FORECAST
# ============================================================
col1, col2 = st.columns([1, 1.7])

with col1:
    _, full_risk, color = get_aqi_category(current_aqi)
    st.markdown(
        f"""
        <div class="aqi-card">
            <div class="aqi-card-title">📡 Current AQI</div>
            <div class="aqi-card-subtitle">Karachi Urban Core</div>
            <div class="aqi-big-number">{current_aqi:.0f}</div>
            <span class="aqi-badge" style="background-color:{color}22;color:{color};border:1px solid {color}55;">{full_risk}</span>
            <div class="aqi-divider"></div>
            <div class="aqi-footer-row">
                <span>{"📈 " if delta is not None and delta >= 0 else "📉 " if delta is not None else "•"}
                {(f"<span style='color:#f87171;font-weight:700;'>↑{abs(delta)}</span> vs last sync" if delta and delta > 0
                  else f"<span style='color:#34d399;font-weight:700;'>↓{abs(delta)}</span> vs last sync" if delta and delta < 0
                  else "No prior reading yet this session")}</span>
                <span>Sync: {sync_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="aqi-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
                <div class="aqi-card-title" style="margin-bottom:0;">✨ Automated 3-Day Forecast</div>
                <span style="color:#34d399;font-size:13px;font-weight:600;">View Details</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns(3)
    forecast_days = [
        ("24H Horizon", day1),
        ("48H Horizon", day2),
        ("72H Horizon", day3),
    ]
    for col, (label, val) in zip((f1, f2, f3), forecast_days):
        short_label, _, fcolor = get_aqi_category(val)
        pct = progress_pct(val)
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-label">{label}</div>
                    <div class="mini-number">{val:.0f}</div>
                    <div class="mini-bar-track">
                        <div class="mini-bar-fill" style="width:{pct}%;background-color:{fcolor};"></div>
                    </div>
                    <div class="mini-risk-label" style="color:{fcolor};">{short_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ROW 2 — TRAJECTORY CHART + MODEL REGISTRY
# ============================================================
col3, col4 = st.columns([1.7, 1])

with col3:
    st.markdown(
        """
        <div class="aqi-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <div class="aqi-card-title" style="margin-bottom:0;">📈 3-Day Trajectory Forecast</div>
                <span class="top-badge" style="color:#34d399;border-color:#34d39955;">AQI Overall</span>
            </div>
            <div class="aqi-card-subtitle" style="margin-bottom:4px;">
                Live reading plus model forecasts for the next 3 days
            </div>
        """,
        unsafe_allow_html=True,
    )

    x_labels = ["Now", "Day +1", "Day +2", "Day +3"]
    y_values = [current_aqi, day1, day2, day3]
    point_colors = [get_aqi_category(v)[2] for v in y_values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=y_values,
            mode="lines+markers+text",
            line=dict(color="#34d399", width=3, shape="spline", smoothing=1.1),
            marker=dict(size=16, color=point_colors, line=dict(width=2, color=colors["bg_card"])),
            text=[f"{v:.0f}" for v in y_values],
            textposition="top center",
            textfont=dict(color=colors["text_primary"], size=13),
            fill="tozeroy",
            fillcolor="rgba(52,211,153,0.08)",
            hovertemplate="%{x}: %{y:.1f} AQI<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        showlegend=False,
        xaxis=dict(showgrid=False, color=colors["text_secondary"]),
        yaxis=dict(showgrid=True, gridcolor=colors["chart_grid"], color=colors["text_secondary"], zeroline=False),
        font=dict(color=colors["text_primary"]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    model_rows = ""
    for key, day_label, version in [
        ("day1", "Day +1 Model", data["day1_model_version"]),
        ("day2", "Day +2 Model", data["day2_model_version"]),
        ("day3", "Day +3 Model", data["day3_model_version"]),
    ]:
        model_rows += f"""
        <div class="model-box">
            <span class="version-badge">v{version}</span>
            <div class="model-name">{day_label}</div>
            <div class="model-sub">{MODEL_TYPES[key]} • Registered</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="aqi-card">
            <div class="aqi-card-title" style="margin-bottom:14px;">🗄️ Model Registry</div>
            {model_rows}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# WHY THIS PREDICTION? — SHAP INSIGHTS
# ============================================================
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;margin:8px 0 16px 0;">
        <div style="font-size:26px;font-weight:800;">Why This Prediction?</div>
        <span class="top-badge" style="color:#34d399;border-color:#34d39955;">SHAP Insights</span>
    </div>
    """,
    unsafe_allow_html=True,
)

shap_error = None
try:
    shap_data = fetch_shap_explanations(st.session_state.api_url, st.session_state.fetch_nonce)
except requests.exceptions.ConnectionError:
    shap_error = f"Couldn't reach the API at **{st.session_state.api_url}/explain**."
except requests.exceptions.Timeout:
    shap_error = "The /explain request timed out — the SHAP cache may still be warming up."
except Exception as exc:
    shap_error = f"SHAP explanations unavailable: {exc}"

if shap_error:
    st.warning(shap_error)
else:
    for key, label in [("day1", "Forecast: Tomorrow"), ("day2", "Forecast: Day +2"), ("day3", "Forecast: Day +3")]:
        horizon = shap_data.get(key) if isinstance(shap_data, dict) else None
        if not horizon:
            continue

        prediction = float(horizon["prediction"])
        base_value = float(horizon["base_value"])
        delta = prediction - base_value
        delta_sign = "↑" if delta >= 0 else "↓"
        delta_class = "positive" if delta >= 0 else "negative"

        features = horizon.get("features", [])
        max_abs = max((abs(float(f["shap_value"])) for f in features), default=0) or 1

        rows_html = ""
        for f in features:
            sval = float(f["shap_value"])
            cls = "positive" if sval >= 0 else "negative"
            pct = max(4, min(100, round(abs(sval) / max_abs * 100)))
            sign = "+" if sval >= 0 else ""
            rows_html += f"""
            <div class="shap-feature-row">
                <div class="shap-feature-name">{f["feature"]}</div>
                <div class="shap-track"><div class="shap-fill {cls}" style="width:{pct}%;"></div></div>
                <div class="shap-feature-value {cls}">{sign}{sval:.1f}</div>
            </div>
            """

        st.markdown(
            f"""
            <div class="aqi-card shap-card">
                <div class="shap-left">
                    <div class="shap-forecast-label">{label}</div>
                    <div class="shap-pred-value">{prediction:.0f}</div>
                    <span class="shap-delta {delta_class}">{delta_sign}{abs(delta):.0f} from Base</span>
                    <div class="shap-base-label">Base Value: {base_value:.0f}</div>
                </div>
                <div class="shap-right">
                    <div class="shap-features-title">Top Influential Features</div>
                    {rows_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f"""
    <div class="footer-text" style="display:flex;justify-content:space-between;">
        <span>© {datetime.now().year} Pearls AQI Predictor • Advanced Air Quality Monitoring & 3-Day Forecasting</span>
        <span>Privacy Policy &nbsp;•&nbsp; Terms of Service &nbsp;•&nbsp; System Status</span>
    </div>
    """,
    unsafe_allow_html=True,
)
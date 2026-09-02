"""
KarachiPulse AQI — Streamlit Dashboard
Single-service version: calls prediction and explainability functions directly,
without making HTTP requests to a FastAPI backend.
"""
import sys
from pathlib import Path

# Parent directory (root) ko sys.path mein add karna taake 'src' mil jaye
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
import base64
import os
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.explainability import explain_predictions
from src.predict import predict

APP_DIR = Path(__file__).resolve().parent

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

DEFAULT_API_URL = os.environ.get("AQI_API_URL", "http://localhost:8000")

MODEL_TYPES = {
    "day1": "XGBoost",
    "day2": "Ridge Regression",
    "day3": "Ridge Regression",
}

st.set_page_config(
    page_title="KarachiPulse AQI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"
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


def get_aqi_category(aqi: float):
    """Return (short_label, full_risk_label, color) for an AQI value."""
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
    if aqi is None:
        return 0
    return max(4, min(100, round((aqi / 200) * 100)))


def format_sync_time(ts_str):
    if not ts_str:
        return "Unknown"
    try:
        if isinstance(ts_str, datetime):
            ts = ts_str
        else:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
            
        now = datetime.now(timezone.utc)
        delta_min = int((now - ts).total_seconds() // 60)
        if delta_min < 1:
            return "Just now"
        if delta_min < 60:
            return f"{delta_min}m ago"
        return f"{delta_min // 60}h ago"
    except Exception:
        return "Unknown"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_predictions(_nonce: int):
    """Call the underlying prediction function directly."""
    result = predict()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_shap_explanations(_nonce: int):
    """Call the underlying explainability function directly."""
    result = explain_predictions()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result

def inject_css():
    bg_main = "#f1f5f9"
    bg_sidebar = "#ffffff"
    bg_card = "#ffffff"
    border = "#e2e8f0"
    text_primary = "#0f172a"
    text_secondary = "#64748b"
    chart_grid = "#e2e8f0"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_main};
            color: {text_primary};
        }}
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
            visibility: visible !important;
            display: block !important;
        }}
        #MainMenu {{
            visibility: hidden;
        }}
        /* Sidebar collapse / expand button styling */
        [data-testid="collapsedControl"] {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: {bg_card} !important;
            border: 1px solid #10b981 !important;
            border-radius: 8px !important;
            width: 38px !important;
            height: 38px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
            margin-left: 10px !important;
            margin-top: 5px !important;
            z-index: 999999 !important;
        }}
        [data-testid="collapsedControl"] svg,
        [data-testid="collapsedControl"] svg path {{
            fill: #10b981 !important;
            stroke: #10b981 !important;
        }}
        [data-testid="collapsedControl"]:hover {{
            background-color: #f8fafc !important;
            border-color: #0f172a !important;
        }}
        
        /* Baaki aapke cards aur UI classes waise hi rahengi */
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
            background-color: #f8fafc;
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
            background-color: rgba(0,0,0,0.04);
            color: {text_primary};
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
            background-color: #10b981;
            color: white;
            width: 100%;
        }}
        .eda-breadcrumb {{
            font-size: 12.5px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            color: {text_secondary};
            margin-bottom: 4px;
        }}
        .eda-breadcrumb span {{
            color: #10b981;
        }}
        .eda-title {{
            font-size: 28px;
            font-weight: 800;
            color: {text_primary};
            margin: 6px 0 6px 0;
        }}
        .eda-subtitle {{
            font-size: 14px;
            color: {text_secondary};
            line-height: 1.5;
            max-width: 820px;
            margin-bottom: 20px;
        }}
        .glass-card {{
            background: #ffffff;
            border: 1px solid {border};
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        }}
        .glass-card-title {{
            font-size: 16px;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 2px;
        }}
        .glass-card-caption {{
            font-size: 13px;
            color: {text_secondary};
            line-height: 1.4;
            margin-bottom: 12px;
        }}
        .glass-card img {{
            width: 100%;
            border-radius: 8px;
            display: block;
            border: 1px solid {border};
        }}
        .pipeline-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 14px;
        }}
        .pipeline-step {{
            background-color: #f8fafc;
            border: 1px solid {border};
            border-radius: 12px;
            padding: 18px 22px;
            text-align: center;
            min-width: 130px;
            flex: 1;
        }}
        .pipeline-step.active {{
            background-color: rgba(16,185,129,0.1);
            border: 1px solid #10b981;
        }}
        .pipeline-icon {{
            font-size: 22px;
            margin-bottom: 8px;
        }}
        .pipeline-label {{
            font-size: 13.5px;
            font-weight: 600;
            color: {text_primary};
        }}
        .pipeline-step.active .pipeline-label {{
            color: #10b981;
            font-weight: 700;
        }}
        .pipeline-arrow {{
            font-size: 18px;
            color: {text_secondary};
            font-weight: bold;
        }}
        .model-card {{
            background: #ffffff;
            border: 1px solid {border};
            border-top: 4px solid #10b981;
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.04);
            margin-bottom: 15px;
            transition: all 0.2s ease;
        }}
        .model-card:hover {{
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }}
        .model-version-tag {{
            display: inline-block;
            background-color: rgba(16,185,129,0.1);
            color: #10b981;
            font-size: 11.5px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 6px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .model-card-title {{
            font-size: 18px;
            font-weight: 800;
            color: {text_primary};
            margin-bottom: 4px;
        }}
        .model-card-details {{
            font-size: 13.5px;
            color: {text_secondary};
            margin-bottom: 18px;
        }}
        .model-card-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid {border};
            padding-top: 12px;
            font-size: 12.5px;
            font-weight: 600;
            color: {text_secondary};
        }}
        .model-active-badge {{
            background-color: #d1fae5;
            color: #065f46;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
        }}
        .shap-card {{
            display: flex;
            gap: 24px;
            align-items: flex-start;
        }}
        @media(max-width: 768px) {{
            .shap-card {{ flex-direction: column; }}
        }}
        .shap-left {{
            min-width: 180px;
            padding-right: 20px;
            border-right: 1px solid {border};
        }}
        .shap-right {{
            flex: 1;
        }}
        .shap-forecast-label {{
            font-size: 13px;
            font-weight: 700;
            color: {text_secondary};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }}
        .shap-pred-value {{
            font-size: 38px;
            font-weight: 800;
            color: {text_primary};
            line-height: 1.1;
            margin-bottom: 4px;
        }}
        .shap-base-label {{
            font-size: 12px;
            color: {text_secondary};
        }}
        .shap-features-title {{
            font-size: 14px;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 12px;
        }}
        .shap-feature-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .shap-feature-name {{
            width: 160px;
            font-weight: 600;
            color: {text_primary};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .shap-track {{
            flex: 1;
            background-color: {border};
            border-radius: 999px;
            height: 8px;
            overflow: hidden;
        }}
        .shap-fill {{
            height: 8px;
            border-radius: 999px;
        }}
        .shap-fill.positive {{
            background-color: #f87171;
        }}
        .shap-fill.negative {{
            background-color: #34d399;
        }}
        .shap-feature-value {{
            width: 45px;
            text-align: right;
            font-weight: 700;
            font-size: 12.5px;
        }}
        .shap-feature-value.positive {{
            color: #f87171;
        }}
        .shap-feature-value.negative {{
            color: #34d399;
        }}
        /* Mobile view fix: Ensure sidebar toggle button stays visible and clickable */
        @media (max-width: 768px) {{
            [data-testid="collapsedControl"] {{
                display: flex !important;
                position: fixed !important;
                top: 10px !important;
                left: 10px !important;
                background-color: #ffffff !important;
                z-index: 9999999 !important;
            }}
        }}
        
        [data-testid="collapsedControl"] svg,
        [data-testid="collapsedControl"] svg path {{
            fill: #10b981 !important;
            stroke: #10b981 !important;
        }}
        @media(max-width: 768px) {{
            .pipeline-row {{
                flex-direction: column !important;
                align-items: stretch !important;
            }}
            .pipeline-step {{
                width: 100% !important;
                min-width: unset !important;
            }}
            .pipeline-arrow {{
                transform: rotate(90deg);
                text-align: center;
                margin: 6px 0;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return dict(
        bg_main=bg_main,
        bg_card=bg_card,
        border=border,
        text_primary=text_primary,
        text_secondary=text_secondary,
        chart_grid=chart_grid,
    )

colors = inject_css()
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

    if st.session_state.page == "pipelines":
        st.markdown("<div class='nav-item-active'>🧩&nbsp;&nbsp;ML Pipelines</div>", unsafe_allow_html=True)
    else:
        if st.button("🧩  ML Pipelines", key="nav_pipelines", use_container_width=True):
            st.session_state.page = "pipelines"
            st.rerun()

    if st.session_state.page == "registry":
        st.markdown("<div class='nav-item-active'>🗄️&nbsp;&nbsp;Model Registry</div>", unsafe_allow_html=True)
    else:
        if st.button("🗄️  Model Registry", key="nav_registry", use_container_width=True):
            st.session_state.page = "registry"
            st.rerun()

    if st.session_state.page == "global":
        if st.button("Refresh Forecast", use_container_width=True, type="primary"):
            st.session_state.fetch_nonce += 1
            fetch_predictions.clear()
            fetch_shap_explanations.clear()
            st.rerun()

if st.session_state.page == "eda":
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
            <span>© {datetime.now().year} KarachiPulse AQI • Advanced Air Quality Monitoring & 3-Day Forecasting</span>
            <span>Privacy Policy &nbsp;•&nbsp; Terms of Service &nbsp;•&nbsp; System Status</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if st.session_state.page == "registry":
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown(
            "<div class='eda-breadcrumb'>DASHBOARD &nbsp;›&nbsp; "
            "<span class='active'>MODEL REGISTRY</span></div>"
            "<div class='eda-title'>Model Registry</div>"
            "<div class='eda-subtitle'>Versioned models currently powering each forecast horizon, "
            "including algorithm type and registration status.</div>",
            unsafe_allow_html=True,
        )
    with top_r:
        if st.button("← Back to Global View", key="back_to_global_registry", use_container_width=True):
            st.session_state.page = "global"
            st.rerun()

    registry_data = None
    registry_error = None
    try:
        registry_data = fetch_predictions(st.session_state.fetch_nonce)
        if isinstance(registry_data, dict) and "error" in registry_data:
            registry_error = registry_data["error"]
            registry_data = None
    except Exception as exc:
        registry_error = f"Prediction system returned an error: {exc}"

    if registry_error:
        st.error(registry_error)
    elif registry_data:
        reg_cols = st.columns(3)
        registry_rows = [
            ("day1", "Day +1 Model", "24H Horizon"),
            ("day2", "Day +2 Model", "48H Horizon"),
            ("day3", "Day +3 Model", "72H Horizon"),
        ]
        for col, (key, day_label, horizon_label) in zip(reg_cols, registry_rows):
            version = registry_data.get(f"{key}_model_version", "—")
            with col:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <span class="model-version-tag">v{version}</span>
                        <div class="model-card-title">{day_label}</div>
                        <div class="model-card-details">{MODEL_TYPES[key]} • Registered</div>
                        <div class="model-card-footer">
                            <span>{horizon_label}</span>
                            <span class="model-active-badge">Active</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        f"""
        <div class="footer-text" style="display:flex;justify-content:space-between;">
            <span>© {datetime.now().year} KarachiPulse AQI • Advanced Air Quality Monitoring & 3-Day Forecasting</span>
            <span>Privacy Policy &nbsp;•&nbsp; Terms of Service &nbsp;•&nbsp; System Status</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if st.session_state.page == "pipelines":
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown(
            "<div class='eda-breadcrumb'>DASHBOARD &nbsp;›&nbsp; "
            "<span class='active'>ML PIPELINES</span></div>"
            "<div class='eda-title'>ML Pipelines</div>"
            "<div class='eda-subtitle'>End-to-end flow of data through the forecasting system, "
            "from raw ingestion to the final forecast output.</div>",
            unsafe_allow_html=True,
        )
    with top_r:
        if st.button("← Back to Global View", key="back_to_global_pipelines", use_container_width=True):
            st.session_state.page = "global"
            st.rerun()

    pipeline_steps = [
        ("🗄️", "Raw Data", False),
        ("⚙️", "Processing", False),
        ("📋", "Feature Store", False),
        ("🧠", "Prediction", False),
        ("📊", "Forecast", True),
    ]
    pipeline_html = ""
    for i, (icon, step_label, is_active) in enumerate(pipeline_steps):
        step_class = "pipeline-step active" if is_active else "pipeline-step"
        pipeline_html += (
            f'<div class="{step_class}">'
            f'<div class="pipeline-icon">{icon}</div>'
            f'<div class="pipeline-label">{step_label}</div>'
            f'</div>'
        )
        if i < len(pipeline_steps) - 1:
            pipeline_html += '<div class="pipeline-arrow">→</div>'

    st.markdown(
        '<div class="aqi-card">'
        '<div class="aqi-card-title" style="margin-bottom:16px;">🧩 ML Pipeline Workflow</div>'
        f'<div class="pipeline-row">{pipeline_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="footer-text" style="display:flex;justify-content:space-between;">
            <span>© {datetime.now().year} KarachiPulse AQI • Advanced Air Quality Monitoring & 3-Day Forecasting</span>
            <span>Privacy Policy &nbsp;•&nbsp; Terms of Service &nbsp;•&nbsp; System Status</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown(
        "<div style='font-size:26px;font-weight:800;'>KarachiPulse AQI</div>"
        "<div style='font-size:13.5px;color:#64748b;'>Advanced Air Quality Monitoring & 3-Day Forecasting</div>",
        unsafe_allow_html=True,
    )
with top_right:
    st.markdown(
        "<div style='text-align:right;padding-top:8px;'>"
        "<span class='top-badge'><span class='live-dot'></span> LIVE</span>"
        "<span class='top-badge'>KARACHI</span></div>",
        unsafe_allow_html=True,
    )
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
    data = fetch_predictions(st.session_state.fetch_nonce)
except Exception as exc:
    error_msg = f"Prediction system returned an error: {exc}"
finally:
    loading_placeholder.empty()

if error_msg:
    st.error(error_msg)
    st.stop()

current_aqi = float(data["current_aqi"])
day1, day2, day3 = float(data["day1"]), float(data["day2"]), float(data["day3"])
timestamp = data["timestamp"]
alert_info = data.get("alert", {})
is_hazardous = alert_info.get("is_hazardous", False)
alert_message = alert_info.get("message", "")

if is_hazardous and alert_message:
    st.error(f"🚨 {alert_message}")

if st.session_state.last_timestamp != timestamp:
    st.session_state.prev_current_aqi = st.session_state.get("_last_seen_aqi")
    st.session_state.last_timestamp = timestamp
    st.session_state["_last_seen_aqi"] = current_aqi

delta = None
if st.session_state.prev_current_aqi is not None:
    delta = round(current_aqi - st.session_state.prev_current_aqi, 1)

sync_label = format_sync_time(timestamp)

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
                {(f"<span style='color:#f87171;font-weight:700;'>↑{abs(delta)}</span> vs last sync" if delta and delta > 0 else f"<span style='color:#34d399;font-weight:700;'>↓{abs(delta)}</span> vs last sync" if delta and delta < 0 else "No prior reading yet this session")}</span>
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
    margin=dict(l=40, r=40, t=40, b=20),
    height=360,
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        color=colors["text_secondary"],
        range=[-0.6, len(x_labels) - 0.4],
    ),
    yaxis=dict(showgrid=True, gridcolor=colors["chart_grid"], color=colors["text_secondary"], zeroline=False),
    font=dict(color=colors["text_primary"]),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

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
    shap_data = fetch_shap_explanations(st.session_state.fetch_nonce)
except Exception as exc:
    shap_error = f"SHAP explanations unavailable: {exc}"

if shap_error:
    st.warning(shap_error)
else:
    for key, label in [("day1", "Forecast: Tomorrow"), ("day2", "Forecast: Day +2"), ("day3", "Forecast: Day +3")]:
        horizon = shap_data.get(key) if isinstance(shap_data, dict) else None
        if not horizon:
            continue

        prediction = float(horizon.get("prediction", 0.0))
        base_value = float(horizon.get("base_value", 0.0))
        feature_rows = horizon.get("features", [])

        st.markdown(
            f"""
            <div class="aqi-card">
                <div class="shap-card">
                    <div class="shap-left">
                        <div class="shap-forecast-label">{label}</div>
                        <div class="shap-pred-value">{prediction:.0f}</div>
                        <div class="shap-base-label">Base value: {base_value:.0f}</div>
                    </div>
                    <div class="shap-right">
                        <div class="shap-features-title">Top feature impacts</div>
            """,
            unsafe_allow_html=True,
        )

        for item in feature_rows[:5]:
            feature = str(item.get("feature", "unknown"))
            shap_value = float(item.get("shap_value", 0.0))
            impact = item.get("impact", "neutral")
            width = min(100, max(8, abs(shap_value) * 12))
            direction = "positive" if shap_value >= 0 else "negative"
            st.markdown(
                f"""
                <div class="shap-feature-row">
                    <div class="shap-feature-name">{feature}</div>
                    <div class="shap-track"><div class="shap-fill {direction}" style="width:{width}%;"></div></div>
                    <div class="shap-feature-value {direction}">{shap_value:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"""
    <div class="footer-text" style="display:flex;justify-content:space-between;">
        <span>© {datetime.now().year} KarachiPulse AQI • Advanced Air Quality Monitoring & 3-Day Forecasting</span>
        <span>Privacy Policy &nbsp;•&nbsp; Terms of Service &nbsp;•&nbsp; System Status</span>
    </div>
    """,
    unsafe_allow_html=True,
)

import requests
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"
REQUEST_TIMEOUT = 120


# ============================================================
# AQI CONFIGURATION
# ============================================================

AQI_BANDS = [
    (0, 50, "Good", "#4CAF7D"),
    (50, 100, "Moderate", "#E0C341"),
    (100, 150, "Unhealthy for Sensitive Groups", "#E08D3C"),
    (150, 200, "Unhealthy", "#D9534F"),
    (200, 300, "Very Unhealthy", "#A15FBF"),
    (300, 500, "Hazardous", "#7A1F3D"),
]

AQI_MAX_SCALE = 500


MODEL_TYPES = {
    "day1": "XGBoost",
    "day2": "Ridge Regression",
    "day3": "Ridge Regression",
}


# ============================================================
# DESIGN TOKENS
# ============================================================

HERO_BG_FROM = "#3961B3"
HERO_BG_TO = "#161C27"
HERO_TEXT = "#F5F6F8"
HERO_MUTED = "#9AA3B2"

PAGE_BG = "#FFFFFF"
ALT_BG = "#F7F8FA"
CARD_BORDER = "#E7E9EE"
TEXT_DARK = "#12151C"
TEXT_MUTED = "#666E7D"

ACCENT = "#D4A24C"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    f"""
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800'
        '&family=IBM+Plex+Mono:wght@500;600&display=swap'
    );

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        height: 2.2rem;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    .stApp {{
        background-color: {PAGE_BG};
    }}

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {{
        background: #111827;
        border-right: 1px solid #202938;
    }}

    [data-testid="stSidebar"] * {{
        color: #E5E7EB;
    }}

    .sidebar-brand {{
        padding: 1.2rem 0.5rem 1.5rem 0.5rem;
    }}

    .sidebar-brand-title {{
        color: #FFFFFF;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}

    .sidebar-brand-subtitle {{
        color: #8F9AAF;
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }}

    .sidebar-section {{
        color: #68758A;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin: 1rem 0 0.45rem 0;
    }}

    .sidebar-footer {{
        color: #68758A;
        font-size: 0.68rem;
        line-height: 1.5;
        padding: 1rem 0.5rem;
        margin-top: 2rem;
        border-top: 1px solid #263143;
    }}

    /* ========================================================
       GENERAL
       ======================================================== */

    .section-inner {{
        max-width: 1120px;
        margin: 0 auto;
        padding: 0 2.2rem;
    }}

    .section-eyebrow {{
        color: {ACCENT};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}

    .section-title {{
        color: {TEXT_DARK};
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.3rem;
    }}

    .section-desc {{
        color: {TEXT_MUTED};
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 1.8rem;
        max-width: 700px;
    }}

    /* ========================================================
       HERO
       ======================================================== */

    .hero-wrap {{
        background: linear-gradient(
            160deg,
            {HERO_BG_FROM} 0%,
            {HERO_BG_TO} 100%
        );
        padding: 3.5rem 0 3rem 0;
    }}

    .hero-badge {{
        display: inline-block;
        color: {ACCENT};
        background: rgba(212, 162, 76, 0.12);
        border: 1px solid rgba(212, 162, 76, 0.35);
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }}

    .hero-title {{
        color: {HERO_TEXT};
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.08;
        margin: 0 0 0.8rem 0;
    }}

    .hero-subtitle {{
        color: {HERO_MUTED};
        font-size: 1.05rem;
        max-width: 650px;
        line-height: 1.6;
        margin-bottom: 1.8rem;
    }}

    .hero-pill-row {{
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
    }}

    .hero-pill {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        min-width: 125px;
    }}

    .hero-pill-label {{
        color: {HERO_MUTED};
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 600;
    }}

    .hero-pill-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: {HERO_TEXT};
        margin-top: 0.2rem;
    }}

    .hero-pill-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.4rem;
    }}

    /* ========================================================
       FORECAST CARDS
       ======================================================== */

    .forecast-card {{
        background: #FFFFFF;
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 1.4rem;
        border-left: 4px solid var(--band-color);
        box-shadow:
            0 1px 2px rgba(16,20,30,0.03),
            0 6px 16px rgba(16,20,30,0.045);
        height: 100%;
    }}

    .forecast-label {{
        color: {TEXT_MUTED};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }}

    .forecast-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        color: {TEXT_DARK};
        line-height: 1.1;
        margin: 0.35rem 0 0.15rem 0;
    }}

    .forecast-category {{
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--band-color);
        margin-bottom: 1rem;
    }}

    .forecast-date {{
        color: {TEXT_MUTED};
        font-size: 0.76rem;
        margin-top: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
    }}

    .spectrum-track {{
        position: relative;
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(
            90deg,
            #4CAF7D 0%, #4CAF7D 10%,
            #E0C341 10%, #E0C341 20%,
            #E08D3C 20%, #E08D3C 30%,
            #D9534F 30%, #D9534F 40%,
            #A15FBF 40%, #A15FBF 60%,
            #7A1F3D 60%, #7A1F3D 100%
        );
    }}

    .spectrum-marker {{
        position: absolute;
        top: -4px;
        width: 3px;
        height: 14px;
        background-color: {TEXT_DARK};
        border-radius: 2px;
        transform: translateX(-50%);
    }}

    /* ========================================================
       ALERT
       ======================================================== */

    .alert-card {{
        border-radius: 12px;
        border: 1px solid var(--alert-color);
        background: color-mix(
            in srgb,
            var(--alert-color) 8%,
            white
        );
        padding: 1rem 1.2rem;
        min-height: 130px;
    }}

    /* ========================================================
       EDA CARDS
       ======================================================== */

    .eda-stat {{
        background: #FFFFFF;
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow:
            0 1px 2px rgba(16,20,30,0.03),
            0 5px 14px rgba(16,20,30,0.035);
    }}

    .eda-stat-label {{
        color: {TEXT_MUTED};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
    }}

    .eda-stat-value {{
        color: {TEXT_DARK};
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }}

    .eda-stat-desc {{
        color: {TEXT_MUTED};
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }}

    /* ========================================================
       MODEL CARDS
       ======================================================== */

    .model-card {{
        background: #FFFFFF;
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
    }}

    .model-day {{
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
    }}

    .model-type {{
        color: {TEXT_DARK};
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.35rem 0 0.15rem 0;
    }}

    .model-version {{
        color: {ACCENT};
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
    }}

    /* ========================================================
       FLOW
       ======================================================== */

    .flow-row {{
        display: flex;
        align-items: stretch;
        gap: 0.5rem;
        flex-wrap: wrap;
    }}

    .flow-step {{
        flex: 1;
        min-width: 190px;
        background: #FFFFFF;
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 1.3rem 1.2rem;
    }}

    .flow-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: {ACCENT};
        color: #1A140A;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 0.7rem;
    }}

    .flow-title {{
        color: {TEXT_DARK};
        font-weight: 700;
        font-size: 0.98rem;
        margin-bottom: 0.3rem;
    }}

    .flow-desc {{
        color: {TEXT_MUTED};
        font-size: 0.85rem;
        line-height: 1.5;
    }}

    .flow-arrow {{
        display: flex;
        align-items: center;
        justify-content: center;
        color: {ACCENT};
        font-size: 1.4rem;
        font-weight: 700;
    }}

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-wrap {{
        background: {HERO_BG_FROM};
        margin-top: 3rem;
        padding: 2.3rem 0;
    }}

    .footer-title {{
        color: {HERO_TEXT};
        font-weight: 700;
        font-size: 1rem;
    }}

    .footer-text {{
        color: {HERO_MUTED};
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }}

    </style>
    """
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_aqi_band(aqi):
    for low, high, label, color in AQI_BANDS:
        if low <= aqi < high:
            return label, color

    return AQI_BANDS[-1][2], AQI_BANDS[-1][3]


def get_predictions():
    response = requests.get(
        API_URL,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


def is_valid_prediction_payload(data):
    if not isinstance(data, dict):
        return False

    if "error" in data:
        return False

    required_keys = [
        "timestamp",
        "day1",
        "day2",
        "day3",
    ]

    return all(key in data for key in required_keys)


def get_valid_prediction_data():
    data = st.session_state.get("prediction")

    if not is_valid_prediction_payload(data):
        return None

    return data


def format_base_timestamp(ts_str):
    try:
        dt = datetime.fromisoformat(ts_str)

        return dt.strftime(
            "%a, %d %b %Y · %H:%M UTC%z"
        )

    except Exception:
        return ts_str


def forecast_date_label(base_ts_str, day_offset):
    try:
        dt = (
            datetime.fromisoformat(base_ts_str)
            + timedelta(days=day_offset)
        )

        return dt.strftime("%d %b")

    except Exception:
        return f"Day +{day_offset}"


# ============================================================
# DATA INITIALIZATION
# ============================================================

def initialize_dashboard_data():

    if (
        "prediction" not in st.session_state
        and "attempted_prediction_load" not in st.session_state
    ):

        st.session_state["attempted_prediction_load"] = True

        try:
            data = get_predictions()

            if not isinstance(data, dict):
                st.session_state["prediction_error"] = (
                    "Latest prediction is temporarily unavailable. "
                    "Please try again shortly."
                )
                return

            if "error" in data:
                st.session_state["prediction_error"] = (
                    "Latest prediction is temporarily unavailable. "
                    "Please try again shortly."
                )
                return

            if not is_valid_prediction_payload(data):
                st.session_state["prediction_error"] = (
                    "Latest prediction is temporarily unavailable. "
                    "Please try again shortly."
                )
                return

            st.session_state["prediction"] = data

        except Exception:

            st.session_state["prediction_error"] = (
                "Latest prediction is temporarily unavailable. "
                "Please try again shortly."
            )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.html(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">
                    🌫️ Karachi AQI
                </div>

                <div class="sidebar-brand-subtitle">
                    AI Air Quality Intelligence
                </div>
            </div>
            """
        )

        st.html(
            '<div class="sidebar-section">Navigation</div>'
        )

        page = st.radio(
            "Navigation",
            [
                "Overview",
                "3-Day Forecast",
                "EDA & Insights",
                "Models",
                "Architecture",
            ],
            label_visibility="collapsed",
        )

        st.html(
            """
            <div class="sidebar-footer">
                10Pearls Shine Internship<br>
                AI Forecasting Project<br><br>
                Karachi, Pakistan
            </div>
            """
        )

    return page


# ============================================================
# HERO
# ============================================================

def render_hero():

    data = get_valid_prediction_data()

    if data is not None:

        pills_html = ""

        for i, key in enumerate(
            ["day1", "day2", "day3"],
            start=1,
        ):

            value = data[key]

            _, color = get_aqi_band(value)

            pills_html += (
                f'<div class="hero-pill">'
                f'<div class="hero-pill-label">'
                f'Day +{i}'
                f'</div>'
                f'<div class="hero-pill-value">'
                f'<span class="hero-pill-dot" '
                f'style="background-color:{color};">'
                f'</span>'
                f'{value:.1f}'
                f'</div>'
                f'</div>'
            )

    else:

        pills_html = "".join(
            f'<div class="hero-pill">'
            f'<div class="hero-pill-label">'
            f'Day +{i}'
            f'</div>'
            f'<div class="hero-pill-value" '
            f'style="color:{HERO_MUTED};">'
            f'--'
            f'</div>'
            f'</div>'
            for i in range(1, 4)
        )

    st.html(
        f"""
        <div class="hero-wrap">

            <div class="section-inner">

                <div class="hero-badge">
                    10Pearls Shine Internship · AI Forecasting
                </div>

                <div class="hero-title">
                    Karachi AQI Predictor
                </div>

                <div class="hero-subtitle">
                    Predicting Karachi's Air Quality Index for the
                    next 3 days using an automated machine learning
                    pipeline.
                </div>

                <div class="hero-pill-row">
                    {pills_html}
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

def render_overview_page(data):

    if not is_valid_prediction_payload(data):
        st.warning(
            "Latest prediction is temporarily unavailable. "
            "Please try again shortly."
        )
        return

    aqi = data["current_aqi"]
    timestamp = data["timestamp"]

    category, color = get_aqi_band(aqi)

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Overview
            </div>

            <div class="section-title">
                Current air quality
            </div>

            <div class="section-desc">
                Latest observed AQI followed by the current
                3-day forecasting status.
            </div>

        </div>
        """
    )

    st.html('<div class="section-inner">')

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current AQI",
        f"{aqi:.1f}",
        category,
    )

    c2.metric(
        "Latest observation",
        format_base_timestamp(timestamp),
    )

    peak = max(
        data["day1"],
        data["day2"],
        data["day3"],
    )

    peak_category, _ = get_aqi_band(peak)

    c3.metric(
        "Peak 3-day forecast",
        f"{peak:.1f}",
        peak_category,
    )

    st.html("</div>")

    render_forecast_chart(data)


# ============================================================
# FORECAST PAGE
# ============================================================

def render_forecast_card(
    label,
    value,
    base_ts,
    day_offset,
):

    category, color = get_aqi_band(value)

    marker_pct = min(
        max(
            value / AQI_MAX_SCALE * 100,
            1,
        ),
        99,
    )

    date_label = forecast_date_label(
        base_ts,
        day_offset,
    )

    return f"""
    <div class="forecast-card"
         style="--band-color:{color};">

        <div class="forecast-label">
            {label}
        </div>

        <div class="forecast-value">
            {value:.1f}
        </div>

        <div class="forecast-category">
            {category}
        </div>

        <div class="spectrum-track">

            <div class="spectrum-marker"
                 style="left:{marker_pct}%;">
            </div>

        </div>

        <div class="forecast-date">
            Forecast for ~ {date_label}
        </div>

    </div>
    """


def render_forecast_chart(data):

    days = [
        "Day +1",
        "Day +2",
        "Day +3",
    ]

    values = [
        data["day1"],
        data["day2"],
        data["day3"],
    ]

    fig = go.Figure()

    for low, high, label, color in AQI_BANDS:

        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=color,
            opacity=0.09,
            line_width=0,
        )

    fig.add_trace(
        go.Scatter(
            x=days,
            y=values,
            mode="lines+markers+text",
            text=[
                f"{value:.1f}"
                for value in values
            ],
            textposition="top center",
            line=dict(
                color=ACCENT,
                width=3,
            ),
            marker=dict(
                size=11,
                color=ACCENT,
                line=dict(
                    width=2,
                    color="#FFFFFF",
                ),
            ),
            hovertemplate=(
                "%{x}: %{y:.2f} AQI"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=TEXT_MUTED,
            family="Inter",
        ),
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
        height=340,
        yaxis=dict(
            title="AQI",
            range=[
                0,
                max(
                    200,
                    max(values) * 1.3,
                ),
            ],
            gridcolor=CARD_BORDER,
            zeroline=False,
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )

    with st.container(border=True):

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


def render_forecast_page(data):

    if not is_valid_prediction_payload(data):
        st.warning(
            "Latest prediction is temporarily unavailable. "
            "Please try again shortly."
        )
        return

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Prediction
            </div>

            <div class="section-title">
                3-Day AQI Forecast
            </div>

            <div class="section-desc">
                Direct predictions produced by the production
                inference pipeline.
            </div>

        </div>
        """
    )

    st.html('<div class="section-inner">')

    c1, c2, c3 = st.columns(3)

    c1.html = None

    with c1:
        st.html(
            render_forecast_card(
                "Day +1",
                data["day1"],
                data["timestamp"],
                1,
            )
        )

    with c2:
        st.html(
            render_forecast_card(
                "Day +2",
                data["day2"],
                data["timestamp"],
                2,
            )
        )

    with c3:
        st.html(
            render_forecast_card(
                "Day +3",
                data["day3"],
                data["timestamp"],
                3,
            )
        )

    st.html("</div>")

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Forecast Trend
            </div>

            <div class="section-title">
                Expected AQI movement
            </div>

        </div>
        """
    )

    st.html('<div class="section-inner">')

    render_forecast_chart(data)

    st.html("</div>")

    render_alert_section(data)


# ============================================================
# ALERTS
# ============================================================

def render_alert_section(data):

    if not is_valid_prediction_payload(data):
        st.warning(
            "Latest prediction is temporarily unavailable. "
            "Please try again shortly."
        )
        return

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Health Alerts
            </div>

            <div class="section-title">
                AQI health guidance
            </div>

            <div class="section-desc">
                Health guidance generated by the prediction API
                according to forecast AQI levels.
            </div>

        </div>
        """
    )

    columns = st.columns(3)

    alert_columns = [
        ("Day +1", "day1_alert"),
        ("Day +2", "day2_alert"),
        ("Day +3", "day3_alert"),
    ]

    for column, (
        day_label,
        alert_key,
    ) in zip(
        columns,
        alert_columns,
    ):

        alert = data.get(alert_key)

        if not alert:
            continue

        category = alert.get(
            "category",
            "Unknown",
        )

        level = alert.get(
            "level",
            "unknown",
        )

        message = alert.get(
            "message",
            "No health guidance available.",
        )

        if level == "good":
            alert_color = AQI_BANDS[0][3]

        elif level == "moderate":
            alert_color = AQI_BANDS[1][3]

        elif level in [
            "unhealthy_sensitive",
            "unhealthy for sensitive groups",
        ]:
            alert_color = AQI_BANDS[2][3]

        elif level == "unhealthy":
            alert_color = AQI_BANDS[3][3]

        elif level == "very_unhealthy":
            alert_color = AQI_BANDS[4][3]

        elif level == "hazardous":
            alert_color = AQI_BANDS[5][3]

        else:
            alert_color = TEXT_MUTED

        with column:

            st.html(
                f"""
                <div class="alert-card"
                     style="--alert-color:{alert_color};">

                    <div style="
                        color:{TEXT_DARK};
                        font-weight:700;
                        font-size:0.95rem;
                        margin-bottom:0.45rem;
                    ">
                        {day_label}
                    </div>

                    <div style="
                        color:{alert_color};
                        font-weight:700;
                        font-size:1rem;
                        margin-bottom:0.35rem;
                    ">
                        {category}
                    </div>

                    <div style="
                        color:{TEXT_MUTED};
                        font-size:0.85rem;
                        line-height:1.5;
                    ">
                        {message}
                    </div>

                </div>
                """
            )


# ============================================================
# STATIC EDA PAGE
# ============================================================

def render_eda_page():

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Exploratory Data Analysis
            </div>

            <div class="section-title">
                Understanding the AQI dataset
            </div>

            <div class="section-desc">
                A visual summary of the historical dataset,
                feature engineering pipeline, and AQI modelling
                inputs used by the project.
            </div>

        </div>
        """
    )

    st.html('<div class="section-inner">')

    c1, c2, c3, c4 = st.columns(4)

    stats = [
        (
            c1,
            "Historical observations",
            "17,544",
            "Hourly observations",
        ),
        (
            c2,
            "Raw variables",
            "11",
            "Weather + pollutant data",
        ),
        (
            c3,
            "Model features",
            "100",
            "Canonical engineered features",
        ),
        (
            c4,
            "Forecast horizon",
            "3 Days",
            "Day +1, +2 and +3",
        ),
    ]

    for column, label, value, desc in stats:

        with column:

            st.html(
                f"""
                <div class="eda-stat">

                    <div class="eda-stat-label">
                        {label}
                    </div>

                    <div class="eda-stat-value">
                        {value}
                    </div>

                    <div class="eda-stat-desc">
                        {desc}
                    </div>

                </div>
                """
            )

    st.html("</div>")

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Feature Engineering
            </div>

            <div class="section-title">
                From raw observations to ML features
            </div>

            <div class="section-desc">
                The project transforms hourly weather and pollutant
                observations into a canonical set of 100 model-ready
                features.
            </div>

        </div>
        """
    )

    feature_labels = [
        "Raw observations",
        "Temporal features",
        "Lag features",
        "Rolling statistics",
        "100 model features",
    ]

    feature_values = [
        11,
        12,
        7,
        7,
        100,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=feature_labels,
            y=feature_values,
            text=feature_values,
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Feature engineering overview",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color=TEXT_MUTED,
        ),
        height=360,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        yaxis=dict(
            title="Feature / variable count",
            gridcolor=CARD_BORDER,
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )

    st.html('<div class="section-inner">')

    with st.container(border=True):

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.html("</div>")

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Key Insights
            </div>

            <div class="section-title">
                What the EDA tells us
            </div>

        </div>
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            """
            **Temporal structure**

            AQI is an hourly time-series problem.
            Historical lag and rolling-window features
            capture recent air-quality behaviour.
            """
        )

    with c2:
        st.info(
            """
            **Historical context**

            Lag periods up to 168 hours allow the models
            to learn short-term and weekly AQI patterns.
            """
        )

    with c3:
        st.info(
            """
            **Feature richness**

            Weather, pollutant, temporal, lag and rolling
            features are combined into the canonical
            100-feature model contract.
            """
        )

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Feature Groups
            </div>

            <div class="section-title">
                What's inside the model input
            </div>

        </div>
        """
    )

    feature_groups = {
        "Pollutants": 5,
        "Weather": 4,
        "Temporal": 8,
        "Lag windows": 7,
        "Rolling windows": 7,
    }

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(feature_groups.keys()),
            y=list(feature_groups.values()),
            text=list(feature_groups.values()),
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Feature engineering components",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color=TEXT_MUTED,
        ),
        height=340,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        yaxis=dict(
            title="Feature count",
            gridcolor=CARD_BORDER,
        ),
        showlegend=False,
    )

    st.html('<div class="section-inner">')

    with st.container(border=True):

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.html("</div>")

    st.caption(
        "EDA is presented as a static analytical view inside "
        "the dashboard; it does not call the prediction API."
    )


# ============================================================
# MODELS PAGE
# ============================================================

def render_models_page(data):

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                Model Registry
            </div>

            <div class="section-title">
                Models powering the forecast
            </div>

            <div class="section-desc">
                Each forecast horizon has its own registered
                production model.
            </div>

        </div>
        """
    )

    if not data:
        return

    cards_html = ""

    for i, key in enumerate(
        ["day1", "day2", "day3"],
        start=1,
    ):

        version = data.get(
            f"{key}_model_version",
            "--",
        )

        model_type = MODEL_TYPES[key]

        cards_html += f"""
        <div class="model-card">

            <div class="model-day">
                Day +{i}
            </div>

            <div class="model-type">
                {model_type}
            </div>

            <div class="model-version">
                Registry version v{version}
            </div>

        </div>
        """

    st.html(
        f"""
        <div class="section-inner">

            <div style="
                display:grid;
                grid-template-columns:
                    repeat(3, 1fr);
                gap:1rem;
            ">

                {cards_html}

            </div>

        </div>
        """
    )


# ============================================================
# ARCHITECTURE PAGE
# ============================================================

def render_architecture_page():

    st.html(
        """
        <div class="section-inner"
             style="padding-top:2.8rem;">

            <div class="section-eyebrow">
                System Architecture
            </div>

            <div class="section-title">
                How Karachi AQI Predictor works
            </div>

            <div class="section-desc">
                End-to-end flow from external data sources to
                the final 3-day AQI forecast.
            </div>

        </div>
        """
    )

    steps = [
        (
            "Live Data",
            "Weather and pollutant observations are collected "
            "from external APIs."
        ),
        (
            "Feature Engineering",
            "Historical observations are transformed into "
            "canonical model-ready features."
        ),
        (
            "Hopsworks",
            "Processed features are stored in the Feature Store "
            "for training and inference."
        ),
        (
            "ML Models",
            "Day +1 uses XGBoost while Day +2 and Day +3 "
            "use Ridge Regression."
        ),
        (
            "FastAPI",
            "The production inference API loads the registered "
            "models and generates predictions."
        ),
        (
            "Streamlit",
            "This dashboard presents forecasts, alerts, model "
            "information and EDA insights."
        ),
    ]

    steps_html = ""

    for i, (title, desc) in enumerate(
        steps,
        start=1,
    ):

        steps_html += f"""
        <div class="flow-step">

            <div class="flow-num">
                {i}
            </div>

            <div class="flow-title">
                {title}
            </div>

            <div class="flow-desc">
                {desc}
            </div>

        </div>
        """

        if i < len(steps):

            steps_html += (
                '<div class="flow-arrow">'
                '&rarr;'
                '</div>'
            )

    st.html(
        f"""
        <div style="
            background:{ALT_BG};
            padding:2rem 0;
        ">

            <div class="section-inner">

                <div class="flow-row">
                    {steps_html}
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.html(
        f"""
        <div class="footer-wrap">

            <div class="section-inner">

                <div class="footer-title">
                    Karachi AQI Predictor
                </div>

                <div class="footer-text">
                    10Pearls Shine Internship Project ·
                    AI-powered 3-day air quality forecasting.
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

initialize_dashboard_data()

selected_page = render_sidebar()


if selected_page == "Overview":

    render_hero()

    data = get_valid_prediction_data()

    render_overview_page(data)


elif selected_page == "3-Day Forecast":

    render_hero()

    data = get_valid_prediction_data()

    render_forecast_page(data)


elif selected_page == "EDA & Insights":

    render_eda_page()


elif selected_page == "Models":

    data = get_valid_prediction_data()

    render_models_page(data)


elif selected_page == "Architecture":

    render_architecture_page()


render_footer()
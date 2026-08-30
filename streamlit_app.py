import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"
CURRENT_API_URL = "http://127.0.0.1:8000/current"
HISTORY_API_URL = "http://127.0.0.1:8000/history"
REQUEST_TIMEOUT = 120

# AQI category thresholds + functional colors (unchanged across redesigns
# for consistency — these are standard-ish EPA-style bands, desaturated
# slightly to sit well on both dark and light backgrounds)
AQI_BANDS = [
    (0, 50, "Good", "#4CAF7D"),
    (50, 100, "Moderate", "#E0C341"),
    (100, 150, "Unhealthy for Sensitive Groups", "#E08D3C"),
    (150, 200, "Unhealthy", "#D9534F"),
    (200, 300, "Very Unhealthy", "#A15FBF"),
    (300, 500, "Hazardous", "#7A1F3D"),
]
AQI_MAX_SCALE = 500

# Model type labels come from the documented project architecture
# (frozen predict.py: Day+1 = XGBoost, Day+2/3 = Ridge) — NOT returned by
# the API, which only provides version numbers. If the architecture
# changes, update this mapping.
MODEL_TYPES = {
    "day1": "XGBoost",
    "day2": "Ridge Regression",
    "day3": "Ridge Regression",
}

# ------------------------------------------------------------
# Design tokens
# ------------------------------------------------------------
HERO_BG_FROM = "#3961B3"
HERO_BG_TO = "#161C27"
HERO_TEXT = "#F5F6F8"
HERO_MUTED = "#9AA3B2"

PAGE_BG = "#FFFFFF"
ALT_BG = "#F7F8FA"
CARD_BORDER = "#E7E9EE"
TEXT_DARK = "#12151C"
TEXT_MUTED = "#666E7D"

ACCENT = "#D4A24C"  # smog-amber — brand accent, distinct from AQI band colors


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
)


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Reclaim full width, remove Streamlit's default gutters so our own
       full-bleed sections control spacing precisely */
    .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        height: 2.2rem;
    }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .stApp {{ background-color: {PAGE_BG}; }}

    /* Bordered containers (st.container(border=True)) styled as cards */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px !important;
        border: 1px solid {CARD_BORDER} !important;
        box-shadow: 0 1px 2px rgba(16, 20, 30, 0.04), 0 4px 14px rgba(16, 20, 30, 0.04);
        padding: 0.4rem 0.4rem;
    }}

    .section-inner {{
        max-width: 1120px;
        margin: 0 auto;
        padding: 0 2.2rem;
    }}

    /* Hero */
    .hero-wrap {{
        background: linear-gradient(160deg, {HERO_BG_FROM} 0%, {HERO_BG_TO} 100%);
        padding: 4.5rem 0 4rem 0;
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
        margin-bottom: 1.4rem;
    }}
    .hero-title {{
        color: {HERO_TEXT};
        font-size: 3.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.08;
        margin: 0 0 1rem 0;
    }}
    .hero-subtitle {{
        color: {HERO_MUTED};
        font-size: 1.12rem;
        max-width: 560px;
        line-height: 1.6;
        margin-bottom: 2.4rem;
    }}
    .hero-pill-row {{
        display: flex;
        gap: 0.9rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }}
    .hero-pill {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 0.9rem 1.3rem;
        min-width: 130px;
    }}
    .hero-pill-label {{
        color: {HERO_MUTED};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 600;
    }}
    .hero-pill-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: {HERO_TEXT};
        margin-top: 0.2rem;
    }}
    .hero-pill-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 0.4rem;
    }}

    /* Section headings */
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
        font-size: 0.98rem;
        margin-bottom: 1.8rem;
        max-width: 640px;
    }}

    /* Forecast cards */
    .forecast-card {{
        background: #FFFFFF;
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 1.5rem 1.5rem 1.6rem 1.5rem;
        border-left: 4px solid var(--band-color);
        box-shadow: 0 1px 2px rgba(16,20,30,0.03), 0 6px 16px rgba(16,20,30,0.045);
        height: 100%;
    }}
    .forecast-label {{
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }}
    .forecast-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.7rem;
        font-weight: 700;
        color: {TEXT_DARK};
        line-height: 1.1;
        margin: 0.35rem 0 0.15rem 0;
    }}
    .forecast-category {{
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--band-color);
        margin-bottom: 1rem;
    }}
    .forecast-date {{
        color: {TEXT_MUTED};
        font-size: 0.78rem;
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
        margin-top: 0.4rem;
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

    /* Alert card */
    .alert-card {{
        border-radius: 12px;
        border: 1px solid var(--alert-color);
        background: color-mix(in srgb, var(--alert-color) 8%, white);
        padding: 1.1rem 1.4rem;
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }}
    .alert-dot {{
        width: 10px; height: 10px;
        min-width: 10px;
        border-radius: 50%;
        background: var(--alert-color);
    }}
    .alert-text {{
        color: {TEXT_DARK};
        font-size: 0.95rem;
        font-weight: 500;
    }}

    /* How it works */
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
        width: 26px; height: 26px;
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
        color: {CARD_BORDER};
        font-size: 1.4rem;
        padding: 0 0.2rem;
    }}

    /* Model info cards */
    .model-card {{
        background: #FFFFFF;
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        text-align: left;
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

    /* Footer */
    .footer-wrap {{
        background: {HERO_BG_FROM};
        margin-top: 1rem;
        padding: 2.6rem 0 2.2rem 0;
    }}
    .footer-title {{
        color: {HERO_TEXT};
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }}
    .footer-text {{
        color: {HERO_MUTED};
        font-size: 0.82rem;
        line-height: 1.6;
    }}

    
    </style>
    """,
    unsafe_allow_html=True,
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
    response = requests.get(API_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()
def get_current():
    response = requests.get(
        CURRENT_API_URL,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()

def get_history():
    response = requests.get(
        HISTORY_API_URL,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()

def format_base_timestamp(ts_str):
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%a, %d %b %Y · %H:%M UTC%z")
    except Exception:
        return ts_str


def forecast_date_label(base_ts_str, day_offset):
    try:
        dt = datetime.fromisoformat(base_ts_str) + timedelta(days=day_offset)
        return dt.strftime("%d %b")
    except Exception:
        return f"Day +{day_offset}"



def fetch_current_and_store():
    with st.spinner("Fetching current AQI..."):
        data = get_current()

    st.session_state["current"] = data
    st.session_state["current_error"] = None
def fetch_history_and_store():
    with st.spinner("Fetching historical AQI data..."):
        data = get_history()

    st.session_state["history"] = data
    st.session_state["history_error"] = None

# ============================================================
# SECTION RENDERERS
# ============================================================

def render_hero():
    data = st.session_state.get("prediction")

    if data:
        pills_html = ""

        for i, key in enumerate(["day1", "day2", "day3"], start=1):
            value = data[key]
            _, color = get_aqi_band(value)

            pills_html += (
                f'<div class="hero-pill">'
                f'<div class="hero-pill-label">Day +{i}</div>'
                f'<div class="hero-pill-value">'
                f'<span class="hero-pill-dot" style="background-color:{color};"></span>'
                f'{value:.1f}'
                f'</div>'
                f'</div>'
            )
    else:
        pills_html = "".join(
            f"""
            <div class="hero-pill">
                <div class="hero-pill-label">Day +{i}</div>
                <div class="hero-pill-value" style="color:{HERO_MUTED};">--</div>
            </div>
            """
            for i in range(1, 4)
        )

        st.markdown(
            f"""
            <div class="alert-card" style="--alert-color:{alert_color}; min-height:130px; display:flex; flex-direction:column; align-items:flex-start; justify-content:center;">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
                    <div class="alert-dot"></div>
                    <div style="color:{TEXT_DARK}; font-weight:700; font-size:0.95rem;">
                        {day_label}
                    </div>
                </div>
                <div style="color:{alert_color}; font-weight:700; font-size:1rem; margin-bottom:0.35rem;">
                    {category}
                </div>
                <div style="color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.5;">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def initialize_dashboard_data():
    """
    Load only the FAST, already-cached data (prediction + current AQI)
    before anything is rendered.

    PERFORMANCE FIX: history used to be fetched here too, before
    render_hero() ran. Since /history can take 17-28s on a cache miss,
    that meant the entire page stayed blank for that whole duration on
    first load. History is now fetched lazily inside
    render_history_section() instead (see below), so the rest of the
    dashboard (hero, forecast, trend, alerts) renders immediately and
    only the history section itself shows a loading spinner while it
    catches up.
    """
    if "prediction" not in st.session_state and "attempted_prediction_load" not in st.session_state:
        st.session_state["attempted_prediction_load"] = True
        try:
            data = get_predictions()
            st.session_state["prediction"] = data
        except Exception:
            pass

    if "current" not in st.session_state and "attempted_current" not in st.session_state:
        st.session_state["attempted_current"] = True
        try:
            fetch_current_and_store()
        except Exception as exc:
            st.session_state["current_error"] = f"Unable to fetch current AQI: {exc}"
def render_current_section():
    current = st.session_state.get("current")

    if not current:
        return

    aqi = current["current_aqi"]
    timestamp = current["timestamp"]
    category, color = get_aqi_band(aqi)

    st.markdown(
        f"""
        <div class="section-inner" style="padding-top:2.6rem;">
            <div class="section-eyebrow">Current Air Quality</div>
            <div class="section-title">Latest observed AQI</div>
            <div class="section-desc">
                This is the latest actual AQI observation available in Hopsworks,
                not a model prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-inner">', unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2 = st.columns(2)

        c1.metric(
            "Current AQI",
            f"{aqi:.1f}",
            category,
        )

        c2.metric(
            "Last observed",
            format_base_timestamp(timestamp),
        )

    st.markdown("</div>", unsafe_allow_html=True)

def render_overview_section(data):
    st.markdown(
        f"""
        <div class="section-inner" style="padding-top:2.6rem;">
            <div class="section-eyebrow">Forecast Overview</div>
            <div class="section-title">Where the forecast stands right now</div>
            <div class="section-desc">
                The current observed AQI is shown separately above. This section summarizes
the 3-day model forecast and does not mix observed and predicted values.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    day1, day2, day3 = data["day1"], data["day2"], data["day3"]
    delta1, delta2 = day2 - day1, day3 - day2
    if delta1 > 2 and delta2 > 2:
        trend_label = "Worsening trend"
    elif delta1 < -2 and delta2 < -2:
        trend_label = "Improving trend"
    else:
        trend_label = "Fluctuating / stable"

    peak = max(day1, day2, day3)
    peak_category, peak_color = get_aqi_band(peak)

    st.markdown('<div class="section-inner">', unsafe_allow_html=True)
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast base timestamp", format_base_timestamp(data["timestamp"]))
        m2.metric("Peak forecasted AQI", f"{peak:.1f}", peak_category)
        m3.metric("3-day trajectory", trend_label)
    st.markdown("</div>", unsafe_allow_html=True)


def render_forecast_card(label, value, base_ts, day_offset):
    category, color = get_aqi_band(value)
    marker_pct = min(max(value / AQI_MAX_SCALE * 100, 1), 99)
    date_label = forecast_date_label(base_ts, day_offset)

    return f"""
    <div class="forecast-card" style="--band-color: {color};">
        <div class="forecast-label">{label}</div>
        <div class="forecast-value">{value:.1f}</div>
        <div class="forecast-category">{category}</div>
        <div class="spectrum-track">
            <div class="spectrum-marker" style="left: {marker_pct}%;"></div>
        </div>
        <div class="forecast-date">Forecast for ~ {date_label}</div>
    </div>
    """


def render_forecast_section(data):
    st.markdown(
        """
        <div class="section-inner" style="padding-top:3rem;">
            <div class="section-eyebrow">3-Day Forecast</div>
            <div class="section-title">Air quality, three days out</div>
            <div class="section-desc">
                Each card is a direct, unmodified prediction from the production inference
                pipeline for that day.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-inner">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(render_forecast_card("Day +1", data["day1"], data["timestamp"], 1), unsafe_allow_html=True)
    c2.markdown(render_forecast_card("Day +2", data["day2"], data["timestamp"], 2), unsafe_allow_html=True)
    c3.markdown(render_forecast_card("Day +3", data["day3"], data["timestamp"], 3), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_trend_section(data):
    st.markdown(
        """
        <div class="section-inner" style="padding-top:3rem;">
            <div class="section-eyebrow">Forecast Trend</div>
            <div class="section-title">How AQI is expected to move</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    days = ["Day +1", "Day +2", "Day +3"]
    values = [data["day1"], data["day2"], data["day3"]]

    fig = go.Figure()
    for low, high, label, color in AQI_BANDS:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.09, line_width=0)

    fig.add_trace(
        go.Scatter(
            x=days,
            y=values,
            mode="lines+markers+text",
            text=[f"{v:.1f}" for v in values],
            textposition="top center",
            textfont=dict(color=TEXT_DARK, size=13),
            line=dict(color=ACCENT, width=3),
            marker=dict(size=11, color=ACCENT, line=dict(width=2, color="#FFFFFF")),
            hovertemplate="%{x}: %{y:.2f} AQI<extra></extra>",
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family="Inter"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=340,
        yaxis=dict(title="AQI", range=[0, max(200, max(values) * 1.3)], gridcolor=CARD_BORDER, zeroline=False),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )

    st.markdown('<div class="section-inner">', unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_history_section():
    # Lazily fetch history the first time this section is actually
    # rendered — i.e. after the hero/forecast/trend/alert sections above
    # have already been drawn to the page. Same one-shot-per-session
    # guard (attempted_history) as before, so reruns still don't
    # repeatedly hit Hopsworks.
    if "history" not in st.session_state and "attempted_history" not in st.session_state:
        st.session_state["attempted_history"] = True
        try:
            fetch_history_and_store()
        except Exception as exc:
            st.session_state["history_error"] = (
                f"Unable to fetch historical AQI: {exc}"
            )

    history = st.session_state.get("history")

    if not history or not history.get("data"):
        return

    dataframe = pd.DataFrame(history["data"])

    if dataframe.empty:
        return

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    dataframe["target_aqi"] = pd.to_numeric(
        dataframe["target_aqi"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["date", "target_aqi"]
    ).sort_values("date")

    if dataframe.empty:
        return

    st.markdown(
        """
        <div class="section-inner" style="padding-top:3rem;">
            <div class="section-eyebrow">Historical Air Quality</div>
            <div class="section-title">90-day AQI history</div>
            <div class="section-desc">
                Daily average AQI observations from the Hopsworks Feature Store.
                These are historical observations, not model predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y=dataframe["target_aqi"],
            mode="lines+markers",
            line=dict(
                color=ACCENT,
                width=2,
            ),
            marker=dict(
                size=5,
            ),
            hovertemplate=(
                "%{x|%d %b %Y}"
                "<br>AQI: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    for low, high, label, color in AQI_BANDS:
        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=color,
            opacity=0.06,
            line_width=0,
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
        height=380,
        xaxis=dict(
            title="Date",
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title="Daily Average AQI",
            gridcolor=CARD_BORDER,
            zeroline=False,
        ),
        showlegend=False,
    )

    st.markdown(
        '<div class="section-inner">',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )
    
def render_alert_section(data):
    """
    Display backend-generated AQI alerts for each forecast day.

    Alert information comes directly from the FastAPI /predict response.
    Existing AQI calculations and forecast values remain unchanged.
    """

    st.markdown(
        """
        <div class="section-inner" style="padding-top:3rem;">
            <div class="section-eyebrow">Health Alerts</div>
            <div class="section-title">AQI health guidance</div>
            <div class="section-desc">
                Health alerts are generated by the backend based on each day's
                predicted AQI level.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alert_columns = [
        ("Day +1", "day1_alert"),
        ("Day +2", "day2_alert"),
        ("Day +3", "day3_alert"),
    ]

    st.markdown(
        '<div class="section-inner">',
        unsafe_allow_html=True,
    )

    columns = st.columns(3)

    for column, (day_label, alert_key) in zip(
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

        # Use the same AQI color system already used
        # throughout the dashboard.
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

            st.markdown(
                f"""
                <div class="alert-card"
                     style="
                        --alert-color:{alert_color};
                        min-height:130px;
                        display:flex;
                        flex-direction:column;
                        align-items:flex-start;
                        justify-content:center;
                     ">

                    <div style="
                        display:flex;
                        align-items:center;
                        gap:0.6rem;
                        margin-bottom:0.5rem;
                    ">
                        <div class="alert-dot"></div>

                        <div style="
                            color:{TEXT_DARK};
                            font-weight:700;
                            font-size:0.95rem;
                        ">
                            {day_label}
                        </div>
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
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


def render_how_it_works():
    steps = [
        ("Live Data", "Weather and pollutant readings are pulled hourly from external APIs."),
        ("Feature Store", "Raw data is transformed into model-ready features and stored in Hopsworks."),
        ("ML Models", "Day+1 uses XGBoost; Day+2 and Day+3 use tuned Ridge Regression models."),
        ("3-Day Forecast", "The FastAPI inference layer serves predictions to this dashboard."),
    ]
    steps_html = ""
    for i, (title, desc) in enumerate(steps, start=1):
        steps_html += f"""
        <div class="flow-step">
            <div class="flow-num">{i}</div>
            <div class="flow-title">{title}</div>
            <div class="flow-desc">{desc}</div>
        </div>
        """
        if i < len(steps):
            steps_html += '<div class="flow-arrow">&rarr;</div>'

    st.markdown(
        f"""
        <div style="background:{ALT_BG}; padding: 3rem 0;">
            <div class="section-inner">
                <div class="section-eyebrow">Architecture</div>
                <div class="section-title">How it works</div>
                <div class="section-desc">
                    An end-to-end, automated pipeline — no manual steps between raw data and the
                    forecast shown above.
                </div>
                <div class="flow-row">{steps_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_info(data):
    cards_html = ""
    for i, key in enumerate(["day1", "day2", "day3"], start=1):
        version = data[f"{key}_model_version"]
        model_type = MODEL_TYPES[key]
        cards_html += f"""
        <div class="model-card">
            <div class="model-day">Day +{i}</div>
            <div class="model-type">{model_type}</div>
            <div class="model-version">Registry version v{version}</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="section-inner" style="padding-top:3rem; padding-bottom:2.5rem;">
            <div class="section-eyebrow">Model Registry</div>
            <div class="section-title">What's serving each prediction</div>
            <div class="section-desc">
                Model types come from the documented project architecture; version numbers are
                read live from the API response.
            </div>
        </div>
        <div class="section-inner">
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                {cards_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        """
        <div class="section-inner" style="padding: 3rem 0;">
            <div class="section-desc" style="font-size: 1rem;">
                No forecast loaded yet. Use the button above to fetch the latest 3-day AQI
                prediction from the inference API.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        f"""
        <div class="footer-wrap">
            <div class="section-inner">
                <div class="footer-title">Karachi AQI Predictor</div>
                <div class="footer-text">
                    10Pearls Shine Internship Project · Forecast values are model predictions,
                    not live sensor readings.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE ASSEMBLY
# ============================================================

initialize_dashboard_data()
render_hero()

if "prediction" in st.session_state:
    prediction_data = st.session_state["prediction"]
    render_current_section()
    render_overview_section(prediction_data)
    render_forecast_section(prediction_data)
    render_trend_section(prediction_data)
    render_history_section()
    render_alert_section(prediction_data)
    render_how_it_works()
    render_model_info(prediction_data)
else:
    render_empty_state()
    render_how_it_works()

render_footer()
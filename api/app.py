from fastapi import FastAPI
from src.predict import (
    get_latest_v6_row,
    connect_to_hopsworks,
)
import pandas as pd
import time
import threading


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Karachi AQI Predictor"
)


# ============================================================
# CACHE CONFIGURATION
# ============================================================

# Cache remains valid for 1 hour.
CACHE_TTL = 3600


# ============================================================
# APPLICATION CACHE
# ============================================================

_prediction_cache = None
_prediction_cache_time = 0

_current_cache = None
_current_cache_time = 0

_history_cache = None
_history_cache_time = 0

# One Hopsworks connection for the whole API process.
_hopsworks_project = None


# ============================================================
# AQI ALERT LOGIC
# ============================================================

def get_aqi_alert(aqi):
    """
    Convert AQI value into a health category and alert message.
    """

    aqi = float(aqi)

    if aqi <= 50:

        return {
            "category": "Good",
            "level": "safe",
            "message": "Air quality is good."
        }

    elif aqi <= 100:

        return {
            "category": "Moderate",
            "level": "moderate",
            "message": "Air quality is acceptable."
        }

    elif aqi <= 150:

        return {
            "category": "Unhealthy for Sensitive Groups",
            "level": "warning",
            "message": (
                "Sensitive groups should reduce prolonged "
                "outdoor activity."
            )
        }

    elif aqi <= 200:

        return {
            "category": "Unhealthy",
            "level": "warning",
            "message": (
                "Everyone should reduce prolonged "
                "outdoor activity."
            )
        }

    elif aqi <= 300:

        return {
            "category": "Very Unhealthy",
            "level": "danger",
            "message": (
                "Health alert: everyone may experience "
                "health effects."
            )
        }

    else:

        return {
            "category": "Hazardous",
            "level": "hazard",
            "message": (
                "Health emergency: avoid outdoor activity."
            )
        }


# ============================================================
# HOPSWORKS CONNECTION
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    Kick off cache warm-up in a background thread and return immediately.

    PERFORMANCE FIX: this used to run the Hopsworks connection + the
    ~14-15s get_latest_v6_row() read directly inside the awaited startup
    event, which meant uvicorn would not accept ANY request (not even
    "/" or "/docs") until that finished. Running the exact same warm-up
    logic in a background daemon thread instead lets the server start
    accepting requests immediately; the cache still gets warmed a few
    seconds later, and any request that arrives before it's ready simply
    falls through to the existing cache-miss path (unchanged) and reads
    Hopsworks on demand for that one request.
    """

    print("\n--- Scheduling API cache warm-up in the background ---")

    threading.Thread(
        target=_warm_up_caches,
        daemon=True,
    ).start()


def _warm_up_caches():
    """
    Same warm-up logic as before (unchanged), just moved out of the
    awaited startup event so it can run in a background thread.
    """

    global _hopsworks_project
    global _prediction_cache
    global _prediction_cache_time
    global _current_cache
    global _current_cache_time

    print("\n--- Warming up API caches (background thread) ---")

    # --------------------------------------------------------
    # Connect to Hopsworks ONCE
    # --------------------------------------------------------

    _hopsworks_project = connect_to_hopsworks()

    # --------------------------------------------------------
    # Warm CURRENT AQI cache
    # --------------------------------------------------------

    print("Loading latest AQI...")

    try:

        feature_row = get_latest_v6_row(
            _hopsworks_project
        )

        current_aqi = float(
            feature_row["target_aqi"].iloc[0]
        )

        _current_cache = {
            "timestamp": feature_row["timestamp"].iloc[0],
            "current_aqi": current_aqi,
            "alert": get_aqi_alert(current_aqi),
        }

        _current_cache_time = time.time()

        print("--- Current AQI cache ready ---")

    except Exception as exc:

        print(
            f"--- Current AQI cache warm-up failed: {exc} ---"
        )

        _current_cache = None
        _current_cache_time = 0

    # --------------------------------------------------------
    # Warm SAVED PREDICTION cache
    # --------------------------------------------------------

    print("Loading latest saved prediction...")

    try:

        _prediction_cache = read_latest_prediction(
            _hopsworks_project
        )

        _prediction_cache_time = time.time()

        print(
            "--- Latest saved prediction loaded successfully ---"
        )

    except Exception as exc:

        print(
            f"--- Prediction cache warm-up failed: {exc} ---"
        )

        _prediction_cache = None
        _prediction_cache_time = 0

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT warm history here.
    #
    # History is loaded only when /history is requested.
    # --------------------------------------------------------

    print(
        "--- History cache will load on first /history request ---"
    )

    print("--- API cache warm-up complete (background) ---\n")
# ============================================================
# BACKGROUND PREDICTION CACHE REFRESH
# ============================================================

def _background_prediction_refresh():
    """
    Refresh prediction cache every hour.

    API requests never wait for this refresh.
    """

    global _hopsworks_project
    global _prediction_cache
    global _prediction_cache_time

    while True:

        # Wait one hour before refreshing.
        time.sleep(CACHE_TTL)

        print("\n--- Hourly prediction cache refresh started ---")

        try:

            if _hopsworks_project is None:
                _hopsworks_project = connect_to_hopsworks()

            # Read latest prediction already saved by
            # automate_prediction.py
            new_prediction = read_latest_prediction(
                _hopsworks_project
            )

            _prediction_cache = new_prediction
            _prediction_cache_time = time.time()

            print(
                "--- Prediction cache refreshed successfully ---"
            )

        except Exception as exc:

            print(
                f"--- Prediction cache refresh failed: {exc} ---"
            )
threading.Thread(
    target=_background_prediction_refresh,
    daemon=True,
).start()

# ============================================================
# READ LATEST SAVED PREDICTION
# ============================================================

def read_latest_prediction(project):
    """
    Read the latest prediction already generated by the
    automated prediction pipeline.

    Feature Group:
        karachi_aqi_predictions v1

    No model loading or inference is performed here.
    """

    fs = project.get_feature_store()

    prediction_fg = fs.get_feature_group(
        name="karachi_aqi_predictions",
        version=1,
    )

    dataframe = prediction_fg.read(
        dataframe_type="pandas"
    )

    if dataframe is None or dataframe.empty:

        raise RuntimeError(
            "No saved predictions found in "
            "karachi_aqi_predictions v1."
        )

    dataframe = dataframe.copy()

    # --------------------------------------------------------
    # Normalize timestamp
    # --------------------------------------------------------

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["timestamp"]
    )

    if dataframe.empty:

        raise RuntimeError(
            "Prediction Feature Group contains "
            "no valid timestamps."
        )

    # --------------------------------------------------------
    # Latest prediction
    # --------------------------------------------------------

    latest = (
        dataframe
        .sort_values("timestamp")
        .iloc[-1]
    )

    day1 = float(latest["day1"])
    day2 = float(latest["day2"])
    day3 = float(latest["day3"])

    result = {
        "timestamp": latest["timestamp"],

        "day1": day1,
        "day1_alert": get_aqi_alert(day1),

        "day2": day2,
        "day2_alert": get_aqi_alert(day2),

        "day3": day3,
        "day3_alert": get_aqi_alert(day3),
    }

    # --------------------------------------------------------
    # Model versions
    # --------------------------------------------------------

    for key in [
        "day1_model_version",
        "day2_model_version",
        "day3_model_version",
    ]:

        if key in dataframe.columns:

            result[key] = int(
                latest[key]
            )

    print(
        f"Latest saved prediction timestamp: "
        f"{latest['timestamp']}"
    )

    return result


# ============================================================
# READ 90-DAY HISTORY
# ============================================================

def read_history(project):

    print(
        "Reading 90-day AQI history from Hopsworks..."
    )

    fs = project.get_feature_store()

    feature_group = fs.get_feature_group(
        name="karachi_aqi_features",
        version=6,
    )

    # --------------------------------------------------------
    # Required date range
    # --------------------------------------------------------

    now = pd.Timestamp.now(
        tz="UTC"
    )

    start_time = (
        now
        - pd.Timedelta(days=90)
    )

    # --------------------------------------------------------
    # Dashboard columns
    # --------------------------------------------------------

    history_columns = [
        "city",
        "timestamp",
        "target_aqi",
        "pm25",
        "pm10",
        "ozone",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "carbon_monoxide",
        "temperature",
        "humidity",
    ]

    # --------------------------------------------------------
    # Read history
    # --------------------------------------------------------

    dataframe = feature_group.read(
        start_time=start_time.to_pydatetime(),
        end_time=now.to_pydatetime(),
        dataframe_type="pandas",
    )

    if dataframe is None or dataframe.empty:

        return {
            "data": []
        }

    dataframe = dataframe.copy()

    # --------------------------------------------------------
    # Keep ONLY dashboard columns
    # --------------------------------------------------------

    available_columns = [
        column
        for column in history_columns
        if column in dataframe.columns
    ]

    dataframe = dataframe[
        available_columns
    ].copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # Karachi only
    # --------------------------------------------------------

    if "city" in dataframe.columns:

        dataframe = dataframe[
            dataframe["city"]
            .astype(str)
            .str.lower()
            == "karachi"
        ].copy()

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    dataframe = dataframe.sort_values(
        "timestamp"
    )

    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    dataframe["date"] = (
        dataframe["timestamp"]
        .dt.date
    )

    aggregation_columns = [
        column
        for column in [
            "target_aqi",
            "pm25",
            "pm10",
            "ozone",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "carbon_monoxide",
            "temperature",
            "humidity",
        ]
        if column in dataframe.columns
    ]

    daily = (
        dataframe
        .groupby(
            "date",
            as_index=False,
        )
        .agg(
            {
                column: "mean"
                for column in aggregation_columns
            }
        )
    )

    # --------------------------------------------------------
    # JSON-safe date
    # --------------------------------------------------------

    daily["date"] = (
        daily["date"]
        .astype(str)
    )

    daily = daily.where(
        pd.notna(daily),
        None,
    )

    return {
        "data": daily.to_dict(
            orient="records"
        )
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Karachi AQI Predictor API is running"
    }

@app.get("/predict")
def get_prediction():

    if _prediction_cache is None:

        return {
            "error": "Prediction is not available yet. Please try again shortly."
        }

    print(
        "Returning cached saved prediction."
    )

    return _prediction_cache
# ============================================================
# CURRENT AQI API
# ============================================================

@app.get("/current")
def get_current():

    global _current_cache
    global _current_cache_time
    global _hopsworks_project

    now = time.time()

    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------

    if (
        _current_cache is not None
        and now - _current_cache_time < CACHE_TTL
    ):

        print(
            "Returning cached current AQI."
        )

        return _current_cache

    # --------------------------------------------------------
    # CACHE MISS
    # --------------------------------------------------------

    print(
        "Reading fresh current AQI..."
    )

    if _hopsworks_project is None:

        _hopsworks_project = (
            connect_to_hopsworks()
        )

    feature_row = get_latest_v6_row(
        _hopsworks_project
    )

    current_aqi = float(
        feature_row[
            "target_aqi"
        ].iloc[0]
    )

    result = {
        "timestamp": feature_row[
            "timestamp"
        ].iloc[0],

        "current_aqi": current_aqi,

        "alert": get_aqi_alert(
            current_aqi
        ),
    }

    _current_cache = result
    _current_cache_time = now

    return result


# ============================================================
# HISTORY API
# ============================================================

@app.get("/history")
def get_history():

    global _history_cache
    global _history_cache_time
    global _hopsworks_project

    now = time.time()

    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------

    if (
        _history_cache is not None
        and now - _history_cache_time < CACHE_TTL
    ):

        print(
            "Returning cached history."
        )

        return _history_cache

    # --------------------------------------------------------
    # CACHE MISS
    # --------------------------------------------------------

    print(
        "History cache miss - loading from Hopsworks..."
    )

    if _hopsworks_project is None:

        _hopsworks_project = (
            connect_to_hopsworks()
        )

    result = read_history(
        _hopsworks_project
    )

    # --------------------------------------------------------
    # CACHE RESULT
    # --------------------------------------------------------

    _history_cache = result
    _history_cache_time = now

    print(
        "--- 90-day history cached successfully ---"
    )

    return result
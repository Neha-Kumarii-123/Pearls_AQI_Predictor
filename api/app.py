from fastapi import FastAPI
from src.predict import (
    predict,
    get_latest_v6_row,
    connect_to_hopsworks,
)
import pandas as pd
import time


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Karachi AQI Predictor"
)


# ============================================================
# CACHE CONFIGURATION
# ============================================================

# How long cached data remains valid.
# Prediction/current: 1 hour
# History: 1 hour
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


# ============================================================
# HOPSWORKS CONNECTION
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize Hopsworks and warm up API caches once.
    """

    global _prediction_cache
    global _prediction_cache_time
    global _current_cache
    global _current_cache_time

    print("\n--- Warming up API caches ---")

    # Connect once
    project = connect_to_hopsworks()

    # --------------------------------------------------------
    # Warm current AQI cache
    # --------------------------------------------------------

    print("Loading latest AQI...")

    feature_row = get_latest_v6_row(project)

    _current_cache = {
        "timestamp": feature_row["timestamp"].iloc[0],
        "current_aqi": float(
            feature_row["target_aqi"].iloc[0]
        ),
    }

    _current_cache_time = time.time()

    # --------------------------------------------------------
    # Warm prediction cache
    # --------------------------------------------------------

    print("Generating initial prediction...")

    _prediction_cache = predict()
    _prediction_cache_time = time.time()

    print("--- API cache warm-up complete ---\n")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Karachi AQI Predictor API is running"
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.get("/predict")
def get_prediction():

    global _prediction_cache
    global _prediction_cache_time

    now = time.time()

    # --------------------------------------------------------
    # Return cached prediction if still valid.
    # --------------------------------------------------------

    if (
        _prediction_cache is not None
        and now - _prediction_cache_time < CACHE_TTL
    ):

        print("Returning cached prediction.")

        return _prediction_cache

    # --------------------------------------------------------
    # Generate fresh prediction.
    # --------------------------------------------------------

    print("Generating fresh AQI prediction...")

    result = predict()

    # --------------------------------------------------------
    # Store result in cache.
    # --------------------------------------------------------

    _prediction_cache = result
    _prediction_cache_time = now

    return result


# ============================================================
# CURRENT AQI API
# ============================================================

@app.get("/current")
def get_current():

    global _current_cache
    global _current_cache_time

    now = time.time()

    # --------------------------------------------------------
    # Return cached current AQI.
    # --------------------------------------------------------

    if (
        _current_cache is not None
        and now - _current_cache_time < CACHE_TTL
    ):

        print("Returning cached current AQI.")

        return _current_cache

    # --------------------------------------------------------
    # Read latest feature row.
    # --------------------------------------------------------

    print("Reading fresh current AQI...")

    project = connect_to_hopsworks()

    feature_row = get_latest_v6_row(
        project
    )

    result = {
        "timestamp": feature_row[
            "timestamp"
        ].iloc[0],
        "current_aqi": float(
            feature_row[
                "target_aqi"
            ].iloc[0]
        ),
    }

    # --------------------------------------------------------
    # Store current AQI in cache.
    # --------------------------------------------------------

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

    now_time = time.time()

    # --------------------------------------------------------
    # Return cached history.
    # --------------------------------------------------------

    if (
        _history_cache is not None
        and now_time - _history_cache_time < CACHE_TTL
    ):

        print("Returning cached history.")

        return _history_cache

    # --------------------------------------------------------
    # Connect to Hopsworks.
    # --------------------------------------------------------

    print("Reading fresh 90-day AQI history...")

    project = connect_to_hopsworks()

    fs = project.get_feature_store()

    feature_group = fs.get_feature_group(
        name="karachi_aqi_features",
        version=6,
    )

    # --------------------------------------------------------
    # Read only the required 90-day window.
    # --------------------------------------------------------

    now = pd.Timestamp.now(
        tz="UTC"
    )

    start_time = (
        now
        - pd.Timedelta(days=90)
    )

    dataframe = feature_group.read(
        start_time=start_time.to_pydatetime(),
        end_time=now.to_pydatetime(),
        dataframe_type="pandas",
    )

    if dataframe is None or dataframe.empty:

        result = {
            "data": []
        }

        _history_cache = result
        _history_cache_time = now_time

        return result

    dataframe = dataframe.copy()

    # --------------------------------------------------------
    # Timestamp conversion.
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
    # Karachi only.
    # --------------------------------------------------------

    dataframe = dataframe[
        dataframe["city"]
        .astype(str)
        .str.lower()
        == "karachi"
    ].copy()

    # --------------------------------------------------------
    # Required columns only.
    # --------------------------------------------------------

    history_columns = [
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

    available_columns = [
        column
        for column in history_columns
        if column in dataframe.columns
    ]

    dataframe = dataframe[
        available_columns
    ].sort_values(
        "timestamp"
    )

    # --------------------------------------------------------
    # Daily aggregation.
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

    daily["date"] = (
        daily["date"]
        .astype(str)
    )

    daily = daily.where(
        pd.notna(daily),
        None,
    )

    result = {
        "data": daily.to_dict(
            orient="records"
        )
    }

    # --------------------------------------------------------
    # Store history in cache.
    # --------------------------------------------------------

    _history_cache = result
    _history_cache_time = now_time

    return result


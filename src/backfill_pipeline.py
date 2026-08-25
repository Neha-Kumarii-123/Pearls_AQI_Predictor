"""
Historical backfill pipeline for Karachi AQI Predictor.

Purpose:
    Fetch historical raw weather + air-quality data from Open-Meteo
    and store the canonical raw dataset in Hopsworks Feature Group v5.

Important:
    Feature engineering is NOT performed here.

    The shared feature_engineering.py is responsible for converting
    these raw columns into the canonical 100 MODEL_FEATURES during
    training and live inference.
"""

from __future__ import annotations

import os

import hopsworks
import openmeteo_requests
import pandas as pd
import requests_cache
from dotenv import load_dotenv
from retry_requests import retry


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------
# Open-Meteo client
# ---------------------------------------------------------------------

cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=-1,
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2,
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)


# ---------------------------------------------------------------------
# Karachi configuration
# ---------------------------------------------------------------------

KARACHI_LATITUDE = 24.8607
KARACHI_LONGITUDE = 67.0011

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


# ---------------------------------------------------------------------
# Hopsworks configuration
# ---------------------------------------------------------------------

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 5

PRIMARY_KEY = [
    "city",
    "timestamp",
]


# ---------------------------------------------------------------------
# Canonical raw schema
# ---------------------------------------------------------------------

RAW_COLUMNS = (
    "city",
    "timestamp",
    "pm25",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
    "temperature",
    "humidity",
    "target_aqi",
)


# ---------------------------------------------------------------------
# Historical range
# ---------------------------------------------------------------------

START_DATE = "2024-08-01"
END_DATE = "2026-08-01"


# ---------------------------------------------------------------------
# Fetch Air Quality
# ---------------------------------------------------------------------

def fetch_air_quality(
    latitude: float = KARACHI_LATITUDE,
    longitude: float = KARACHI_LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:

    print(
        f"\nFetching air-quality data: "
        f"{start_date} → {end_date}"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": [
            "pm10",
            "pm2_5",
            "ozone",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "carbon_monoxide",
            "us_aqi",
        ],
    }

    responses = openmeteo.weather_api(
        AIR_QUALITY_URL,
        params=params,
    )

    response = responses[0]
    hourly = response.Hourly()

    timestamps = pd.date_range(
        start=pd.to_datetime(
            hourly.Time(),
            unit="s",
            utc=True,
        ),
        end=pd.to_datetime(
            hourly.TimeEnd(),
            unit="s",
            utc=True,
        ),
        freq=pd.Timedelta(
            seconds=hourly.Interval()
        ),
        inclusive="left",
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pm25": hourly.Variables(1).ValuesAsNumpy(),
            "pm10": hourly.Variables(0).ValuesAsNumpy(),
            "ozone": hourly.Variables(2).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(3).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(4).ValuesAsNumpy(),
            "carbon_monoxide": hourly.Variables(5).ValuesAsNumpy(),
            "target_aqi": hourly.Variables(6).ValuesAsNumpy(),
        }
    )

    print(
        f"Air-quality observations: {len(df)}"
    )

    return df


# ---------------------------------------------------------------------
# Fetch Weather
# ---------------------------------------------------------------------

def fetch_weather(
    latitude: float = KARACHI_LATITUDE,
    longitude: float = KARACHI_LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:

    print(
        f"\nFetching weather data: "
        f"{start_date} → {end_date}"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
        ],
    }

    responses = openmeteo.weather_api(
        WEATHER_URL,
        params=params,
    )

    response = responses[0]
    hourly = response.Hourly()

    timestamps = pd.date_range(
        start=pd.to_datetime(
            hourly.Time(),
            unit="s",
            utc=True,
        ),
        end=pd.to_datetime(
            hourly.TimeEnd(),
            unit="s",
            utc=True,
        ),
        freq=pd.Timedelta(
            seconds=hourly.Interval()
        ),
        inclusive="left",
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
        }
    )

    print(
        f"Weather observations: {len(df)}"
    )

    return df


# ---------------------------------------------------------------------
# Build canonical raw dataframe
# ---------------------------------------------------------------------

def build_raw_dataframe(
    air_quality: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:

    print("\nMerging air-quality and weather data...")

    df = pd.merge(
        air_quality,
        weather,
        on="timestamp",
        how="inner",
    )

    if df.empty:
        raise RuntimeError(
            "No overlapping timestamps between "
            "air-quality and weather data."
        )

    df["city"] = "karachi"

    df = df[
        [
            "city",
            "timestamp",
            "pm25",
            "pm10",
            "ozone",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "carbon_monoxide",
            "temperature",
            "humidity",
            "target_aqi",
        ]
    ].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    numeric_columns = [
        column
        for column in RAW_COLUMNS
        if column not in ("city", "timestamp")
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).astype("float64")

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# Validate raw dataframe
# ---------------------------------------------------------------------

def validate_raw_dataframe(
    df: pd.DataFrame,
) -> None:

    print("\nValidating canonical raw dataframe...")

    # Column check
    missing_columns = [
        column
        for column in RAW_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Timestamp check
    if df["timestamp"].isna().any():
        raise RuntimeError(
            "Invalid timestamp values detected."
        )

    if df["timestamp"].duplicated().any():
        raise RuntimeError(
            "Duplicate timestamps detected."
        )

    # Numeric missing values
    numeric_columns = [
        column
        for column in RAW_COLUMNS
        if column not in ("city", "timestamp")
    ]

    missing_counts = (
        df[numeric_columns]
        .isna()
        .sum()
    )

    missing = missing_counts[
        missing_counts > 0
    ]

    if not missing.empty:
        details = ", ".join(
            f"{column}={int(count)}"
            for column, count in missing.items()
        )

        raise RuntimeError(
            "Missing values detected: "
            + details
        )

    # Hourly continuity
    intervals = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    if not intervals.empty:

        if not intervals.eq(
            pd.Timedelta(hours=1)
        ).all():

            raise RuntimeError(
                "Dataset is not a continuous hourly "
                "time series."
            )

    print(
        "Validation passed."
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        "Missing values: 0"
    )


# ---------------------------------------------------------------------
# Upload to Hopsworks
# ---------------------------------------------------------------------

def upload_to_hopsworks(
    dataframe: pd.DataFrame,
) -> None:

    print(
        "\nConnecting to Hopsworks..."
    )

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is not set."
        )

    project = hopsworks.login(
        api_key_value=api_key
    )

    fs = project.get_feature_store()

    print(
        f"Accessing Feature Group "
        f"{FEATURE_GROUP_NAME} "
        f"v{FEATURE_GROUP_VERSION}..."
    )

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=PRIMARY_KEY,
        event_time="timestamp",
        description=(
            "Canonical raw hourly Karachi AQI "
            "weather and pollutant observations. "
            "Feature engineering is performed "
            "outside the Feature Group."
        ),
        online_enabled=True,
    )

    batch_size = 3000

    print(
        f"\nUploading {len(dataframe)} rows..."
    )

    for start in range(
        0,
        len(dataframe),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(dataframe),
        )

        batch = dataframe.iloc[
            start:end
        ].copy()

        print(
            f"Inserting rows "
            f"{start} → {end}"
        )

        feature_group.insert(
            batch,
            write_options={
                "wait_for_job": True,
            },
        )

    print(
        "\nHistorical backfill completed successfully."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print(
        "\n=============================================="
    )
    print(
        " KARACHI AQI HISTORICAL BACKFILL — v5"
    )
    print(
        "=============================================="
    )

    # 1. Fetch
    air_quality = fetch_air_quality()

    weather = fetch_weather()

    # 2. Build canonical raw dataset
    raw_df = build_raw_dataframe(
        air_quality,
        weather,
    )

    # 3. Validate
    validate_raw_dataframe(
        raw_df
    )

    # 4. Display sample
    print(
        "\n--- Canonical Raw Data Sample ---"
    )

    print(
        raw_df.head()
    )

    print(
        "\nCanonical columns:"
    )

    print(
        list(raw_df.columns)
    )

    print(
        "\n--- Uploading to Hopsworks v5 ---"
    )

    # 5. Upload
    upload_to_hopsworks(
        raw_df
    )

    print(
        "\n=============================================="
    )
    print(
        " BACKFILL SUCCESS"
    )
    print(
        " Feature Group: karachi_aqi_features v5"
    )
    print(
        f" Rows: {len(raw_df)}"
    )
    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
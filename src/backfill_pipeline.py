
"""
Historical feature backfill pipeline for Karachi AQI Predictor — v6.

Purpose:
    Fetch historical raw weather + air-quality data directly from
    Open-Meteo, transform it using the project's shared
    feature_engineering.py, and store the canonical computed
    100 MODEL_FEATURES in Hopsworks Feature Group v6.

Important:
    - v5 remains untouched.
    - v6 does NOT read from v5.
    - v6 does NOT store raw data.
    - All feature calculations come from the existing
      shared feature_engineering.py.
    - The canonical 100 MODEL_FEATURES are unchanged.
"""

from __future__ import annotations

import os

import hopsworks
import openmeteo_requests
import pandas as pd
import requests_cache
from dotenv import load_dotenv
from retry_requests import retry

from feature_engineering import (
    MODEL_FEATURES,
    REQUIRED_RAW_COLUMNS,
    MAX_LOOKBACK_HOURS,
    build_rich_features,
    validate_feature_frame,
)


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
FEATURE_GROUP_VERSION = 6

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
#
# Same historical period as v5 so that the resulting feature
# distribution remains directly comparable.
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

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------
# Validate canonical raw dataframe
# ---------------------------------------------------------------------

def validate_raw_dataframe(
    df: pd.DataFrame,
) -> None:

    print("\nValidating canonical raw dataframe...")

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

    if df["timestamp"].isna().any():
        raise RuntimeError(
            "Invalid timestamp values detected."
        )

    if df["timestamp"].duplicated().any():
        raise RuntimeError(
            "Duplicate timestamps detected."
        )

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

    if len(df) <= MAX_LOOKBACK_HOURS:
        raise RuntimeError(
            "Not enough historical observations for "
            "feature engineering."
        )

    print("Raw validation passed.")

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
# Build canonical computed features
# ---------------------------------------------------------------------

def build_computed_features(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n==============================================")
    print(" RUNNING SHARED FEATURE ENGINEERING")
    print("==============================================")

    features = build_rich_features(
        raw_df
    )

    print(
        "\nGenerated feature frame shape:",
        features.shape
    )

    print(
        "Expected MODEL_FEATURES:",
        len(MODEL_FEATURES)
    )

    if len(MODEL_FEATURES) != 100:
        raise RuntimeError(
            "Expected exactly 100 MODEL_FEATURES."
        )

    actual_model_features = [
        column
        for column in features.columns
        if column in MODEL_FEATURES
    ]

    if actual_model_features != list(
        MODEL_FEATURES
    ):
        raise RuntimeError(
            "Generated feature order does not "
            "match canonical MODEL_FEATURES."
        )

    print(
        "100-feature schema check: PASSED"
    )

    # -------------------------------------------------------------
    # Remove warm-up rows.
    #
    # The feature engineering requires up to 168 hours of history.
    # -------------------------------------------------------------

    complete_features = features.dropna(
        subset=list(MODEL_FEATURES)
    ).copy()

    if complete_features.empty:
        raise RuntimeError(
            "No complete feature rows were produced."
        )

    print(
        "Complete rows after warm-up:",
        len(complete_features)
    )

    # -------------------------------------------------------------
    # Validate complete feature frame.
    # -------------------------------------------------------------

    validate_feature_frame(
        complete_features,
        require_complete=True,
    )

    print(
        "Canonical feature validation: PASSED"
    )

    # -------------------------------------------------------------
    # Store only the canonical computed features plus identifiers.
    #
    # target_aqi is intentionally retained because it is part of
    # the project's existing historical source information and
    # feature-engineering context. It is NOT part of MODEL_FEATURES.
    # -------------------------------------------------------------

    output_columns = [
        "city",
        "timestamp",
        "target_aqi",
    ] + list(MODEL_FEATURES)

    computed = complete_features[
        output_columns
    ].copy()

    # -------------------------------------------------------------
    # Final safety checks.
    # -------------------------------------------------------------

    if len(MODEL_FEATURES) != 100:
        raise RuntimeError(
            "Canonical feature count changed."
        )

    if computed[list(MODEL_FEATURES)].isna().any().any():
        raise RuntimeError(
            "Computed feature dataset contains NaN values."
        )

    actual_feature_order = [
        column
        for column in computed.columns
        if column in MODEL_FEATURES
    ]

    if actual_feature_order != list(
        MODEL_FEATURES
    ):
        raise RuntimeError(
            "Computed feature order does not match "
            "MODEL_FEATURES."
        )

    print(
        "Final computed feature dataset shape:",
        computed.shape
    )

    print(
        "Final computed MODEL_FEATURES:",
        len(MODEL_FEATURES)
    )

    return computed


# ---------------------------------------------------------------------
# Upload computed features to Hopsworks v6
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
        f"Creating/accessing Feature Group "
        f"{FEATURE_GROUP_NAME} "
        f"v{FEATURE_GROUP_VERSION}..."
    )

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=PRIMARY_KEY,
        event_time="timestamp",
        description=(
            "Canonical computed hourly Karachi AQI "
            "features generated by the shared "
            "feature_engineering.py pipeline. "
            "Contains the project's canonical "
            "100 MODEL_FEATURES."
        ),
        online_enabled=True,
    )

    # Smaller batches reduce the chance of temporary
    # Hopsworks connection failures during large uploads.
    batch_size = 1000

    # Retry only the individual batch that fails.
    max_retries = 3

    print(
        f"\nUploading {len(dataframe)} computed rows..."
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
            f"\nInserting rows "
            f"{start} → {end}"
        )

        for attempt in range(
            1,
            max_retries + 1,
        ):

            try:

                print(
                    f"Upload attempt "
                    f"{attempt}/{max_retries}"
                )

                feature_group.insert(
                    batch,
                    write_options={
                        "wait_for_job": True,
                    },
                )

                print(
                    f"Batch {start} → {end} "
                    f"uploaded successfully."
                )

                break

            except Exception as error:

                print(
                    f"Upload attempt {attempt} "
                    f"failed: {error}"
                )

                if attempt == max_retries:

                    raise RuntimeError(
                        f"Failed to upload batch "
                        f"{start} → {end} after "
                        f"{max_retries} attempts."
                    ) from error

                print(
                    "Retrying this batch..."
                )

    print(
        "\nComputed feature backfill uploaded successfully."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print(
        "\n=============================================="
    )

    print(
        " KARACHI AQI COMPUTED FEATURE BACKFILL — v6"
    )

    print(
        "=============================================="
    )

    # -------------------------------------------------------------
    # 1. Fetch raw data DIRECTLY from Open-Meteo
    # -------------------------------------------------------------

    air_quality = fetch_air_quality()

    weather = fetch_weather()

    # -------------------------------------------------------------
    # 2. Build canonical raw dataframe
    # -------------------------------------------------------------

    raw_df = build_raw_dataframe(
        air_quality,
        weather,
    )

    # -------------------------------------------------------------
    # 3. Validate raw data
    # -------------------------------------------------------------

    validate_raw_dataframe(
        raw_df
    )

    # -------------------------------------------------------------
    # 4. Display raw sample
    # -------------------------------------------------------------

    print(
        "\n--- Raw Data Sample ---"
    )

    print(
        raw_df.head()
    )

    print(
        "\nRaw columns:"
    )

    print(
        list(raw_df.columns)
    )

    # -------------------------------------------------------------
    # 5. Transform raw → canonical 100 features
    # -------------------------------------------------------------

    computed_features = build_computed_features(
        raw_df
    )

    # -------------------------------------------------------------
    # 6. Display computed feature sample
    # -------------------------------------------------------------

    print(
        "\n--- Computed Feature Sample ---"
    )

    print(
        computed_features.head()
    )

    print(
        "\nComputed feature columns:"
    )

    print(
        list(computed_features.columns)
    )

    # -------------------------------------------------------------
    # 7. Upload ONLY computed features to v6
    # -------------------------------------------------------------

    print(
        "\n--- Uploading COMPUTED FEATURES to Hopsworks v6 ---"
    )

    upload_to_hopsworks(
        computed_features
    )

    # -------------------------------------------------------------
    # Success
    # -------------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        " V6 COMPUTED FEATURE BACKFILL SUCCESS"
    )

    print(
        f" Feature Group: "
        f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
    )

    print(
        f" Computed rows: "
        f"{len(computed_features)}"
    )

    print(
        f" MODEL_FEATURES stored: "
        f"{len(MODEL_FEATURES)}"
    )

    print(
        " Source: Open-Meteo API"
    )

    print(
        " Transformation: shared feature_engineering.py"
    )

    print(
        " v5: untouched"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()


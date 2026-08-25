import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from time import sleep

import requests
import pandas as pd


# ---------------------------------------------------------
# Project root / src import
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from src.feature_engineering import (  # noqa: E402
    MODEL_FEATURES,
    REQUIRED_RAW_COLUMNS,
    MAX_LOOKBACK_HOURS,
    build_rich_features,
    validate_feature_frame,
)


LAT = 24.8607
LON = 67.0011


def main():

    print("\n==============================================")
    print(" LIVE FEATURE ENGINEERING COMPATIBILITY TEST")
    print("==============================================")

    # ---------------------------------------------------------
    # 1. Request enough historical data
    # ---------------------------------------------------------

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=200)

    common = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }

    # ---------------------------------------------------------
    # 2. Air Quality
    # ---------------------------------------------------------

    print("\n--- Fetching Air Quality ---")

    aq_params = {
        **common,
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

    aq_response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params=aq_params,
        timeout=60,
    )

    print("AQ HTTP:", aq_response.status_code)
    aq_response.raise_for_status()

    aq = aq_response.json()["hourly"]

    print("AQ hours:", len(aq["time"]))

    # ---------------------------------------------------------
    # 3. Weather
    # ---------------------------------------------------------

    print("\n--- Fetching Weather ---")

    weather_params = {
        **common,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
        ],
    }

    weather = None

    for attempt in range(3):

        try:

            weather_response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=weather_params,
                timeout=60,
            )

            print("Weather HTTP:", weather_response.status_code)

            weather_response.raise_for_status()

            weather = weather_response.json()["hourly"]

            print("Weather hours:", len(weather["time"]))

            break

        except requests.RequestException as exc:

            print(
                f"Weather attempt {attempt + 1} failed:",
                exc
            )

            if attempt == 2:
                raise

            sleep(5)

    # ---------------------------------------------------------
    # 4. Build raw DataFrames
    # ---------------------------------------------------------

    print("\n--- Combining Raw Data ---")

    aq_df = pd.DataFrame(aq)
    weather_df = pd.DataFrame(weather)

    aq_df["time"] = pd.to_datetime(
        aq_df["time"],
        utc=True
    )

    weather_df["time"] = pd.to_datetime(
        weather_df["time"],
        utc=True
    )

    df = aq_df.merge(
        weather_df,
        on="time",
        how="inner",
    )

    # ---------------------------------------------------------
    # 5. Convert to canonical raw schema
    # ---------------------------------------------------------

    df = df.rename(
        columns={
            "time": "timestamp",
            "pm2_5": "pm25",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "us_aqi": "target_aqi",
        }
    )

    print("\nRaw merged shape:", df.shape)

    print("\nRaw columns:")
    print(list(df.columns))

    # ---------------------------------------------------------
    # 6. Verify required raw columns
    # ---------------------------------------------------------

    print("\n--- Checking Required Raw Columns ---")

    missing_raw = [
        column
        for column in REQUIRED_RAW_COLUMNS
        if column not in df.columns
    ]

    if missing_raw:
        raise RuntimeError(
            "Missing required raw columns: "
            + ", ".join(missing_raw)
        )

    print("All required raw columns are present.")

    # ---------------------------------------------------------
    # 7. Verify missing values
    # ---------------------------------------------------------

    print("\n--- Raw Missing Values ---")

    raw_missing = df[
        list(REQUIRED_RAW_COLUMNS)
    ].isna().sum()

    print(raw_missing)

    if raw_missing.sum() > 0:

        raise RuntimeError(
            "Raw data contains missing values."
        )

    print("Raw data contains no missing values.")

    # ---------------------------------------------------------
    # 8. Verify hourly continuity
    # ---------------------------------------------------------

    print("\n--- Checking Hourly Continuity ---")

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    intervals = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    continuous = intervals.eq(
        pd.Timedelta(hours=1)
    ).all()

    print("Hourly continuity:", continuous)

    if not continuous:

        raise RuntimeError(
            "Raw data is not a continuous hourly series."
        )

    # ---------------------------------------------------------
    # 9. Verify sufficient history
    # ---------------------------------------------------------

    print("\n--- Checking Historical Lookback ---")

    print(
        "Required maximum lookback:",
        MAX_LOOKBACK_HOURS,
        "hours"
    )

    print(
        "Available raw observations:",
        len(df)
    )

    if len(df) <= MAX_LOOKBACK_HOURS:

        raise RuntimeError(
            "Not enough historical observations "
            "for the canonical feature pipeline."
        )

    print("Sufficient historical context available.")

    # ---------------------------------------------------------
    # 10. Run SHARED feature engineering
    # ---------------------------------------------------------

    print("\n==============================================")
    print(" Running SHARED build_rich_features()")
    print("==============================================")

    features = build_rich_features(df)

    print("\nFeature frame shape:")
    print(features.shape)

    print("\nTotal MODEL_FEATURES:")
    print(len(MODEL_FEATURES))

    # ---------------------------------------------------------
    # 11. Validate canonical feature frame
    # ---------------------------------------------------------

    print("\n--- Validating Canonical Feature Frame ---")

    # The first part of the historical frame is expected to contain
    # warm-up NaNs because the feature pipeline uses up to 168 hours
    # of historical lag/rolling features.

    complete_features = features.dropna(
        subset=list(MODEL_FEATURES)
    ).copy()

    if complete_features.empty:
        raise RuntimeError(
            "No complete feature row was produced after historical warm-up."
        )

    latest_features = complete_features.iloc[[-1]].copy()

    print(
        "Complete feature rows after warm-up:",
        len(complete_features)
    )

    print(
        "Latest complete timestamp:",
        latest_features["timestamp"].iloc[0]
    )

    validate_feature_frame(
        latest_features,
        require_complete=True,
    )

    print(
        "Canonical latest-row validation: PASSED"
    )

   
    print(
        "Canonical feature validation: PASSED"
    )

    # ---------------------------------------------------------
    # 12. Inspect latest feature row
    # ---------------------------------------------------------

    latest = latest_features.iloc[0]

    print("\n==============================================")
    print(" LATEST PRODUCTION FEATURE ROW")
    print("==============================================")

    print(
        "Timestamp:",
        latest["timestamp"]
    )

    print(
        "Number of model features:",
        len(MODEL_FEATURES)
    )

    print(
        "Latest row NaN count:",
        latest[list(MODEL_FEATURES)].isna().sum()
    )

    # ---------------------------------------------------------
    # 13. Final checks
    # ---------------------------------------------------------

    if latest[list(MODEL_FEATURES)].isna().any():

        raise RuntimeError(
            "Latest production feature row contains NaN values."
        )

    actual_feature_order = [
        column
        for column in features.columns
        if column in MODEL_FEATURES
    ]

    if actual_feature_order != list(MODEL_FEATURES):

        raise RuntimeError(
            "MODEL_FEATURES order does not match canonical contract."
        )

    # ---------------------------------------------------------
    # SUCCESS
    # ---------------------------------------------------------

    print("\n==============================================")
    print(" LIVE FEATURE COMPATIBILITY TEST PASSED")
    print("==============================================")

    print(
        "\nThe live raw-data structure is compatible with"
    )
    print(
        "the same feature engineering pipeline used by training."
    )

    print(
        "\nNext step: integrate this logic into"
    )
    print(
        "src/feature_pipeline.py"
    )


if __name__ == "__main__":
    main()
    
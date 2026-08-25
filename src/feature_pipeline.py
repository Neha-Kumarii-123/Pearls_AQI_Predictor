"""
Production feature pipeline for the Karachi AQI Predictor.

Production flow:

    Open-Meteo historical data
        ↓
    Canonical raw dataframe
        ↓
    shared build_rich_features()
        ↓
    100 MODEL_FEATURES
        ↓
    latest complete feature row
        ↓
    Hopsworks Feature Store

Important:
    This file is an orchestration layer only.

    All feature engineering must remain inside
    src/feature_engineering.py so that training and
    production use exactly the same feature definitions.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

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
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Karachi configuration
# ---------------------------------------------------------------------

KARACHI_LATITUDE = 24.8607
KARACHI_LONGITUDE = 67.0011

CITY = "karachi"

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


# ---------------------------------------------------------------------
# Hopsworks configuration
# ---------------------------------------------------------------------

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 5

PRIMARY_KEY = ["city", "timestamp"]


# ---------------------------------------------------------------------
# Production lookback
# ---------------------------------------------------------------------

# The canonical feature engineering currently needs up to 168 hours.
#
# We intentionally request more than the absolute minimum so that:
#
#     168 hours
#          +
#     warm-up / boundary safety
#
# gives us a reliable latest complete row.
#
# The compatibility test successfully used 200 hours.

LOOKBACK_HOURS = 200


# ---------------------------------------------------------------------
# HTTP configuration
# ---------------------------------------------------------------------

REQUEST_TIMEOUT = 60

MAX_WEATHER_RETRIES = 3


class FeaturePipelineError(RuntimeError):
    """Raised when production feature generation fails."""


class AirQualityFeaturePipeline:
    """
    Production orchestrator for Karachi AQI feature generation.

    This class deliberately does NOT implement feature engineering
    itself. It delegates that responsibility to build_rich_features()
    from feature_engineering.py.
    """

    def __init__(
        self,
        city: str = CITY,
        latitude: float = KARACHI_LATITUDE,
        longitude: float = KARACHI_LONGITUDE,
        lookback_hours: int = LOOKBACK_HOURS,
    ) -> None:

        self.city = city
        self.latitude = latitude
        self.longitude = longitude
        self.lookback_hours = max(
            lookback_hours,
            MAX_LOOKBACK_HOURS + 1,
        )

        logger.info(
            "Initialized AirQualityFeaturePipeline "
            "for city=%s, lookback=%s hours",
            self.city,
            self.lookback_hours,
        )

    # -----------------------------------------------------------------
    # 1. Build historical request window
    # -----------------------------------------------------------------

    def _get_request_window(self) -> tuple[str, str]:
        """
        Return UTC start/end dates for the Open-Meteo requests.

        Open-Meteo archive endpoints operate using calendar dates,
        while our canonical feature engineering operates on hourly
        UTC timestamps.
        """

        end = datetime.now(timezone.utc)

        start = end - timedelta(
            hours=self.lookback_hours
        )

        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")

        return start_date, end_date

    # -----------------------------------------------------------------
    # 2. Fetch Air Quality
    # -----------------------------------------------------------------

    def fetch_air_quality(self) -> pd.DataFrame:
        """
        Fetch historical hourly air-quality observations.

        Required production fields:

            pm10
            pm2_5
            ozone
            nitrogen_dioxide
            sulphur_dioxide
            carbon_monoxide
            us_aqi
        """

        start_date, end_date = self._get_request_window()

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
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

        logger.info(
            "Fetching historical air-quality data: "
            "%s → %s",
            start_date,
            end_date,
        )

        try:
            response = requests.get(
                AIR_QUALITY_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            logger.info(
                "Open-Meteo air-quality HTTP status: %s",
                response.status_code,
            )

            response.raise_for_status()

            payload = response.json()

            hourly = payload.get("hourly")

            if not hourly:
                raise FeaturePipelineError(
                    "Open-Meteo air-quality response "
                    "does not contain an hourly block."
                )

            required_api_columns = [
                "time",
                "pm10",
                "pm2_5",
                "ozone",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "carbon_monoxide",
                "us_aqi",
            ]

            missing = [
                column
                for column in required_api_columns
                if column not in hourly
            ]

            if missing:
                raise FeaturePipelineError(
                    "Air-quality response is missing fields: "
                    + ", ".join(missing)
                )

            df = pd.DataFrame(hourly)

            df["time"] = pd.to_datetime(
                df["time"],
                utc=True,
                errors="coerce",
            )

            if df["time"].isna().any():
                raise FeaturePipelineError(
                    "Air-quality response contains invalid timestamps."
                )

            logger.info(
                "Air-quality observations received: %s",
                len(df),
            )

            return df

        except requests.RequestException as exc:
            raise FeaturePipelineError(
                f"Air-quality API request failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # 3. Fetch Weather
    # -----------------------------------------------------------------

    def fetch_weather(self) -> pd.DataFrame:
        """
        Fetch historical hourly temperature and humidity.
        """

        start_date, end_date = self._get_request_window()

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
            ],
        }

        logger.info(
            "Fetching historical weather data: "
            "%s → %s",
            start_date,
            end_date,
        )

        last_exception: Optional[Exception] = None

        for attempt in range(1, MAX_WEATHER_RETRIES + 1):

            try:

                response = requests.get(
                    WEATHER_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                logger.info(
                    "Open-Meteo weather HTTP status: %s "
                    "(attempt %s/%s)",
                    response.status_code,
                    attempt,
                    MAX_WEATHER_RETRIES,
                )

                response.raise_for_status()

                payload = response.json()

                hourly = payload.get("hourly")

                if not hourly:
                    raise FeaturePipelineError(
                        "Open-Meteo weather response "
                        "does not contain an hourly block."
                    )

                required_api_columns = [
                    "time",
                    "temperature_2m",
                    "relative_humidity_2m",
                ]

                missing = [
                    column
                    for column in required_api_columns
                    if column not in hourly
                ]

                if missing:
                    raise FeaturePipelineError(
                        "Weather response is missing fields: "
                        + ", ".join(missing)
                    )

                df = pd.DataFrame(hourly)

                df["time"] = pd.to_datetime(
                    df["time"],
                    utc=True,
                    errors="coerce",
                )

                if df["time"].isna().any():
                    raise FeaturePipelineError(
                        "Weather response contains invalid timestamps."
                    )

                logger.info(
                    "Weather observations received: %s",
                    len(df),
                )

                return df

            except (
                requests.RequestException,
                FeaturePipelineError,
            ) as exc:

                last_exception = exc

                logger.warning(
                    "Weather request attempt %s failed: %s",
                    attempt,
                    exc,
                )

                if attempt < MAX_WEATHER_RETRIES:
                    # Keep retry logic simple and dependency-free.
                    import time

                    time.sleep(5)

        raise FeaturePipelineError(
            "Weather API failed after "
            f"{MAX_WEATHER_RETRIES} attempts: "
            f"{last_exception}"
        )

    # -----------------------------------------------------------------
    # 4. Build canonical raw dataframe
    # -----------------------------------------------------------------

    def build_raw_dataframe(
        self,
        air_quality: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Open-Meteo air-quality and weather data and convert
        them to the exact raw schema expected by feature_engineering.py.
        """

        logger.info(
            "Combining air-quality and weather observations..."
        )

        aq = air_quality.copy()
        wx = weather.copy()

        df = aq.merge(
            wx,
            on="time",
            how="inner",
        )

        if df.empty:
            raise FeaturePipelineError(
                "Air-quality and weather data have no "
                "overlapping hourly timestamps."
            )

        # -------------------------------------------------------------
        # Canonical raw schema
        # -------------------------------------------------------------

        df = df.rename(
            columns={
                "time": "timestamp",
                "pm2_5": "pm25",
                "temperature_2m": "temperature",
                "relative_humidity_2m": "humidity",
                "us_aqi": "target_aqi",
            }
        )

        # Explicitly attach city only after feature engineering if
        # desired. city is metadata, not a required raw feature.
        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        logger.info(
            "Merged raw dataframe shape: %s",
            df.shape,
        )

        logger.info(
            "Canonical raw columns: %s",
            list(df.columns),
        )

        return df

    # -----------------------------------------------------------------
    # 5. Validate raw dataframe
    # -----------------------------------------------------------------

    def validate_raw_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate the raw production dataframe before feature
        engineering.

        No artificial defaults or silent filling are performed here.
        """

        missing_columns = [
            column
            for column in REQUIRED_RAW_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise FeaturePipelineError(
                "Production raw dataframe is missing required columns: "
                + ", ".join(missing_columns)
            )

        # -------------------------------------------------------------
        # Timestamp validation
        # -------------------------------------------------------------

        timestamps = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

        if timestamps.isna().any():
            raise FeaturePipelineError(
                "Production raw dataframe contains invalid timestamps."
            )

        if timestamps.duplicated().any():
            raise FeaturePipelineError(
                "Production raw dataframe contains duplicate timestamps."
            )

        # -------------------------------------------------------------
        # Sort
        # -------------------------------------------------------------

        df["timestamp"] = timestamps

        df.sort_values(
            "timestamp",
            inplace=True,
        )

        df.reset_index(
            drop=True,
            inplace=True,
        )

        # -------------------------------------------------------------
        # Numeric validation
        # -------------------------------------------------------------

        numeric_columns = [
            column
            for column in REQUIRED_RAW_COLUMNS
            if column != "timestamp"
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # -------------------------------------------------------------
        # Missing-value validation
        # -------------------------------------------------------------

        missing_counts = df[
            list(REQUIRED_RAW_COLUMNS)
        ].isna().sum()

        missing = missing_counts[
            missing_counts > 0
        ]

        if not missing.empty:

            details = ", ".join(
                f"{column}={int(count)}"
                for column, count in missing.items()
            )

            raise FeaturePipelineError(
                "Production raw data contains missing values: "
                + details
            )

        # -------------------------------------------------------------
        # Hourly continuity
        # -------------------------------------------------------------

        intervals = (
            df["timestamp"]
            .diff()
            .dropna()
        )

        if not intervals.empty:

            continuous = intervals.eq(
                pd.Timedelta(hours=1)
            ).all()

            if not continuous:
                raise FeaturePipelineError(
                    "Production raw data is not a continuous "
                    "hourly time series."
                )

        # -------------------------------------------------------------
        # Historical warm-up
        # -------------------------------------------------------------

        if len(df) <= MAX_LOOKBACK_HOURS:

            raise FeaturePipelineError(
                "Insufficient historical observations. "
                f"Need more than {MAX_LOOKBACK_HOURS} rows, "
                f"received {len(df)}."
            )

        logger.info(
            "Raw-data validation passed: %s hourly observations.",
            len(df),
        )

    # -----------------------------------------------------------------
    # 6. Build canonical rich features
    # -----------------------------------------------------------------

    def build_production_features(
        self,
        raw_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the exact shared feature engineering used by training.
        """

        logger.info(
            "Running shared build_rich_features()..."
        )

        features = build_rich_features(
            raw_df
        )

        logger.info(
            "Rich feature frame shape: %s",
            features.shape,
        )

        if len(MODEL_FEATURES) != 100:
            raise FeaturePipelineError(
                "Canonical MODEL_FEATURES contract is not 100 features. "
                f"Found {len(MODEL_FEATURES)}."
            )

        # -------------------------------------------------------------
        # Remove warm-up rows.
        #
        # Historical lag/rolling features intentionally contain NaNs
        # at the beginning of the time series.
        # -------------------------------------------------------------

        complete_features = features.dropna(
            subset=list(MODEL_FEATURES)
        ).copy()

        if complete_features.empty:
            raise FeaturePipelineError(
                "No complete production feature row exists "
                "after historical warm-up."
            )

        # -------------------------------------------------------------
        # Select latest complete observation.
        # -------------------------------------------------------------

        latest = complete_features.iloc[
            [-1]
        ].copy()

        # -------------------------------------------------------------
        # Validate the exact canonical feature contract.
        # -------------------------------------------------------------

        validate_feature_frame(
            latest,
            require_complete=True,
        )

        logger.info(
            "Canonical feature validation passed."
        )

        logger.info(
            "Latest complete feature timestamp: %s",
            latest["timestamp"].iloc[0],
        )

        # -------------------------------------------------------------
        # Add production metadata.
        # -------------------------------------------------------------

        latest.insert(
            0,
            "city",
            self.city,
        )

        return latest

    # -----------------------------------------------------------------
    # 7. Prepare Hopsworks row
    # -----------------------------------------------------------------

    def prepare_feature_store_row(
        self,
        latest_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare the exact production row for Hopsworks.

        The Feature Store row contains:

            city
            timestamp
            100 MODEL_FEATURES

        target_aqi is intentionally not included in the serving
        feature vector because it is a source/history variable,
        not one of the model input columns.

        The historical AQI information required by the model is
        already represented through the canonical lag/rolling
        MODEL_FEATURES.
        """

        required_columns = [
            "city",
            "timestamp",
            *MODEL_FEATURES,
        ]

        missing = [
            column
            for column in required_columns
            if column not in latest_features.columns
        ]

        if missing:
            raise FeaturePipelineError(
                "Cannot prepare Hopsworks row. Missing columns: "
                + ", ".join(missing)
            )

        row = latest_features[
            required_columns
        ].copy()

        # -------------------------------------------------------------
        # Explicit timestamp representation
        # -------------------------------------------------------------

        row["timestamp"] = pd.to_datetime(
            row["timestamp"],
            utc=True,
        )

        # Hopsworks event time / primary key expects a stable
        # integer timestamp in the same convention used by the
        # existing project pipeline.
        row["timestamp"] = (
            row["timestamp"].astype("int64") // 10**6
        )

        # -------------------------------------------------------------
        # Explicit numeric dtypes
        # -------------------------------------------------------------

        temporal_features = {
            "hour",
            "day",
            "month",
            "day_of_week",
        }

        for column in MODEL_FEATURES:

            if column in temporal_features:

                row[column] = row[column].astype(
                    "int64"
                )

            else:

                row[column] = row[column].astype(
                    "float64"
                )

        logger.info(
            "Prepared Hopsworks row with %s model features.",
            len(MODEL_FEATURES),
        )

        return row

    # -----------------------------------------------------------------
    # 8. Connect to Hopsworks
    # -----------------------------------------------------------------

    def get_feature_store(self):
        """
        Authenticate with Hopsworks and return the Feature Store.
        """

        api_key = os.getenv(
            "HOPSWORKS_API_KEY"
        )

        if not api_key:
            raise FeaturePipelineError(
                "HOPSWORKS_API_KEY is not set."
            )

        logger.info(
            "Authenticating with Hopsworks..."
        )

        project = hopsworks.login(
            api_key_value=api_key,
            host=HOPSWORKS_HOST,
            cert_folder=tempfile.gettempdir(),
        )

        return project.get_feature_store()

    # -----------------------------------------------------------------
    # 9. Store latest production row
    # -----------------------------------------------------------------

    def save_to_feature_store(
        self,
        feature_row: pd.DataFrame,
    ) -> bool:
        """
        Insert the latest canonical production feature row
        into Hopsworks Feature Group v4.
        """

        if feature_row.empty:
            raise FeaturePipelineError(
                "Cannot insert an empty feature row."
            )

        fs = self.get_feature_store()

        logger.info(
            "Accessing Feature Group %s v%s...",
            FEATURE_GROUP_NAME,
            FEATURE_GROUP_VERSION,
        )

        feature_group = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=PRIMARY_KEY,
            event_time="timestamp",
            online_enabled=True,
            description=(
                "Canonical 100-feature production feature group "
                "for Karachi AQI prediction."
            ),
        )

        logger.info(
            "Inserting latest production feature row..."
        )

        feature_group.insert(
            feature_row,
            write_options={
                "wait_for_job": True,
            },
        )

        logger.info(
            "Successfully inserted production feature row "
            "into Hopsworks."
        )

        return True

    # -----------------------------------------------------------------
    # 10. Complete pipeline
    # -----------------------------------------------------------------

    def run(
        self,
        write_to_feature_store: bool = True,
    ) -> pd.DataFrame:
        """
        Execute the complete production feature pipeline.

        Returns:
            DataFrame containing the latest complete production row.
        """

        logger.info(
            "=================================================="
        )
        logger.info(
            "Starting Karachi AQI production feature pipeline"
        )
        logger.info(
            "=================================================="
        )

        # -------------------------------------------------------------
        # Fetch
        # -------------------------------------------------------------

        air_quality = (
            self.fetch_air_quality()
        )

        weather = (
            self.fetch_weather()
        )

        # -------------------------------------------------------------
        # Build raw canonical schema
        # -------------------------------------------------------------

        raw_df = self.build_raw_dataframe(
            air_quality,
            weather,
        )

        # -------------------------------------------------------------
        # Validate
        # -------------------------------------------------------------

        self.validate_raw_dataframe(
            raw_df
        )

        # -------------------------------------------------------------
        # Shared feature engineering
        # -------------------------------------------------------------

        rich_features = (
            self.build_production_features(
                raw_df
            )
        )

        # -------------------------------------------------------------
        # Prepare serving/Feature Store row
        # -------------------------------------------------------------

        feature_row = (
            self.prepare_feature_store_row(
                rich_features
            )
        )

        # -------------------------------------------------------------
        # Store
        # -------------------------------------------------------------

        if write_to_feature_store:

            self.save_to_feature_store(
                feature_row
            )

        logger.info(
            "Production feature pipeline completed successfully."
        )

        return feature_row


# ---------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------

if __name__ == "__main__":

    pipeline = AirQualityFeaturePipeline(
        city="karachi"
    )

    try:

        row = pipeline.run(
            write_to_feature_store=True
        )

        print("\n==============================================")
        print(" PRODUCTION FEATURE PIPELINE SUCCESS")
        print("==============================================")

        print(
            "\nCity:",
            row["city"].iloc[0],
        )

        print(
            "Timestamp:",
            row["timestamp"].iloc[0],
        )

        print(
            "Model feature count:",
            len(MODEL_FEATURES),
        )

        print(
            "Feature NaN count:",
            int(
                row[list(MODEL_FEATURES)]
                .isna()
                .sum()
                .sum()
            ),
        )

        print(
            "\nLatest production feature row:"
        )

        print(
            row[
                ["city", "timestamp"]
                + list(MODEL_FEATURES)
            ].T
        )

    except Exception as exc:

        logger.exception(
            "Production feature pipeline failed: %s",
            exc,
        )

        raise
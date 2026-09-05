
"""
Production feature pipeline for the Karachi AQI Predictor — v6.

Purpose
-------
Run the production feature update safely and repeatedly.

Production architecture:

    Hopsworks v6 historical computed rows
                    +
          Open-Meteo recent raw observations
                    ↓
          reconstruct raw context
                    ↓
          shared build_rich_features()
                    ↓
          generate missing production rows
                    ↓
             100 MODEL_FEATURES
                    ↓
             Hopsworks v6

Production guarantees
---------------------
1. v6 remains the canonical production Feature Group.
2. MODEL_FEATURES are never redefined here.
3. Historical raw context is reconstructed from v6.
4. Open-Meteo supplies recent raw observations.
5. The pipeline does not depend on a hardcoded calendar date/time.
6. Historical context is anchored to the latest timestamp in v6.
7. Missed hourly observations can be caught up.
8. Existing timestamps are never intentionally inserted again.
9. Duplicate timestamps are rejected.
10. The shared feature_engineering.py remains the single source
    of truth for feature calculations.
11. The pipeline is safe to execute repeatedly.
12. Production writes are enabled by default.
13. Dry-run mode is available explicitly.
14. A gap in the historical source data is never silently filled.
15. Only complete feature rows are allowed into v6.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

from src.feature_engineering import (
    MODEL_FEATURES,
    REQUIRED_RAW_COLUMNS,
    MAX_LOOKBACK_HOURS,
    build_rich_features,
    validate_feature_frame,
)


# =====================================================================
# ENVIRONMENT
# =====================================================================

load_dotenv()


# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =====================================================================
# KARACHI CONFIGURATION
# =====================================================================

CITY = "karachi"

KARACHI_LATITUDE = 24.8607
KARACHI_LONGITUDE = 67.0011


# =====================================================================
# OPEN-METEO CONFIGURATION
# =====================================================================

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

REQUEST_TIMEOUT = 60


# =====================================================================
# HOPSWORKS CONFIGURATION
# =====================================================================

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 6

PRIMARY_KEY = [
    "city",
    "timestamp",
]


# =====================================================================
# PRODUCTION CONTEXT
# =====================================================================

# The feature engineering contract requires up to 168 hours of
# historical context.
#
# Extra rows provide a safety margin.

CONTEXT_BUFFER_HOURS = 8

REQUIRED_CONTEXT_ROWS = (
    MAX_LOOKBACK_HOURS + CONTEXT_BUFFER_HOURS
)


# ---------------------------------------------------------------------
# API FETCH WINDOW
# ---------------------------------------------------------------------
#
# This is NOT a production schedule and NOT a hardcoded date/time.
#
# It only tells Open-Meteo how much recent data to request.
#
# The value is deliberately larger than the minimum feature
# lookback so normal missed hourly executions can be recovered.
#
# If a deployment is offline longer than this window, the pipeline
# fails safely rather than silently producing incomplete history.
#

API_RECENT_HOURS = (
    MAX_LOOKBACK_HOURS
    + CONTEXT_BUFFER_HOURS
    + 24
)


# =====================================================================
# RAW SOURCE COLUMNS
# =====================================================================

RAW_SOURCE_COLUMNS = (
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


# =====================================================================
# EXCEPTION
# =====================================================================


class FeaturePipelineError(RuntimeError):
    """Raised when production feature generation fails."""


# =====================================================================
# PIPELINE
# =====================================================================


class AirQualityFeaturePipeline:
    """
    Production hourly feature update pipeline.

    Hopsworks v6:
        Canonical historical computed feature store.

    Open-Meteo:
        Recent raw observations.

    feature_engineering.py:
        Single source of truth for feature calculations.

    Output:
        One or more newly generated production rows.
    """

    def __init__(
        self,
        city: str = CITY,
        latitude: float = KARACHI_LATITUDE,
        longitude: float = KARACHI_LONGITUDE,
    ) -> None:

        self.city = city
        self.latitude = latitude
        self.longitude = longitude

        logger.info(
            "Initialized production feature pipeline "
            "for city=%s",
            self.city,
        )


    # =================================================================
    # 1. HOPSWORKS CONNECTION
    # =================================================================

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
            "Connecting to Hopsworks..."
        )

        try:

            project = hopsworks.login(
                api_key_value=api_key,
                host=HOPSWORKS_HOST,
                cert_folder=tempfile.gettempdir(),
            )

        except Exception as exc:

            raise FeaturePipelineError(
                f"Hopsworks authentication failed: {exc}"
            ) from exc

        logger.info(
            "Hopsworks authentication successful."
        )

        return project.get_feature_store()


    # =================================================================
    # 2. GET FEATURE GROUP
    # =================================================================

    def get_feature_group(
        self,
        fs,
    ):
        """
        Retrieve the existing canonical v6 Feature Group.

        This pipeline never creates a new Feature Group.
        """

        logger.info(
            "Accessing Feature Group %s v%s...",
            FEATURE_GROUP_NAME,
            FEATURE_GROUP_VERSION,
        )

        try:

            feature_group = fs.get_feature_group(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
            )

        except Exception as exc:

            raise FeaturePipelineError(
                f"Unable to access Feature Group "
                f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}: "
                f"{exc}"
            ) from exc

        return feature_group


    # =================================================================
    # 3. NORMALIZE V6 TIMESTAMP
    # =================================================================

    @staticmethod
    def normalize_v6_timestamp(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert the existing v6 timestamp representation into
        timezone-aware UTC pandas timestamps.

        v6 is expected to store timestamp as Unix milliseconds.
        """

        dataframe = dataframe.copy()

        if "timestamp" not in dataframe.columns:

            raise FeaturePipelineError(
                "v6 data does not contain timestamp."
            )

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            unit="ms",
            utc=True,
            errors="coerce",
        )

        if dataframe["timestamp"].isna().any():

            raise FeaturePipelineError(
                "v6 contains invalid timestamps."
            )

        return dataframe


    # =================================================================
    # 4. DISCOVER LATEST V6 TIMESTAMP
    # =================================================================

    def discover_latest_v6_timestamp(
        self,
        feature_group,
    ) -> pd.Timestamp:
        """
        Discover the latest stored timestamp in v6.

        IMPORTANT:

        This method does NOT use a 96-hour wall-clock assumption.

        It first asks Hopsworks for the latest available data.

        If the Feature Store implementation returns a bounded recent
        window, the method verifies that the returned data actually
        contains a usable Karachi row.

        The production pipeline therefore never assumes that the
        latest v6 timestamp is equal to datetime.now().
        """

        logger.info(
            "Discovering latest stored v6 timestamp..."
        )

        # -------------------------------------------------------------
        # First attempt:
        # Read the Feature Group without imposing a production
        # calendar/date assumption.
        #
        # Hopsworks Feature Groups support read() for retrieving the
        # stored data. This gives us the canonical latest timestamp.
        # -------------------------------------------------------------

        try:

            data = feature_group.read(
                dataframe_type="pandas",
                read_options={"use_arrow_flight": False}
            )

        except Exception as exc:

            raise FeaturePipelineError(
                "Failed to discover latest timestamp from v6: "
                f"{exc}"
            ) from exc

        if data is None or data.empty:

            raise FeaturePipelineError(
                "v6 Feature Group contains no rows."
            )

        data = self.normalize_v6_timestamp(
            data
        )

        if "city" not in data.columns:

            raise FeaturePipelineError(
                "v6 data does not contain city."
            )

        data = data[
            data["city"].astype(str).str.lower()
            == self.city.lower()
        ].copy()

        if data.empty:

            raise FeaturePipelineError(
                f"No v6 rows found for city={self.city}."
            )

        latest_timestamp = (
            data["timestamp"].max()
        )

        logger.info(
            "Latest stored v6 timestamp: %s",
            latest_timestamp,
        )

        return latest_timestamp


    # =================================================================
    # 5. READ RECENT V6 CONTEXT
    # =================================================================

    def read_recent_v6_context(
        self,
        feature_group,
        latest_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Read enough v6 rows ending at the latest stored timestamp.

        The window is anchored to v6 itself rather than datetime.now().
        """

        context_start = (
            latest_timestamp
            - timedelta(
                hours=REQUIRED_CONTEXT_ROWS
            )
        )

        context_end = (
            latest_timestamp
            + timedelta(hours=1)
        )

        logger.info(
            "Reading v6 historical context..."
        )

        logger.info(
            "    start = %s",
            context_start.isoformat(),
        )

        logger.info(
            "    end   = %s",
            context_end.isoformat(),
        )

        try:

            history = feature_group.read(
                start_time=context_start,
                end_time=context_end,
                dataframe_type="pandas",
                read_options={"use_arrow_flight": False}
            )

        except Exception as exc:

            raise FeaturePipelineError(
                "Failed to read v6 historical context: "
                f"{exc}"
            ) from exc

        if history is None or history.empty:

            raise FeaturePipelineError(
                "v6 returned no historical context."
            )

        logger.info(
            "Historical rows retrieved from v6: %s",
            len(history),
        )

        return history


    # =================================================================
    # 6. VALIDATE V6 CONTEXT
    # =================================================================

    def validate_historical_context(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate v6 context and reconstruct the raw source columns
        required by build_rich_features().
        """

        required_columns = [
            "city",
            "timestamp",
            "target_aqi",
            *MODEL_FEATURES,
        ]

        missing = [
            column
            for column in required_columns
            if column not in history.columns
        ]

        if missing:

            raise FeaturePipelineError(
                "Hopsworks v6 is missing required columns: "
                + ", ".join(missing)
            )

        history = self.normalize_v6_timestamp(
            history
        )

        history = history[
            history["city"].astype(str).str.lower()
            == self.city.lower()
        ].copy()

        if history.empty:

            raise FeaturePipelineError(
                f"No v6 rows found for city={self.city}."
            )

        history = (
            history
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # -------------------------------------------------------------
        # Duplicate timestamps are never silently resolved.
        # -------------------------------------------------------------

        duplicates = history[
            history["timestamp"].duplicated(
                keep=False
            )
        ]

        if not duplicates.empty:

            duplicate_values = (
                duplicates["timestamp"]
                .astype(str)
                .unique()
                .tolist()
            )

            raise FeaturePipelineError(
                "Duplicate timestamps detected in v6 context: "
                + ", ".join(duplicate_values)
            )

        # -------------------------------------------------------------
        # Reconstruct canonical raw history.
        # -------------------------------------------------------------

        raw_history = history[
            [
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

        # -------------------------------------------------------------
        # Numeric normalization.
        # -------------------------------------------------------------

        for column in REQUIRED_RAW_COLUMNS:

            if column == "timestamp":
                continue

            raw_history[column] = pd.to_numeric(
                raw_history[column],
                errors="coerce",
            ).astype("float64")

        # -------------------------------------------------------------
        # Missing values.
        # -------------------------------------------------------------

        missing_counts = (
            raw_history[
                list(REQUIRED_RAW_COLUMNS)
            ]
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

            raise FeaturePipelineError(
                "v6 source data contains missing values: "
                + details
            )

        # -------------------------------------------------------------
        # Hourly continuity.
        # -------------------------------------------------------------

        intervals = (
            raw_history["timestamp"]
            .diff()
            .dropna()
        )

        invalid_intervals = intervals[
            intervals
            != pd.Timedelta(hours=1)
        ]

        if not invalid_intervals.empty:

            raise FeaturePipelineError(
                "v6 historical context is not continuous hourly data."
            )

        # -------------------------------------------------------------
        # Minimum lookback.
        # -------------------------------------------------------------

        if len(raw_history) < MAX_LOOKBACK_HOURS:

            raise FeaturePipelineError(
                "Insufficient historical context in v6. "
                f"Need at least {MAX_LOOKBACK_HOURS} rows, "
                f"received {len(raw_history)}."
            )

        logger.info(
            "Historical v6 context validation PASSED."
        )

        logger.info(
            "Context rows: %s",
            len(raw_history),
        )

        logger.info(
            "Context range: %s → %s",
            raw_history["timestamp"].iloc[0],
            raw_history["timestamp"].iloc[-1],
        )

        return raw_history


    # =================================================================
    # 7. FETCH RECENT AIR QUALITY
    # =================================================================

    def fetch_recent_air_quality(
        self,
    ) -> pd.DataFrame:
        """
        Fetch recent hourly air-quality observations from Open-Meteo.

        The window is relative to the current execution time.
        No calendar date is hardcoded.
        """

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "past_hours": API_RECENT_HOURS,
            "forecast_hours": 1,
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
            "Fetching recent hourly air-quality data..."
        )

        try:

            response = requests.get(
                AIR_QUALITY_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            payload = response.json()

        except requests.RequestException as exc:

            raise FeaturePipelineError(
                f"Air-quality API request failed: {exc}"
            ) from exc

        hourly = payload.get("hourly")

        if not hourly:

            raise FeaturePipelineError(
                "Air-quality API returned no hourly data."
            )

        required = [
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
            for column in required
            if column not in hourly
        ]

        if missing:

            raise FeaturePipelineError(
                "Air-quality API is missing fields: "
                + ", ".join(missing)
            )

        df = pd.DataFrame(hourly)

        df["timestamp"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="coerce",
        )

        if df["timestamp"].isna().any():

            raise FeaturePipelineError(
                "Air-quality API returned invalid timestamps."
            )

        df = df.rename(
            columns={
                "pm2_5": "pm25",
                "us_aqi": "target_aqi",
            }
        )

        df = df[
            [
                "timestamp",
                "pm25",
                "pm10",
                "ozone",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "carbon_monoxide",
                "target_aqi",
            ]
        ].copy()

        return df


    # =================================================================
    # 8. FETCH RECENT WEATHER
    # =================================================================

    def fetch_recent_weather(
        self,
    ) -> pd.DataFrame:
        """
        Fetch recent hourly weather observations from Open-Meteo.
        """

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "past_hours": API_RECENT_HOURS,
            "forecast_hours": 1,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
            ],
        }

        logger.info(
            "Fetching recent hourly weather data..."
        )

        try:

            response = requests.get(
                WEATHER_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            payload = response.json()

        except requests.RequestException as exc:

            raise FeaturePipelineError(
                f"Weather API request failed: {exc}"
            ) from exc

        hourly = payload.get("hourly")

        if not hourly:

            raise FeaturePipelineError(
                "Weather API returned no hourly data."
            )

        required = [
            "time",
            "temperature_2m",
            "relative_humidity_2m",
        ]

        missing = [
            column
            for column in required
            if column not in hourly
        ]

        if missing:

            raise FeaturePipelineError(
                "Weather API is missing fields: "
                + ", ".join(missing)
            )

        df = pd.DataFrame(hourly)

        df["timestamp"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="coerce",
        )

        if df["timestamp"].isna().any():

            raise FeaturePipelineError(
                "Weather API returned invalid timestamps."
            )

        df = df.rename(
            columns={
                "temperature_2m": "temperature",
                "relative_humidity_2m": "humidity",
            }
        )

        df = df[
            [
                "timestamp",
                "temperature",
                "humidity",
            ]
        ].copy()

        return df


    # =================================================================
    # 9. BUILD RECENT RAW DATASET
    # =================================================================

    def build_recent_raw_dataset(
        self,
        air_quality: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Open-Meteo air-quality and weather observations.
        """

        merged = air_quality.merge(
            weather,
            on="timestamp",
            how="inner",
        )

        if merged.empty:

            raise FeaturePipelineError(
                "Air-quality and weather APIs have no "
                "overlapping timestamps."
            )

        merged = (
            merged
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        # -------------------------------------------------------------
        # Only completed/current timestamps are usable.
        # -------------------------------------------------------------

        now = pd.Timestamp(
            datetime.now(timezone.utc)
        )

        merged = merged[
            merged["timestamp"] <= now
        ].copy()

        if merged.empty:

            raise FeaturePipelineError(
                "No usable completed hourly observation "
                "is available from Open-Meteo."
            )

        # -------------------------------------------------------------
        # Numeric normalization.
        # -------------------------------------------------------------

        for column in REQUIRED_RAW_COLUMNS:

            if column == "timestamp":
                continue

            merged[column] = pd.to_numeric(
                merged[column],
                errors="coerce",
            ).astype("float64")

        # -------------------------------------------------------------
        # Missing values.
        # -------------------------------------------------------------

        missing = (
            merged[
                list(REQUIRED_RAW_COLUMNS)
            ]
            .isna()
            .sum()
        )

        missing = missing[
            missing > 0
        ]

        if not missing.empty:

            details = ", ".join(
                f"{column}={int(count)}"
                for column, count in missing.items()
            )

            raise FeaturePipelineError(
                "Open-Meteo recent data contains missing values: "
                + details
            )

        return merged[
            list(REQUIRED_RAW_COLUMNS)
        ].copy()


    # =================================================================
    # 10. FIND MISSING PRODUCTION HOURS
    # =================================================================

    def find_missing_hours(
        self,
        raw_history: pd.DataFrame,
        recent_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Identify observations after the latest timestamp stored in v6.

        Existing timestamps are excluded.

        Missed hourly observations are therefore processed in order.
        """

        latest_v6 = (
            raw_history["timestamp"].max()
        )

        candidates = recent_raw[
            recent_raw["timestamp"]
            > latest_v6
        ].copy()

        if candidates.empty:

            logger.info(
                "No new hourly observations are available."
            )

            return candidates

        candidates = (
            candidates
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        # -------------------------------------------------------------
        # The first new timestamp must immediately follow V6.
        #
        # This prevents silently skipping an unavailable hour.
        # -------------------------------------------------------------

        expected_first_timestamp = (
            latest_v6
            + pd.Timedelta(hours=1)
        )

        actual_first_timestamp = (
            candidates["timestamp"].iloc[0]
        )

        if actual_first_timestamp != expected_first_timestamp:

            raise FeaturePipelineError(
                "There is a gap between v6 and Open-Meteo data.\n"
                f"Latest v6 timestamp: "
                f"{latest_v6}\n"
                f"First available API timestamp: "
                f"{actual_first_timestamp}\n"
                f"Expected first new timestamp: "
                f"{expected_first_timestamp}\n\n"
                "The pipeline stopped instead of silently "
                "creating incomplete history."
            )

        # -------------------------------------------------------------
        # New observations themselves must be continuous.
        # -------------------------------------------------------------

        intervals = (
            candidates["timestamp"]
            .diff()
            .dropna()
        )

        if not intervals.empty:

            invalid_intervals = intervals[
                intervals
                != pd.Timedelta(hours=1)
            ]

            if not invalid_intervals.empty:

                raise FeaturePipelineError(
                    "Open-Meteo recent observations contain "
                    "a missing hourly timestamp."
                )

        logger.info(
            "New hourly observations detected: %s",
            len(candidates),
        )

        logger.info(
            "New range: %s → %s",
            candidates["timestamp"].iloc[0],
            candidates["timestamp"].iloc[-1],
        )

        return candidates


    # =================================================================
    # 11. BUILD NEW FEATURES
    # =================================================================

    def build_new_feature_rows(
        self,
        raw_history: pd.DataFrame,
        new_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the exact shared feature engineering on:

            historical v6 raw context
                        +
                 new Open-Meteo rows

        Only newly generated timestamps are returned.
        """

        if new_raw.empty:

            return pd.DataFrame()

        combined = pd.concat(
            [
                raw_history,
                new_raw,
            ],
            ignore_index=True,
        )

        # -------------------------------------------------------------
        # Duplicate timestamps must never silently overwrite source
        # history.
        # -------------------------------------------------------------

        duplicate_mask = combined[
            "timestamp"
        ].duplicated(
            keep=False
        )

        if duplicate_mask.any():

            duplicates = (
                combined.loc[
                    duplicate_mask,
                    "timestamp",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise FeaturePipelineError(
                "Duplicate timestamps detected while combining "
                "v6 history and Open-Meteo data: "
                + ", ".join(duplicates)
            )

        combined = (
            combined
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # -------------------------------------------------------------
        # Final continuity check.
        # -------------------------------------------------------------

        intervals = (
            combined["timestamp"]
            .diff()
            .dropna()
        )

        invalid_intervals = intervals[
            intervals
            != pd.Timedelta(hours=1)
        ]

        if not invalid_intervals.empty:

            raise FeaturePipelineError(
                "Combined historical + new raw data is "
                "not continuous hourly data."
            )

        logger.info(
            "Running shared build_rich_features()..."
        )

        features = build_rich_features(
            combined
        )

        logger.info(
            "Rich feature frame generated: "
            "%s rows × %s columns",
            features.shape[0],
            features.shape[1],
        )

        # -------------------------------------------------------------
        # Canonical contract.
        # -------------------------------------------------------------

        if len(MODEL_FEATURES) != 100:

            raise FeaturePipelineError(
                "MODEL_FEATURES contract changed. "
                f"Expected 100, found {len(MODEL_FEATURES)}."
            )

        # -------------------------------------------------------------
        # Select only new timestamps.
        # -------------------------------------------------------------

        new_timestamps = set(
            new_raw["timestamp"]
        )

        latest_features = features[
            features["timestamp"].isin(
                new_timestamps
            )
        ].copy()

        if latest_features.empty:

            raise FeaturePipelineError(
                "Feature engineering produced no rows "
                "for the new production observations."
            )

        # -------------------------------------------------------------
        # Validate complete feature rows.
        # -------------------------------------------------------------

        validate_feature_frame(
            latest_features,
            require_complete=True,
        )

        logger.info(
            "Production 100-feature validation PASSED "
            "for %s new row(s).",
            len(latest_features),
        )

        latest_features.insert(
            0,
            "city",
            self.city,
        )

        return latest_features


    # =================================================================
    # 12. PREPARE V6 ROWS
    # =================================================================

    def prepare_feature_store_rows(
        self,
        latest_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare rows according to the existing v6 schema:

            city
            timestamp
            target_aqi
            100 MODEL_FEATURES
        """

        if latest_features.empty:

            return latest_features

        output_columns = [
            "city",
            "timestamp",
            "target_aqi",
            *MODEL_FEATURES,
        ]

        missing = [
            column
            for column in output_columns
            if column not in latest_features.columns
        ]

        if missing:

            raise FeaturePipelineError(
                "Cannot prepare v6 rows. Missing columns: "
                + ", ".join(missing)
            )

        rows = latest_features[
            output_columns
        ].copy()

        # -------------------------------------------------------------
        # Timestamp.
        #
        # Keep timezone-aware datetime here. Hopsworks receives the
        # same logical timestamp type used by the existing Feature
        # Group schema.
        # -------------------------------------------------------------

        rows["timestamp"] = pd.to_datetime(
            rows["timestamp"],
            utc=True,
        )

        # -------------------------------------------------------------
        # target_aqi.
        # -------------------------------------------------------------

        rows["target_aqi"] = (
            pd.to_numeric(
                rows["target_aqi"],
                errors="coerce",
            )
            .astype("float64")
        )

        # -------------------------------------------------------------
        # MODEL_FEATURES dtypes.
        # -------------------------------------------------------------

        temporal_features = {
            "hour",
            "day",
            "month",
            "day_of_week",
        }

        for column in MODEL_FEATURES:

            if column in temporal_features:

                rows[column] = (
                    pd.to_numeric(
                        rows[column],
                        errors="coerce",
                    )
                    .astype("int64")
                )

            else:

                rows[column] = (
                    pd.to_numeric(
                        rows[column],
                        errors="coerce",
                    )
                    .astype("float64")
                )

        # -------------------------------------------------------------
        # Final NaN protection.
        # -------------------------------------------------------------

        if rows[
            list(MODEL_FEATURES)
        ].isna().any().any():

            raise FeaturePipelineError(
                "Final production rows contain NaN "
                "inside MODEL_FEATURES."
            )

        if rows[
            ["city", "timestamp", "target_aqi"]
        ].isna().any().any():

            raise FeaturePipelineError(
                "Final production rows contain NaN "
                "in required metadata/source columns."
            )

        # -------------------------------------------------------------
        # Primary-key duplicate protection.
        # -------------------------------------------------------------

        if rows.duplicated(
            subset=PRIMARY_KEY
        ).any():

            raise FeaturePipelineError(
                "Duplicate city/timestamp primary keys "
                "detected in production batch."
            )

        logger.info(
            "Prepared %s production row(s) "
            "with %s MODEL_FEATURES.",
            len(rows),
            len(MODEL_FEATURES),
        )

        return rows


    # =================================================================
    # 13. FINAL DUPLICATE CHECK
    # =================================================================

    def verify_rows_are_new(
        self,
        feature_group,
        rows: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform one final duplicate check immediately before insert.

        This protects against another production process inserting
        the same timestamp between the initial read and this write.
        """

        if rows.empty:

            return rows

        earliest = (
            rows["timestamp"].min()
        )

        latest = (
            rows["timestamp"].max()
        )

        logger.info(
            "Performing final v6 duplicate check..."
        )

        try:

            existing = feature_group.read(
                start_time=earliest,
                end_time=latest + timedelta(hours=1),
                dataframe_type="pandas",
            )

        except Exception as exc:

            raise FeaturePipelineError(
                "Failed final duplicate check against v6: "
                f"{exc}"
            ) from exc

        if existing is None or existing.empty:

            return rows

        existing = self.normalize_v6_timestamp(
            existing
        )

        if "city" not in existing.columns:

            raise FeaturePipelineError(
                "Final duplicate check returned data "
                "without city column."
            )

        existing = existing[
            existing["city"].astype(str).str.lower()
            == self.city.lower()
        ]

        existing_keys = set(
            zip(
                existing["city"].astype(str),
                existing["timestamp"],
            )
        )

        keep_mask = []

        for _, row in rows.iterrows():

            key = (
                str(row["city"]),
                pd.Timestamp(row["timestamp"]),
            )

            if key in existing_keys:

                logger.warning(
                    "Skipping timestamp already present "
                    "in v6: %s",
                    row["timestamp"],
                )

                keep_mask.append(False)

            else:

                keep_mask.append(True)

        rows = rows.loc[
            keep_mask
        ].copy()

        if rows.empty:

            logger.info(
                "All generated rows already exist in v6."
            )

        return rows


    # =================================================================
    # 14. SAVE NEW ROWS
    # =================================================================

    def save_to_feature_store(
        self,
        feature_group,
        feature_rows: pd.DataFrame,
    ) -> int:
        """
        Insert only genuinely new production rows.
        """

        if feature_rows.empty:

            logger.info(
                "Nothing to insert."
            )

            return 0

        logger.info(
            "Writing %s new production row(s) to v6...",
            len(feature_rows),
        )

        try:

            feature_group.insert(
                feature_rows,
                write_options={
                    "wait_for_job": True,
                },
            )

        except Exception as exc:

            raise FeaturePipelineError(
                "Failed to insert production rows into v6: "
                f"{exc}"
            ) from exc

        logger.info(
            "Successfully inserted %s production row(s).",
            len(feature_rows),
        )

        return len(feature_rows)


    # =================================================================
    # 15. COMPLETE RUN
    # =================================================================

    def run(
        self,
        write_to_feature_store: bool = True,
    ) -> pd.DataFrame:
        """
        Execute one production feature update.

        Normal hourly execution:
            approximately one new row.

        Missed execution:
            multiple rows may be caught up if Open-Meteo still
            provides the complete missing hourly observations.

        Already up to date:
            returns an empty DataFrame.

        Dry run:
            write_to_feature_store=False
        """

        logger.info(
            "=================================================="
        )

        logger.info(
            "KARACHI AQI PRODUCTION FEATURE PIPELINE — v6"
        )

        logger.info(
            "=================================================="
        )

        # -------------------------------------------------------------
        # Connect.
        # -------------------------------------------------------------

        fs = self.get_feature_store()

        feature_group = self.get_feature_group(
            fs
        )

        # -------------------------------------------------------------
        # Discover the actual latest V6 timestamp.
        # -------------------------------------------------------------

        latest_v6_timestamp = (
            self.discover_latest_v6_timestamp(
                feature_group
            )
        )

        # -------------------------------------------------------------
        # Read historical context anchored to V6.
        # -------------------------------------------------------------

        history = (
            self.read_recent_v6_context(
                feature_group,
                latest_v6_timestamp,
            )
        )

        raw_history = (
            self.validate_historical_context(
                history
            )
        )

        # -------------------------------------------------------------
        # Fetch recent Open-Meteo data.
        # -------------------------------------------------------------

        air_quality = (
            self.fetch_recent_air_quality()
        )

        weather = (
            self.fetch_recent_weather()
        )

        recent_raw = (
            self.build_recent_raw_dataset(
                air_quality,
                weather,
            )
        )

        # -------------------------------------------------------------
        # Determine genuinely missing production hours.
        # -------------------------------------------------------------

        new_raw = self.find_missing_hours(
            raw_history,
            recent_raw,
        )

        if new_raw.empty:

            logger.info(
                "=================================================="
            )

            logger.info(
                "PRODUCTION PIPELINE ALREADY UP TO DATE"
            )

            logger.info(
                "=================================================="
            )

            return pd.DataFrame()

        # -------------------------------------------------------------
        # Generate features using shared engineering.
        # -------------------------------------------------------------

        latest_features = (
            self.build_new_feature_rows(
                raw_history,
                new_raw,
            )
        )

        # -------------------------------------------------------------
        # Prepare exact v6 schema.
        # -------------------------------------------------------------

        feature_rows = (
            self.prepare_feature_store_rows(
                latest_features
            )
        )

        # -------------------------------------------------------------
        # Final duplicate protection.
        # -------------------------------------------------------------

        feature_rows = (
            self.verify_rows_are_new(
                feature_group,
                feature_rows,
            )
        )

        if feature_rows.empty:

            logger.info(
                "No genuinely new rows remain after "
                "final duplicate protection."
            )

            return feature_rows

        # -------------------------------------------------------------
        # Write or dry-run.
        # -------------------------------------------------------------

        if write_to_feature_store:

            inserted = (
                self.save_to_feature_store(
                    feature_group,
                    feature_rows,
                )
            )

            logger.info(
                "Inserted rows: %s",
                inserted,
            )

        else:

            logger.info(
                "DRY RUN: no rows were inserted into Hopsworks."
            )

        logger.info(
            "=================================================="
        )

        logger.info(
            "PRODUCTION FEATURE PIPELINE COMPLETED"
        )

        logger.info(
            "=================================================="
        )

        return feature_rows


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================

if __name__ == "__main__":

    pipeline = AirQualityFeaturePipeline(
        city="karachi"
    )

    try:

        # -------------------------------------------------------------
        # REAL PRODUCTION MODE
        #
        # Writes are enabled.
        # -------------------------------------------------------------

        rows = pipeline.run(
            write_to_feature_store=True
        )

        if rows.empty:

            print(
                "\n=============================================="
            )

            print(
                " V6 ALREADY UP TO DATE"
            )

            print(
                "=============================================="
            )

            print(
                "\nNo new production rows required."
            )

        else:

            print(
                "\n=============================================="
            )

            print(
                " PRODUCTION FEATURE PIPELINE SUCCESS"
            )

            print(
                "=============================================="
            )

            print(
                "\nCity:",
                rows["city"].iloc[0],
            )

            print(
                "Rows processed:",
                len(rows),
            )

            print(
                "First timestamp:",
                rows["timestamp"].iloc[0],
            )

            print(
                "Last timestamp:",
                rows["timestamp"].iloc[-1],
            )

            print(
                "MODEL_FEATURES:",
                len(MODEL_FEATURES),
            )

            print(
                "NaN count:",
                int(
                    rows[
                        list(MODEL_FEATURES)
                    ]
                    .isna()
                    .sum()
                    .sum()
                ),
            )

            print(
                "\nProduction rows:"
            )

            print(
                rows[
                    [
                        "city",
                        "timestamp",
                        "target_aqi",
                    ]
                ].to_string(
                    index=False
                )
            )

    except Exception as exc:

        logger.exception(
            "Production feature pipeline failed: %s",
            exc,
        )

        raise


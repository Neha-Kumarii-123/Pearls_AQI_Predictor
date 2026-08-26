
"""
Gap-fill feature pipeline for the Karachi AQI Predictor — v6.

Purpose
-------
Repair the missing time period between the last timestamp currently
stored in Hopsworks Feature Group v6 and the latest available
historical data from Open-Meteo.

This pipeline is intentionally separate from:

    backfill_pipeline.py
        Initial/full historical v6 construction.

    feature_pipeline.py
        Normal hourly production update.

Architecture
------------

    Existing V6 latest timestamp
                +
       168-hour historical context
                +
        Missing raw period
                ↓
          Open-Meteo APIs
                ↓
       continuous raw dataframe
                ↓
       shared build_rich_features()
                ↓
       canonical 100 MODEL_FEATURES
                ↓
     keep ONLY timestamps missing
                ↓
          Hopsworks v6

Important
---------
1. v5 remains untouched.
2. Existing v6 rows are never deleted.
3. Existing v6 rows are never overwritten by this script.
4. Only missing timestamps are inserted.
5. The shared feature_engineering.py is the single source of truth.
6. No feature definitions are duplicated here.
7. The pipeline is idempotent:
       running it again should not create duplicate rows.
8. The gap-fill process requires continuous hourly raw data.
9. At least MAX_LOOKBACK_HOURS of historical context is included
   before the first missing timestamp.
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

from feature_engineering import (
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
    "https://archive-api.open-meteo.com/v1/archive"
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
# GAP-FILL CONFIGURATION
# =====================================================================

# We need the complete historical context required by the shared
# feature engineering before the first missing timestamp.
#
# MAX_LOOKBACK_HOURS is currently 168.
#
# We intentionally add a small safety margin.
CONTEXT_EXTRA_HOURS = 8

CONTEXT_HOURS = (
    MAX_LOOKBACK_HOURS
    + CONTEXT_EXTRA_HOURS
)


# Open-Meteo archive uses date-based requests.
#
# The API is queried through dates derived from the actual V6
# timestamps rather than hard-coded historical dates.
# This makes the script reusable for future gap repairs.


# =====================================================================
# RAW SOURCE SCHEMA
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


class GapFillPipelineError(RuntimeError):
    """Raised when the v6 gap-fill pipeline fails."""


# =====================================================================
# PIPELINE
# =====================================================================


class AQIGapFillPipeline:
    """
    Repair missing hourly computed feature rows in Hopsworks v6.

    Existing v6 data is used as historical context.

    Open-Meteo supplies the missing raw observations.

    The shared feature_engineering.py creates the canonical
    computed feature frame.

    Only genuinely missing timestamps are inserted.
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
            "Initialized v6 gap-fill pipeline "
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
            raise GapFillPipelineError(
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

            raise GapFillPipelineError(
                f"Unable to authenticate with Hopsworks: {exc}"
            ) from exc

        logger.info(
            "Hopsworks authentication successful."
        )

        return project.get_feature_store()


    # =================================================================
    # 2. GET EXISTING V6
    # =================================================================

    def get_feature_group(
        self,
        fs,
    ):
        """
        Retrieve the existing v6 Feature Group.

        We intentionally do NOT create a Feature Group here.

        v6 must already exist.
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

            raise GapFillPipelineError(
                f"Unable to access "
                f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}: "
                f"{exc}"
            ) from exc

        return feature_group


    # =================================================================
    # 3. READ EXISTING V6
    # =================================================================

    def read_existing_v6(
        self,
        feature_group,
    ) -> pd.DataFrame:
        """
        Read the existing v6 dataset.

        This is used to determine:

            - the current V6 coverage
            - the latest stored timestamp
            - which timestamps already exist

        We read the existing dataset here because the gap is expected
        to be relatively small and this provides the safest duplicate
        protection.
        """

        logger.info(
            "Reading existing v6 dataset..."
        )

        try:

            existing = feature_group.read(
                dataframe_type="pandas",
            )

        except Exception as exc:

            raise GapFillPipelineError(
                f"Failed to read existing v6 dataset: {exc}"
            ) from exc

        if existing is None or existing.empty:

            raise GapFillPipelineError(
                "Feature Group v6 is empty. "
                "Use backfill_pipeline.py for initial population."
            )

        required_columns = [
            "city",
            "timestamp",
            "target_aqi",
            *MODEL_FEATURES,
        ]

        missing = [
            column
            for column in required_columns
            if column not in existing.columns
        ]

        if missing:

            raise GapFillPipelineError(
                "Existing v6 dataset is missing required columns: "
                + ", ".join(missing)
            )

        # -------------------------------------------------------------
        # Normalize timestamp.
        #
        # Hopsworks may return timestamp as datetime or integer
        # milliseconds depending on the reader/backend.
        # -------------------------------------------------------------

        timestamp_series = existing["timestamp"]

        if pd.api.types.is_numeric_dtype(
            timestamp_series
        ):

            existing["timestamp"] = pd.to_datetime(
                timestamp_series,
                unit="ms",
                utc=True,
                errors="coerce",
            )

        else:

            existing["timestamp"] = pd.to_datetime(
                timestamp_series,
                utc=True,
                errors="coerce",
            )

        if existing["timestamp"].isna().any():

            raise GapFillPipelineError(
                "Existing v6 dataset contains invalid timestamps."
            )

        existing["city"] = (
            existing["city"]
            .astype(str)
        )

        existing = existing[
            existing["city"].str.lower()
            == self.city.lower()
        ].copy()

        if existing.empty:

            raise GapFillPipelineError(
                f"No v6 rows found for city={self.city}."
            )

        existing = (
            existing
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if existing["timestamp"].duplicated().any():

            duplicates = (
                existing["timestamp"]
                .duplicated()
                .sum()
            )

            raise GapFillPipelineError(
                "Existing v6 dataset contains "
                f"{duplicates} duplicate timestamps."
            )

        logger.info(
            "Existing v6 rows: %s",
            len(existing),
        )

        logger.info(
            "Existing v6 range: %s → %s",
            existing["timestamp"].iloc[0],
            existing["timestamp"].iloc[-1],
        )

        return existing


    # =================================================================
    # 4. DETERMINE GAP
    # =================================================================

    def determine_gap(
        self,
        existing_v6: pd.DataFrame,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Determine the missing period after the latest V6 timestamp.

        The end of the gap is the latest completed historical hour
        available from Open-Meteo archive.

        Returns:

            gap_start
            gap_end
        """

        latest_v6_timestamp = (
            existing_v6["timestamp"].iloc[-1]
        )

        gap_start = (
            latest_v6_timestamp
            + pd.Timedelta(hours=1)
        )

        # -------------------------------------------------------------
        # Open-Meteo archive data is date-based.
        #
        # Use yesterday as the safest completed historical boundary.
        #
        # This avoids mixing the gap-fill process with the normal
        # production/live update process.
        # -------------------------------------------------------------

        now_utc = datetime.now(
            timezone.utc
        )

        latest_completed_date = (
            pd.Timestamp(
                now_utc.date(),
                tz="UTC",
            )
            - pd.Timedelta(days=1)
        )

        gap_end = (
            latest_completed_date
            + pd.Timedelta(hours=23)
        )

        logger.info(
            "Latest V6 timestamp: %s",
            latest_v6_timestamp,
        )

        logger.info(
            "Calculated gap start: %s",
            gap_start,
        )

        logger.info(
            "Latest safe archive endpoint: %s",
            gap_end,
        )

        if gap_start > gap_end:

            logger.info(
                "No historical gap requires filling."
            )

            return gap_start, gap_end

        gap_hours = int(
            (
                gap_end - gap_start
            ).total_seconds()
            / 3600
        ) + 1

        logger.info(
            "Gap size: %s hourly rows",
            gap_hours,
        )

        return gap_start, gap_end


    # =================================================================
    # 5. DETERMINE API FETCH WINDOW
    # =================================================================

    def determine_fetch_window(
        self,
        gap_start: pd.Timestamp,
        gap_end: pd.Timestamp,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Include sufficient historical context before the gap.

        Example:

            gap starts:
                2026-08-02 00:00

            context:
                previous 176 hours

            API fetch:
                context + missing gap

        This ensures lag_168 and rolling_168 features have the
        required historical information.
        """

        context_start = (
            gap_start
            - pd.Timedelta(
                hours=CONTEXT_HOURS
            )
        )

        fetch_start = context_start

        fetch_end = gap_end

        logger.info(
            "Open-Meteo fetch window:"
        )

        logger.info(
            "    start = %s",
            fetch_start,
        )

        logger.info(
            "    end   = %s",
            fetch_end,
        )

        return fetch_start, fetch_end


    # =================================================================
    # 6. FETCH AIR QUALITY
    # =================================================================

    def fetch_air_quality(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch historical hourly air-quality data from Open-Meteo.
        """

        logger.info(
            "Fetching air-quality data: %s → %s",
            start_date,
            end_date,
        )

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

        try:

            response = requests.get(
                AIR_QUALITY_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            payload = response.json()

        except requests.RequestException as exc:

            raise GapFillPipelineError(
                f"Air-quality API request failed: {exc}"
            ) from exc

        hourly = payload.get("hourly")

        if not hourly:

            raise GapFillPipelineError(
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

            raise GapFillPipelineError(
                "Air-quality API response is missing: "
                + ", ".join(missing)
            )

        df = pd.DataFrame(hourly)

        df["timestamp"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="coerce",
        )

        if df["timestamp"].isna().any():

            raise GapFillPipelineError(
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

        logger.info(
            "Air-quality rows received: %s",
            len(df),
        )

        return df


    # =================================================================
    # 7. FETCH WEATHER
    # =================================================================

    def fetch_weather(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch historical hourly weather data from Open-Meteo.
        """

        logger.info(
            "Fetching weather data: %s → %s",
            start_date,
            end_date,
        )

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

        try:

            response = requests.get(
                WEATHER_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            payload = response.json()

        except requests.RequestException as exc:

            raise GapFillPipelineError(
                f"Weather API request failed: {exc}"
            ) from exc

        hourly = payload.get("hourly")

        if not hourly:

            raise GapFillPipelineError(
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

            raise GapFillPipelineError(
                "Weather API response is missing: "
                + ", ".join(missing)
            )

        df = pd.DataFrame(hourly)

        df["timestamp"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="coerce",
        )

        if df["timestamp"].isna().any():

            raise GapFillPipelineError(
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

        logger.info(
            "Weather rows received: %s",
            len(df),
        )

        return df


    # =================================================================
    # 8. BUILD RAW DATASET
    # =================================================================

    def build_raw_dataframe(
        self,
        air_quality: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Open-Meteo air-quality and weather observations into
        the project's canonical raw schema.
        """

        logger.info(
            "Merging air-quality and weather data..."
        )

        df = air_quality.merge(
            weather,
            on="timestamp",
            how="inner",
        )

        if df.empty:

            raise GapFillPipelineError(
                "No overlapping timestamps between "
                "air-quality and weather data."
            )

        df["city"] = self.city

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
            errors="coerce",
        )

        if df["timestamp"].isna().any():

            raise GapFillPipelineError(
                "Raw dataframe contains invalid timestamps."
            )

        numeric_columns = [
            column
            for column in RAW_SOURCE_COLUMNS
            if column != "timestamp"
        ]

        for column in numeric_columns:

            if column == "city":
                continue

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype("float64")

        df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        return df


    # =================================================================
    # 9. VALIDATE RAW DATASET
    # =================================================================

    def validate_raw_dataframe(
        self,
        raw_df: pd.DataFrame,
    ) -> None:
        """
        Ensure the raw context is suitable for feature engineering.
        """

        logger.info(
            "Validating gap-fill raw dataframe..."
        )

        required_columns = [
            "city",
            *RAW_SOURCE_COLUMNS,
        ]

        missing = [
            column
            for column in required_columns
            if column not in raw_df.columns
        ]

        if missing:

            raise GapFillPipelineError(
                "Raw dataframe is missing columns: "
                + ", ".join(missing)
            )

        if raw_df["timestamp"].duplicated().any():

            raise GapFillPipelineError(
                "Raw dataframe contains duplicate timestamps."
            )

        numeric_columns = [
            column
            for column in RAW_SOURCE_COLUMNS
            if column != "timestamp"
        ]

        missing_counts = (
            raw_df[numeric_columns]
            .isna()
            .sum()
        )

        missing_values = (
            missing_counts[
                missing_counts > 0
            ]
        )

        if not missing_values.empty:

            details = ", ".join(
                f"{column}={int(count)}"
                for column, count
                in missing_values.items()
            )

            raise GapFillPipelineError(
                "Raw dataframe contains missing values: "
                + details
            )

        intervals = (
            raw_df["timestamp"]
            .diff()
            .dropna()
        )

        if not intervals.empty:

            if not intervals.eq(
                pd.Timedelta(hours=1)
            ).all():

                raise GapFillPipelineError(
                    "Gap-fill raw context is not a continuous "
                    "hourly time series."
                )

        if len(raw_df) <= MAX_LOOKBACK_HOURS:

            raise GapFillPipelineError(
                "Insufficient raw context for feature engineering."
            )

        logger.info(
            "Raw dataframe validation PASSED."
        )

        logger.info(
            "Raw rows: %s",
            len(raw_df),
        )

        logger.info(
            "Raw range: %s → %s",
            raw_df["timestamp"].iloc[0],
            raw_df["timestamp"].iloc[-1],
        )


    # =================================================================
    # 10. BUILD COMPUTED FEATURES
    # =================================================================

    def build_computed_features(
        self,
        raw_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the exact shared feature engineering.

        No feature definitions are implemented in this file.
        """

        logger.info(
            "=================================================="
        )

        logger.info(
            "RUNNING SHARED FEATURE ENGINEERING"
        )

        logger.info(
            "=================================================="
        )

        features = build_rich_features(
            raw_df
        )

        logger.info(
            "Generated feature frame: %s rows × %s columns",
            features.shape[0],
            features.shape[1],
        )

        if len(MODEL_FEATURES) != 100:

            raise GapFillPipelineError(
                "Expected exactly 100 MODEL_FEATURES, "
                f"found {len(MODEL_FEATURES)}."
            )

        actual_model_features = [
            column
            for column in features.columns
            if column in MODEL_FEATURES
        ]

        if actual_model_features != list(
            MODEL_FEATURES
        ):

            raise GapFillPipelineError(
                "Generated feature order does not match "
                "canonical MODEL_FEATURES."
            )

        logger.info(
            "100-feature schema check PASSED."
        )

        # -------------------------------------------------------------
        # Remove warm-up rows.
        # -------------------------------------------------------------

        complete_features = features.dropna(
            subset=list(MODEL_FEATURES)
        ).copy()

        if complete_features.empty:

            raise GapFillPipelineError(
                "No complete computed feature rows were produced."
            )

        # -------------------------------------------------------------
        # Validate computed features.
        # -------------------------------------------------------------

        validate_feature_frame(
            complete_features,
            require_complete=True,
        )

        logger.info(
            "Canonical feature validation PASSED."
        )

        output_columns = [
            "city",
            "timestamp",
            "target_aqi",
            *MODEL_FEATURES,
        ]

        missing = [
            column
            for column in output_columns
            if column not in complete_features.columns
        ]

        if missing:

            raise GapFillPipelineError(
                "Computed feature frame is missing: "
                + ", ".join(missing)
            )

        computed = complete_features[
            output_columns
        ].copy()

        if computed[
            list(MODEL_FEATURES)
        ].isna().any().any():

            raise GapFillPipelineError(
                "Computed feature dataset contains NaN values."
            )

        logger.info(
            "Computed dataset: %s rows × %s columns",
            len(computed),
            len(computed.columns),
        )

        return computed


    # =================================================================
    # 11. KEEP ONLY GAP ROWS
    # =================================================================

    def select_missing_rows(
        self,
        computed: pd.DataFrame,
        existing_v6: pd.DataFrame,
        gap_start: pd.Timestamp,
        gap_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Keep only timestamps inside the gap that are not already
        present in V6.

        This is the main idempotency protection.
        """

        logger.info(
            "Selecting only missing timestamps..."
        )

        computed = computed.copy()

        computed["timestamp"] = pd.to_datetime(
            computed["timestamp"],
            utc=True,
        )

        existing_timestamps = set(
            existing_v6["timestamp"]
        )

        # -------------------------------------------------------------
        # First restrict to the actual gap.
        # -------------------------------------------------------------

        candidates = computed[
            (
                computed["timestamp"]
                >= gap_start
            )
            & (
                computed["timestamp"]
                <= gap_end
            )
        ].copy()

        logger.info(
            "Computed rows inside requested gap: %s",
            len(candidates),
        )

        # -------------------------------------------------------------
        # Then remove anything already present in V6.
        # -------------------------------------------------------------

        missing_rows = candidates[
            ~candidates["timestamp"].isin(
                existing_timestamps
            )
        ].copy()

        missing_rows = (
            missing_rows
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        logger.info(
            "Rows genuinely missing from V6: %s",
            len(missing_rows),
        )

        if not missing_rows.empty:

            logger.info(
                "Missing range: %s → %s",
                missing_rows["timestamp"].iloc[0],
                missing_rows["timestamp"].iloc[-1],
            )

        return missing_rows


    # =================================================================
    # 12. VALIDATE GAP BEFORE INSERT
    # =================================================================

    def validate_missing_rows(
        self,
        missing_rows: pd.DataFrame,
        gap_start: pd.Timestamp,
        gap_end: pd.Timestamp,
    ) -> None:
        """
        Perform final safety checks before writing to Hopsworks.
        """

        if missing_rows.empty:

            logger.info(
                "No rows require insertion."
            )

            return

        # -------------------------------------------------------------
        # No duplicate timestamps in upload batch.
        # -------------------------------------------------------------

        if missing_rows["timestamp"].duplicated().any():

            raise GapFillPipelineError(
                "Upload batch contains duplicate timestamps."
            )

        # -------------------------------------------------------------
        # Upload timestamps must be continuous.
        # -------------------------------------------------------------

        intervals = (
            missing_rows["timestamp"]
            .diff()
            .dropna()
        )

        if not intervals.empty:

            if not intervals.eq(
                pd.Timedelta(hours=1)
            ).all():

                raise GapFillPipelineError(
                    "Rows selected for insertion are not "
                    "a continuous hourly sequence."
                )

        # -------------------------------------------------------------
        # No NaN in canonical features.
        # -------------------------------------------------------------

        if missing_rows[
            list(MODEL_FEATURES)
        ].isna().any().any():

            raise GapFillPipelineError(
                "Upload batch contains NaN values "
                "in MODEL_FEATURES."
            )

        # -------------------------------------------------------------
        # All rows must be inside requested gap.
        # -------------------------------------------------------------

        outside_gap = (
            (missing_rows["timestamp"] < gap_start)
            | (missing_rows["timestamp"] > gap_end)
        )

        if outside_gap.any():

            raise GapFillPipelineError(
                "Upload batch contains timestamps outside "
                "the calculated gap."
            )

        logger.info(
            "Final upload validation PASSED."
        )


    # =================================================================
    # 13. INSERT MISSING ROWS
    # =================================================================

    def insert_missing_rows(
        self,
        feature_group,
        missing_rows: pd.DataFrame,
    ) -> None:
        """
        Insert only the missing computed rows.

        Existing V6 rows are never deleted or modified.
        """

        if missing_rows.empty:

            logger.info(
                "Nothing to insert into v6."
            )

            return

        logger.info(
            "Preparing to insert %s missing rows into v6...",
            len(missing_rows),
        )

        dataframe = missing_rows.copy()

        # -------------------------------------------------------------
        # Ensure Hopsworks receives the expected timestamp type.
        # -------------------------------------------------------------

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            utc=True,
        )

        # -------------------------------------------------------------
        # Exact output schema.
        # -------------------------------------------------------------

        output_columns = [
            "city",
            "timestamp",
            "target_aqi",
            *MODEL_FEATURES,
        ]

        dataframe = dataframe[
            output_columns
        ].copy()

        # -------------------------------------------------------------
        # Upload in manageable batches.
        # -------------------------------------------------------------

        batch_size = 1000

        max_retries = 3

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

            logger.info(
                "Inserting rows %s → %s",
                start,
                end,
            )

            uploaded = False

            for attempt in range(
                1,
                max_retries + 1,
            ):

                try:

                    logger.info(
                        "Upload attempt %s/%s",
                        attempt,
                        max_retries,
                    )

                    feature_group.insert(
                        batch,
                        write_options={
                            "wait_for_job": True,
                        },
                    )

                    uploaded = True

                    logger.info(
                        "Batch %s → %s uploaded successfully.",
                        start,
                        end,
                    )

                    break

                except Exception as exc:

                    logger.error(
                        "Upload attempt %s failed: %s",
                        attempt,
                        exc,
                    )

                    if attempt == max_retries:

                        raise GapFillPipelineError(
                            f"Failed to upload batch "
                            f"{start} → {end} after "
                            f"{max_retries} attempts."
                        ) from exc

                    logger.info(
                        "Retrying batch..."
                    )

            if not uploaded:

                raise GapFillPipelineError(
                    f"Batch {start} → {end} was not uploaded."
                )


    # =================================================================
    # 14. COMPLETE RUN
    # =================================================================

    def run(
        self,
        write_to_feature_store: bool = True,
    ) -> pd.DataFrame:
        """
        Execute the complete gap-fill process.

        Returns:
            DataFrame containing only the rows that were selected
            for insertion.
        """

        logger.info(
            "=================================================="
        )

        logger.info(
            "KARACHI AQI V6 GAP-FILL PIPELINE"
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
        # Read existing V6.
        # -------------------------------------------------------------

        existing_v6 = self.read_existing_v6(
            feature_group
        )

        # -------------------------------------------------------------
        # Determine missing period.
        # -------------------------------------------------------------

        gap_start, gap_end = (
            self.determine_gap(
                existing_v6
            )
        )

        if gap_start > gap_end:

            logger.info(
                "V6 is already caught up to the safe historical "
                "boundary. No gap-fill required."
            )

            return pd.DataFrame()

        # -------------------------------------------------------------
        # Determine API fetch window.
        # -------------------------------------------------------------

        fetch_start, fetch_end = (
            self.determine_fetch_window(
                gap_start,
                gap_end,
            )
        )

        # -------------------------------------------------------------
        # Fetch raw data.
        # -------------------------------------------------------------

        air_quality = self.fetch_air_quality(
            start_date=fetch_start.strftime("%Y-%m-%d"),
            end_date=fetch_end.strftime("%Y-%m-%d"),
        )

        weather = self.fetch_weather(
            start_date=fetch_start.strftime("%Y-%m-%d"),
            end_date=fetch_end.strftime("%Y-%m-%d"),
        )

        # -------------------------------------------------------------
        # Build canonical raw dataframe.
        # -------------------------------------------------------------

        raw_df = self.build_raw_dataframe(
            air_quality,
            weather,
        )

        # -------------------------------------------------------------
        # Validate raw dataframe.
        # -------------------------------------------------------------

        self.validate_raw_dataframe(
            raw_df
        )

        # -------------------------------------------------------------
        # Ensure raw API window covers the exact required period.
        # -------------------------------------------------------------

        if raw_df["timestamp"].min() > fetch_start:

            raise GapFillPipelineError(
                "Open-Meteo returned data later than the "
                "required context start."
            )

        if raw_df["timestamp"].max() < gap_end:

            raise GapFillPipelineError(
                "Open-Meteo did not return data through "
                "the required gap end."
            )

        # -------------------------------------------------------------
        # Run shared feature engineering.
        # -------------------------------------------------------------

        computed = self.build_computed_features(
            raw_df
        )

        # -------------------------------------------------------------
        # Select ONLY genuinely missing rows.
        # -------------------------------------------------------------

        missing_rows = self.select_missing_rows(
            computed=computed,
            existing_v6=existing_v6,
            gap_start=gap_start,
            gap_end=gap_end,
        )

        # -------------------------------------------------------------
        # Final validation.
        # -------------------------------------------------------------

        self.validate_missing_rows(
            missing_rows,
            gap_start,
            gap_end,
        )

        # -------------------------------------------------------------
        # Dry-run mode.
        # -------------------------------------------------------------

        if not write_to_feature_store:

            logger.info(
                "DRY RUN: no data will be written to Hopsworks."
            )

            return missing_rows

        # -------------------------------------------------------------
        # Insert only missing rows.
        # -------------------------------------------------------------

        self.insert_missing_rows(
            feature_group,
            missing_rows,
        )

        logger.info(
            "=================================================="
        )

        logger.info(
            "V6 GAP-FILL COMPLETED SUCCESSFULLY"
        )

        logger.info(
            "=================================================="
        )

        logger.info(
            "Inserted rows: %s",
            len(missing_rows),
        )

        return missing_rows


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================

if __name__ == "__main__":

    pipeline = AQIGapFillPipeline(
        city="karachi"
    )

    try:

        # -------------------------------------------------------------
        # SAFETY FIRST:
        #
        # The first execution is intentionally a DRY RUN.
        #
        # It calculates and validates the missing rows but DOES NOT
        # write anything to Hopsworks.
        #
        # After reviewing the output, change this to True.
        # -------------------------------------------------------------

        rows = pipeline.run(
            write_to_feature_store=True
        )

        print(
            "\n=============================================="
        )

        print(
            " V6 GAP-FILL DRY RUN COMPLETED"
        )

        print(
            "=============================================="
        )

        if rows.empty:

            print(
                "\nNo missing rows require insertion."
            )

        else:

            print(
                "\nRows ready for insertion:",
                len(rows),
            )

            print(
                "First missing timestamp:",
                rows["timestamp"].iloc[0],
            )

            print(
                "Last missing timestamp:",
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
                "\nPreview:"
            )

            print(
                rows[
                    [
                        "city",
                        "timestamp",
                        "target_aqi",
                    ]
                    + list(MODEL_FEATURES)
                ].head()
            )

            print(
                "\nNo rows were written to Hopsworks."
            )

            print(
                "\nIf this looks correct, change:"
            )

            print(
                "    write_to_feature_store=False"
            )

            print(
                "to:"
            )

            print(
                "    write_to_feature_store=True"
            )

            print(
                "\nand run the script again."
            )

    except Exception as exc:

        logger.exception(
            "V6 gap-fill pipeline failed: %s",
            exc,
        )

        raise


"""Shared rich feature engineering for hourly Karachi AQI data."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_RAW_COLUMNS = (
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

BASE_FEATURES = (
    "pm25",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
    "temperature",
    "humidity",
    "humidex",
    "aqi_change_rate",
    "hour",
    "day",
    "month",
    "day_of_week",
)

INTERACTION_FEATURES = (
    "pm25_humidity_interaction",
    "pm10_humidity_interaction",
)

HISTORICAL_SOURCE_COLUMNS = (
    "target_aqi",
    "temperature",
    "humidity",
    "pm25",
    "pm10",
    "humidex",
)

LAG_PERIODS = (1, 2, 3, 24, 48, 72, 168)
ROLLING_MEAN_WINDOWS = (6, 12, 24, 48, 72, 168)
ROLLING_STD_WINDOWS = (24,)
MAX_LOOKBACK_HOURS = 168


def _historical_feature_names() -> tuple[str, ...]:
    names = []
    for source_column in HISTORICAL_SOURCE_COLUMNS:
        names.extend(
            f"{source_column}_lag_{period}"
            for period in LAG_PERIODS
        )
        names.extend(
            f"{source_column}_rolling_mean_{window}"
            for window in ROLLING_MEAN_WINDOWS
        )
        names.extend(
            f"{source_column}_rolling_std_{window}"
            for window in ROLLING_STD_WINDOWS
        )
    return tuple(names)


MODEL_FEATURES = (
    BASE_FEATURES
    + INTERACTION_FEATURES
    + _historical_feature_names()
)

if len(MODEL_FEATURES) != 100:
    raise RuntimeError(
        f"Expected 100 canonical model features, found {len(MODEL_FEATURES)}"
    )

TARGET_COLUMNS = ("target_day1", "target_day2", "target_day3")


class FeatureEngineeringError(ValueError):
    """Raised when raw data cannot satisfy the feature contract."""


def get_feature_contract() -> dict[str, object]:
    """Return the contract metadata used by training and serving."""
    return {
        "required_raw_columns": REQUIRED_RAW_COLUMNS,
        "model_features": MODEL_FEATURES,
        "model_feature_count": len(MODEL_FEATURES),
        "historical_source_columns": HISTORICAL_SOURCE_COLUMNS,
        "lag_periods": LAG_PERIODS,
        "rolling_mean_windows": ROLLING_MEAN_WINDOWS,
        "rolling_std_windows": ROLLING_STD_WINDOWS,
        "max_lookback_hours": MAX_LOOKBACK_HOURS,
        "frequency": "1h",
        "timestamp_timezone": "UTC",
    }


def normalize_and_validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps and validate a sorted, continuous hourly input."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing_columns = [
        column for column in REQUIRED_RAW_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise FeatureEngineeringError(
            "Missing required raw columns: " + ", ".join(missing_columns)
        )

    normalized = data.copy()
    timestamp_values = normalized["timestamp"]
    if pd.api.types.is_numeric_dtype(timestamp_values):
        normalized["timestamp"] = pd.to_datetime(
            timestamp_values, unit="ms", utc=True, errors="coerce"
        )
    else:
        normalized["timestamp"] = pd.to_datetime(
            timestamp_values, utc=True, errors="coerce"
        )

    if normalized["timestamp"].isna().any():
        raise FeatureEngineeringError("timestamp contains unparseable values")
    if normalized["timestamp"].duplicated().any():
        raise FeatureEngineeringError("timestamp contains duplicate values")
    if not normalized["timestamp"].is_monotonic_increasing:
        raise FeatureEngineeringError("timestamp must be sorted in ascending order")

    intervals = normalized["timestamp"].diff().dropna()
    if not intervals.empty and not intervals.eq(pd.Timedelta(hours=1)).all():
        raise FeatureEngineeringError(
            "timestamp must form a continuous hourly series without missing hours"
        )

    numeric_columns = [
        column for column in REQUIRED_RAW_COLUMNS if column != "timestamp"
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column], errors="coerce"
        ).astype("float64")

    return normalized


def calculate_humidex(temperature: pd.Series, humidity: pd.Series) -> pd.Series:
    """Calculate Humidex using the project's existing backfill formula."""
    vapor_pressure = (
        6.11
        * (10 ** ((7.5 * temperature) / (237.7 + temperature)))
        * (humidity / 100.0)
    )
    return temperature + (5 / 9) * (vapor_pressure - 10)


def build_rich_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical rich feature frame without generating targets."""
    present_targets = [column for column in TARGET_COLUMNS if column in data.columns]
    if present_targets:
        raise FeatureEngineeringError(
            "Target columns must be added outside feature engineering: "
            + ", ".join(present_targets)
        )

    features = normalize_and_validate_input(data)
    derived = {
        "humidex": calculate_humidex(
            features["temperature"], features["humidity"]
        ),
        "aqi_change_rate": features["target_aqi"].diff(),
        "hour": features["timestamp"].dt.hour.astype("int64"),
        "day": features["timestamp"].dt.day.astype("int64"),
        "month": features["timestamp"].dt.month.astype("int64"),
        "day_of_week": features["timestamp"].dt.dayofweek.astype("int64"),
        "pm25_humidity_interaction": features["pm25"] * features["humidity"],
        "pm10_humidity_interaction": features["pm10"] * features["humidity"],
    }
    features = pd.concat([features, pd.DataFrame(derived, index=features.index)], axis=1)

    historical_derived = {}
    for source_column in HISTORICAL_SOURCE_COLUMNS:
        for period in LAG_PERIODS:
            historical_derived[f"{source_column}_lag_{period}"] = (
                features[source_column].shift(period)
            )
        shifted = features[source_column].shift(1)
        for window in ROLLING_MEAN_WINDOWS:
            historical_derived[f"{source_column}_rolling_mean_{window}"] = (
                shifted.rolling(window=window).mean()
            )
        for window in ROLLING_STD_WINDOWS:
            historical_derived[f"{source_column}_rolling_std_{window}"] = (
                shifted.rolling(window=window).std()
            )
    features = pd.concat(
        [features, pd.DataFrame(historical_derived, index=features.index)], axis=1
    )

    output_columns = [
        column for column in ("city", "timestamp") if column in features.columns
    ]
    output_columns.extend(BASE_FEATURES)
    output_columns.append("target_aqi")
    output_columns.extend(INTERACTION_FEATURES)
    output_columns.extend(_historical_feature_names())
    return features[output_columns]


def validate_feature_frame(
    data: pd.DataFrame,
    *,
    require_complete: bool = False,
) -> None:
    """Validate model columns, dtypes, order, and optional warm-up completeness."""
    missing_features = [
        column for column in MODEL_FEATURES if column not in data.columns
    ]
    if missing_features:
        raise FeatureEngineeringError(
            "Missing canonical model features: " + ", ".join(missing_features)
        )

    forbidden_features = [column for column in TARGET_COLUMNS if column in data.columns]
    if forbidden_features:
        raise FeatureEngineeringError(
            "Target columns are not part of the shared feature frame: "
            + ", ".join(forbidden_features)
        )

    if require_complete:
        null_counts = data[list(MODEL_FEATURES)].isna().sum()
        incomplete = null_counts[null_counts > 0]
        if not incomplete.empty:
            raise FeatureEngineeringError(
                "Feature frame contains insufficient warm-up history or missing "
                "values: "
                + ", ".join(
                    f"{column}={int(count)}"
                    for column, count in incomplete.items()
                )
            )

    temporal_features = {"hour", "day", "month", "day_of_week"}
    bad_float_dtypes = [
        column
        for column in MODEL_FEATURES
        if column not in temporal_features
        and data[column].dtype != np.dtype("float64")
    ]
    if bad_float_dtypes:
        raise FeatureEngineeringError(
            "Expected float64 model features: " + ", ".join(bad_float_dtypes)
        )

    bad_integer_dtypes = [
        column
        for column in temporal_features
        if data[column].dtype != np.dtype("int64")
    ]
    if bad_integer_dtypes:
        raise FeatureEngineeringError(
            "Expected int64 temporal features: " + ", ".join(bad_integer_dtypes)
        )

    actual_order = [column for column in data.columns if column in MODEL_FEATURES]
    if actual_order != list(MODEL_FEATURES):
        raise FeatureEngineeringError(
            "Canonical model features are not in the required order"
        )


__all__ = [
    "BASE_FEATURES",
    "FeatureEngineeringError",
    "HISTORICAL_SOURCE_COLUMNS",
    "INTERACTION_FEATURES",
    "LAG_PERIODS",
    "MAX_LOOKBACK_HOURS",
    "MODEL_FEATURES",
    "REQUIRED_RAW_COLUMNS",
    "ROLLING_MEAN_WINDOWS",
    "ROLLING_STD_WINDOWS",
    "build_rich_features",
    "calculate_humidex",
    "get_feature_contract",
    "normalize_and_validate_input",
    "validate_feature_frame",
]
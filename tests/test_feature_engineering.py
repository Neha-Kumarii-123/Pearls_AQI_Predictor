import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_engineering import (
    BASE_FEATURES,
    FeatureEngineeringError,
    HISTORICAL_SOURCE_COLUMNS,
    INTERACTION_FEATURES,
    LAG_PERIODS,
    MODEL_FEATURES,
    ROLLING_MEAN_WINDOWS,
    ROLLING_STD_WINDOWS,
    build_rich_features,
    calculate_humidex,
    normalize_and_validate_input,
    validate_feature_frame,
)


def make_fixture(rows=180):
    timestamp = pd.date_range(
        "2024-08-01 00:00:00", periods=rows, freq="h", tz="UTC"
    )
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "pm25": index + 10,
            "pm10": index + 20,
            "ozone": index + 30,
            "nitrogen_dioxide": index + 40,
            "sulphur_dioxide": index + 50,
            "carbon_monoxide": index + 60,
            "temperature": index + 25,
            "humidity": index + 70,
            "target_aqi": index + 100,
        }
    )


def test_exact_canonical_count_and_names():
    assert len(MODEL_FEATURES) == 100
    assert len(set(MODEL_FEATURES)) == 100
    assert MODEL_FEATURES[: len(BASE_FEATURES)] == BASE_FEATURES
    assert MODEL_FEATURES[len(BASE_FEATURES) : len(BASE_FEATURES) + 2] == (
        INTERACTION_FEATURES
    )
    expected = list(BASE_FEATURES) + list(INTERACTION_FEATURES)
    for source in HISTORICAL_SOURCE_COLUMNS:
        expected.extend(f"{source}_lag_{period}" for period in LAG_PERIODS)
        expected.extend(
            f"{source}_rolling_mean_{window}"
            for window in ROLLING_MEAN_WINDOWS
        )
        expected.extend(
            f"{source}_rolling_std_{window}"
            for window in ROLLING_STD_WINDOWS
        )
    assert list(MODEL_FEATURES) == expected


def test_required_input_columns_are_validated():
    data = make_fixture().drop(columns="ozone")
    with pytest.raises(FeatureEngineeringError, match="ozone"):
        build_rich_features(data)


def test_timestamps_are_normalized_to_utc():
    data = make_fixture(rows=3)
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    normalized = normalize_and_validate_input(data)
    assert str(normalized["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert normalized["timestamp"].dt.tz is not None


def test_duplicate_timestamps_are_rejected():
    data = make_fixture(rows=3)
    data.loc[2, "timestamp"] = data.loc[1, "timestamp"]
    with pytest.raises(FeatureEngineeringError, match="duplicate"):
        normalize_and_validate_input(data)


def test_missing_hours_are_rejected():
    data = make_fixture().drop(index=10).reset_index(drop=True)
    with pytest.raises(FeatureEngineeringError, match="continuous hourly"):
        normalize_and_validate_input(data)


def test_unsorted_timestamps_are_rejected():
    data = make_fixture().iloc[[1, 0] + list(range(2, 180))].reset_index(drop=True)
    with pytest.raises(FeatureEngineeringError, match="sorted"):
        normalize_and_validate_input(data)


def test_lags_are_causal_and_correct():
    features = build_rich_features(make_fixture())
    assert features.loc[5, "target_aqi_lag_1"] == 104
    assert features.loc[5, "target_aqi_lag_3"] == 102
    assert features.loc[72, "pm25_lag_72"] == 10
    assert features.loc[168, "humidity_lag_168"] == 70


def test_rolling_features_exclude_current_observation():
    features = build_rich_features(make_fixture())
    expected = pd.Series(np.arange(0, 6, dtype=float) + 100)
    assert features.loc[6, "target_aqi_rolling_mean_6"] == expected.mean()
    assert features.loc[6, "target_aqi_rolling_std_24"] != 106
    assert features.loc[168, "target_aqi_rolling_mean_168"] == 183.5


def test_humidex_rate_and_interactions_are_correct():
    data = make_fixture()
    features = build_rich_features(data)
    expected_humidex = calculate_humidex(data["temperature"], data["humidity"])
    pd.testing.assert_series_equal(
        features["humidex"], expected_humidex, check_names=False
    )
    assert np.isnan(features.loc[0, "aqi_change_rate"])
    assert features.loc[5, "aqi_change_rate"] == 1
    assert features.loc[5, "pm25_humidity_interaction"] == 15 * 75
    assert features.loc[5, "pm10_humidity_interaction"] == 25 * 75


def test_warmup_nan_is_preserved_and_complete_frame_starts_at_168():
    features = build_rich_features(make_fixture(rows=169))
    assert np.isnan(features.loc[0, "target_aqi_lag_1"])
    assert np.isnan(features.loc[167, "target_aqi_lag_168"])
    assert features.loc[168, "target_aqi_lag_168"] == 100
    assert features.loc[168, "target_aqi_rolling_mean_168"] == 183.5
    validate_feature_frame(features.iloc[[168]].copy(), require_complete=True)


def test_no_horizon_or_wind_features_are_generated():
    features = build_rich_features(make_fixture())
    assert not any(column in features.columns for column in (
        "target_day1", "target_day2", "target_day3",
        "wind_speed", "wind_direction",
        "pm10_wind_interaction", "wind_speed_direction",
    ))


def test_target_columns_are_rejected_as_feature_inputs():
    data = make_fixture()
    data["target_day1"] = data["target_aqi"]
    with pytest.raises(FeatureEngineeringError, match="target_day1"):
        build_rich_features(data)


def test_no_future_values_enter_features():
    original = build_rich_features(make_fixture(rows=180))
    changed = make_fixture(rows=180)
    changed.loc[170:, "target_aqi"] = 999999
    changed_features = build_rich_features(changed)
    for column in MODEL_FEATURES:
        pd.testing.assert_series_equal(
            original.loc[:169, column],
            changed_features.loc[:169, column],
            check_names=False,
        )


def test_169_row_fixture_has_all_100_features_at_forecast_origin():
    features = build_rich_features(make_fixture(rows=169))
    forecast_origin = features.iloc[[-1]]
    assert list(forecast_origin[list(MODEL_FEATURES)].columns) == list(MODEL_FEATURES)
    assert forecast_origin[list(MODEL_FEATURES)].notna().all().all()
    validate_feature_frame(forecast_origin, require_complete=True)
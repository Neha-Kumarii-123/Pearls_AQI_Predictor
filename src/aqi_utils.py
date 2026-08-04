from __future__ import annotations

import numpy as np


PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0.0, 54.0, 0, 50),
    (55.0, 154.0, 51, 100),
    (155.0, 254.0, 101, 150),
    (255.0, 354.0, 151, 200),
    (355.0, 424.0, 201, 300),
    (425.0, 504.0, 301, 400),
    (505.0, 604.0, 401, 500),
]


def _concentration_to_aqi(concentration, breakpoints):
    """
    Convert pollutant concentration to a continuous AQI value using EPA-style
    breakpoint interpolation. Supports scalar or NumPy array inputs.
    """
    values = np.asarray(concentration, dtype=float)
    aqi_values = np.full(values.shape, np.nan, dtype=float)

    for c_low, c_high, i_low, i_high in breakpoints:
        mask = (values >= c_low) & (values <= c_high)
        if np.any(mask):
            ratio = (values[mask] - c_low) / (c_high - c_low)
            aqi_values[mask] = (i_high - i_low) * ratio + i_low

    return aqi_values


def calculate_aqi_from_pm(pm25=None, pm10=None):
    """
    Compute a continuous AQI target from PM2.5 and PM10 concentrations.
    Returns the max AQI across the available pollutant inputs to mimic
    regulatory AQI behavior and preserve consistency across live and historical data.
    """
    pollutant_values = []

    if pm25 is not None:
        pollutant_values.append(_concentration_to_aqi(pm25, PM25_BREAKPOINTS))

    if pm10 is not None:
        pollutant_values.append(_concentration_to_aqi(pm10, PM10_BREAKPOINTS))

    if not pollutant_values:
        return None

    combined = np.nanmax(np.stack([np.asarray(v, dtype=float) for v in pollutant_values], axis=0), axis=0)
    if np.isscalar(combined):
        return float(combined)

    return combined.astype(float)

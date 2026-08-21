import os
import hopsworks
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables from .env file
load_dotenv()


def fetch_data_for_eda():
    """Connect to Hopsworks and fetch AQI dataset."""
    print("Connecting to Hopsworks Feature Store...")

    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    fs = project.get_feature_store()

    print(
        "Fetching dataset from Feature Group "
        "'karachi_aqi_features' v4..."
    )

    fg = fs.get_feature_group(
        name="karachi_aqi_features",
        version=4
    )

    df = fg.read()

    return df


def analyze_outliers(df):
    """Detect numerical outliers using IQR."""

    print("\n--- Outlier Analysis (IQR Method) ---")

    numerical_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    numerical_cols = [
        col for col in numerical_cols
        if col != "timestamp"
    ]

    outlier_summary = {}

    for col in numerical_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ]

        outlier_count = len(outliers)

        outlier_percentage = (
            outlier_count / len(df)
        ) * 100

        outlier_summary[col] = {
            "outlier_count": outlier_count,
            "percentage": round(
                outlier_percentage,
                2
            )
        }

        print(
            f"Column '{col}': "
            f"{outlier_count} outliers "
            f"({outlier_percentage:.2f}%)"
        )

    return outlier_summary


def analyze_aqi_statistics(df):
    """Analyze AQI distribution."""

    target_col = (
        "target_aqi"
        if "target_aqi" in df.columns
        else "target"
    )

    print("\n--- AQI Statistics ---")

    aqi = df[target_col]

    print(f"Mean     : {aqi.mean():.4f}")
    print(f"Median   : {aqi.median():.4f}")
    print(f"Std      : {aqi.std():.4f}")
    print(f"Minimum  : {aqi.min():.4f}")
    print(f"Maximum  : {aqi.max():.4f}")
    print(f"Skewness : {aqi.skew():.4f}")

    print("\nAQI Percentiles:")

    print(
        aqi.quantile([
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ])
    )


def analyze_time_patterns(df):
    """Analyze hourly, weekly and monthly AQI patterns."""

    target_col = (
        "target_aqi"
        if "target_aqi" in df.columns
        else "target"
    )

    print("\n--- Time Pattern Analysis ---")

    # Hourly pattern
    if "hour" in df.columns:

        print("\nAverage AQI by Hour:")

        hourly = (
            df.groupby("hour")[target_col]
            .mean()
            .round(2)
        )

        print(hourly)

    # Weekly pattern
    if "day_of_week" in df.columns:

        print("\nAverage AQI by Day of Week:")

        weekly = (
            df.groupby("day_of_week")[target_col]
            .mean()
            .round(2)
        )

        print(weekly)

    # Monthly pattern
    if "month" in df.columns:

        print("\nAverage AQI by Month:")

        monthly = (
            df.groupby("month")[target_col]
            .mean()
            .round(2)
        )

        print(monthly)


def analyze_correlations(df):
    """Find numerical variables correlated with AQI."""

    target_col = (
        "target_aqi"
        if "target_aqi" in df.columns
        else "target"
    )

    print("\n--- Correlation With AQI ---")

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    correlations = (
        numeric_df
        .corr()[target_col]
        .drop(target_col)
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    print(correlations)


def analyze_aqi_autocorrelation(df):
    """Analyze AQI relationship with previous AQI values."""

    target_col = (
        "target_aqi"
        if "target_aqi" in df.columns
        else "target"
    )

    print("\n--- AQI Autocorrelation ---")

    # Make sure data is chronological
    if "timestamp" in df.columns:

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    aqi = df[target_col]

    lags = [
        1,
        3,
        6,
        12,
        24,
        48,
        72,
        168
    ]

    for lag in lags:

        correlation = aqi.autocorr(
            lag=lag
        )

        print(
            f"Lag {lag:>3} hours: "
            f"{correlation:.4f}"
        )


def analyze_forecast_horizons(df):
    """Analyze AQI changes at forecasting horizons."""

    target_col = (
        "target_aqi"
        if "target_aqi" in df.columns
        else "target"
    )

    print("\n--- Forecast Horizon Analysis ---")

    if "timestamp" in df.columns:

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    aqi = df[target_col]

    for horizon in [24, 48, 72]:

        future_aqi = aqi.shift(-horizon)

        change = future_aqi - aqi

        print(
            f"\nHorizon: +{horizon} hours"
        )

        print(
            f"Mean AQI change       : "
            f"{change.mean():.4f}"
        )

        print(
            f"Mean absolute change  : "
            f"{change.abs().mean():.4f}"
        )

        print(
            f"Std of AQI change     : "
            f"{change.std():.4f}"
        )


if __name__ == "__main__":

    # ---------------------------------
    # Fetch dataset
    # ---------------------------------

    df = fetch_data_for_eda()

    # ---------------------------------
    # Basic dataset information
    # ---------------------------------

    print(
        f"\nDataset Shape: {df.shape}"
    )

    print(
        "\nMissing Values per Column:"
    )

    print(
        df.isnull().sum()
    )

    # ---------------------------------
    # Outlier analysis
    # ---------------------------------

    analyze_outliers(df)

    # ---------------------------------
    # AQI statistics
    # ---------------------------------

    analyze_aqi_statistics(df)

    # ---------------------------------
    # Time patterns
    # ---------------------------------

    analyze_time_patterns(df)

    # ---------------------------------
    # Correlation analysis
    # ---------------------------------

    analyze_correlations(df)

    # ---------------------------------
    # AQI temporal behavior
    # ---------------------------------

    analyze_aqi_autocorrelation(df)

    # ---------------------------------
    # Forecast horizon analysis
    # ---------------------------------

    analyze_forecast_horizons(df)

    print("\n========================================")
    print("EDA COMPLETE")
    print("========================================")

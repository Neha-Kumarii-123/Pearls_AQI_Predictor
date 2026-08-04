import os
import datetime
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Open-Meteo API Client with Caching & Retry mechanism
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def calculate_humidex(temp_c, humidity):
    """Calculates Canadian Humidex from Temperature and Humidity."""
    e = 6.11 * (10 ** ((7.5 * temp_c) / (237.7 + temp_c))) * (humidity / 100.0)
    humidex = temp_c + (5/9) * (e - 10)
    return humidex

def fetch_historical_data(latitude=24.8607, longitude=67.0011, start_date="2024-08-01", end_date="2026-07-25"):
    """
    Fetches hourly historical Air Quality and Weather data for Karachi from Open-Meteo.
    """
    print(f" Fetching historical data for Karachi ({start_date} to {end_date})...")
    
    # 1. Fetch Air Quality historical metrics
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10", "pm2_5", "us_aqi"]
    }
    air_responses = openmeteo.weather_api(air_quality_url, params=air_params)
    air_res = air_responses[0]
    
    hourly_air = air_res.Hourly()
    air_dates = pd.date_range(
        start=pd.to_datetime(hourly_air.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_air.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_air.Interval()),
        inclusive="left"
    )
    
    df_air = pd.DataFrame({
        "timestamp_dt": air_dates,
        "pm25": hourly_air.Variables(1).ValuesAsNumpy(),
        "pm10": hourly_air.Variables(0).ValuesAsNumpy(),
        "target_aqi": hourly_air.Variables(2).ValuesAsNumpy()
    })

    # 2. Fetch Weather historical metrics (Temperature & Humidity)
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m"]
    }
    weather_responses = openmeteo.weather_api(weather_url, params=weather_params)
    weather_res = weather_responses[0]
    
    hourly_weather = weather_res.Hourly()
    weather_dates = pd.date_range(
        start=pd.to_datetime(hourly_weather.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_weather.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_weather.Interval()),
        inclusive="left"
    )
    
    df_weather = pd.DataFrame({
        "timestamp_dt": weather_dates,
        "temperature": hourly_weather.Variables(0).ValuesAsNumpy(),
        "humidity": hourly_weather.Variables(1).ValuesAsNumpy()
    })

    # Merge Air Quality and Weather Dataframes on timestamp
    df = pd.merge(df_air, df_weather, on="timestamp_dt", how="inner")
    
    # Fill missing values if any exist using forward fill
    df = df.ffill().bfill()
    
    return df

def process_features(df):
    """
    Applies feature engineering transformations matching the live pipeline schema.
    Keeps `target_aqi` as the direct source AQI label, without any PM-to-AQI formula.
    """
    print(" Processing features (Humidex, Temporal signals, Rate of Change)...")
    
    # 1. Constant Identifier
    df["city"] = "karachi"

    # Target variable remains the direct AQI value from the source dataset.
    df["target_aqi"] = df["target_aqi"].astype("float64")
    
    # 2. Convert datetime to UTC Unix timestamp in milliseconds
    df["timestamp"] = df["timestamp_dt"].astype('int64') // 10**6
    
    # 3. Domain Metric: Canadian Humidex
    df["humidex"] = calculate_humidex(df["temperature"], df["humidity"])
    
    # 4. Temporal Features
    df["hour"] = df["timestamp_dt"].dt.hour.astype('int64')
    df["day"] = df["timestamp_dt"].dt.day.astype('int64')
    df["month"] = df["timestamp_dt"].dt.month.astype('int64')
    df["day_of_week"] = df["timestamp_dt"].dt.dayofweek.astype('int64')
    
    # 5. Derived Feature: AQI Change Rate (matching slide requirement)
    df["aqi_change_rate"] = df["target_aqi"].diff().fillna(0.0)
    
    # Convert all numeric float columns to float64 (double) to match Hopsworks schema
    float_cols = ["pm25", "pm10", "temperature", "humidity", "humidex", "aqi_change_rate", "target_aqi"]
    df[float_cols] = df[float_cols].astype('float64')
    
    # Clean up columns to match Hopsworks Feature Group schema exactly
    feature_cols = [
        "city", "timestamp", "pm25", "pm10", 
        "temperature", "humidity", "humidex", 
        "aqi_change_rate",
        "hour", "day", "month", "day_of_week", "target_aqi"
    ]
    
    return df[feature_cols]
def upload_to_hopsworks(dataframe):
    """
    Connects to Hopsworks Feature Store and performs a batch upload of historical data.
    """
    print(" Connecting to Hopsworks Feature Store...")
    
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in environment variables!")
        
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    
    # Get or create Feature Group 
    print(" Accessing/Creating Feature Group: karachi_aqi_features (v2)...")
    feature_group = fs.get_or_create_feature_group(
        name="karachi_aqi_features",
        version=2,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="Live weather telemetry & Canadian Humidex domain features for AQI prediction",
        online_enabled=False
    )
    
    # Batch insert historical data using dataframe parameter safely
    print(f" Inserting {len(dataframe)} historical feature rows into Hopsworks...")
    batch_size = 3000
    for i in range(0, len(dataframe), batch_size):
        df_batch = dataframe.iloc[i : i + batch_size]
        print(f"Inserting rows {i} to {i + len(df_batch)} into Hopsworks...")
        feature_group.insert(df_batch, write_options={"wait_for_job": True})
        
    print(" Historical backfill insertion completed successfully!")

if __name__ == "__main__":
    # Fetch 2 years of historical data for Karachi (Aug 2024 to July 2026)
    df_raw = fetch_historical_data(start_date="2024-08-01", end_date="2026-07-25")
    
    # Apply Feature Engineering
    df_processed = process_features(df_raw)
    
    print("\n--- Sample Processed Historical Features ---")
    print(df_processed.head())
    print("-------------------------------------------\n")
    
    # Push to Hopsworks Cloud Feature Store
    upload_to_hopsworks(df_processed)
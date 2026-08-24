import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from time import sleep

LAT = 24.8607
LON = 67.0011

end = datetime.now(timezone.utc)
start = end - timedelta(hours=200)

common = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": start.strftime("%Y-%m-%d"),
    "end_date": end.strftime("%Y-%m-%d"),
    "timezone": "UTC",
}

# -----------------------------
# 1. Air Quality
# -----------------------------
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


# -----------------------------
# 2. Weather
# -----------------------------
print("\n--- Fetching Weather ---")

weather_params = {
    **common,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
    ],
}

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

    except requests.RequestException as e:
        print(f"Weather attempt {attempt + 1} failed:", e)

        if attempt == 2:
            raise

        sleep(5)


# -----------------------------
# 3. Build DataFrames
# -----------------------------
print("\n--- Combining Data ---")

aq_df = pd.DataFrame(aq)
weather_df = pd.DataFrame(weather)

aq_df["time"] = pd.to_datetime(aq_df["time"], utc=True)
weather_df["time"] = pd.to_datetime(weather_df["time"], utc=True)

df = aq_df.merge(
    weather_df,
    on="time",
    how="inner",
)

df = df.rename(
    columns={
        "time": "timestamp",
        "pm2_5": "pm25",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "us_aqi": "target_aqi",
    }
)

print("\nMerged shape:", df.shape)

print("\nColumns:")
print(list(df.columns))

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst timestamp:")
print(df["timestamp"].iloc[0])

print("\nLast timestamp:")
print(df["timestamp"].iloc[-1])

print("\nHourly continuity:")
print(
    df["timestamp"]
    .diff()
    .dropna()
    .eq(pd.Timedelta(hours=1))
    .all()
)

print("\n--- First 5 rows ---")
print(df.head())

print("\nDATA COMBINATION TEST COMPLETE")
import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


print("--- Connecting to Hopsworks ---")

project = hopsworks.login()
fs = project.get_feature_store()


feature_group = fs.get_feature_group(
    name="karachi_aqi_features",
    version=4
)

print("--- Reading Feature Group ---")

df = feature_group.read()

print("\n--- Raw timestamp information ---")

print("dtype:")
print(df["timestamp"].dtype)

print("\nFirst 10 raw timestamp values:")
print(df["timestamp"].head(10).to_string())

print("\nLast 10 raw timestamp values:")
print(df["timestamp"].tail(10).to_string())

print("\n--- After pandas conversion ---")

converted = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

print(converted.head(10).to_string())

print("\n--- Timestamp range ---")

print("Minimum:", converted.min())
print("Maximum:", converted.max())

print("\n--- Invalid timestamps ---")

print(
    "Invalid:",
    converted.isna().sum()
)

print(
    "Total:",
    len(converted)
)
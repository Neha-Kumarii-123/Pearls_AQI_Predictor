import os
import hopsworks
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise ValueError("HOPSWORKS_API_KEY environment variable not set.")

print("Authenticating with Hopsworks Cloud Feature Store...")
project = hopsworks.login(
    api_key_value=api_key,
    host="eu-west.cloud.hopsworks.ai"
)

feature_store = project.get_feature_store()
aqi_fg = feature_store.get_feature_group("karachi_aqi_features", version=2)
df = aqi_fg.read()

# Timestamp ko readable date-time mein convert karein
df['readable_time'] = pd.to_datetime(df['timestamp'], unit='ms')

# Latest rows ko sort karke readable time ke sath print karein
df_sorted = df.sort_values(by="timestamp", ascending=False)

print("\n--- LATEST ROWS WITH EXACT TIMESTAMPS ---")
print(df_sorted[['timestamp', 'readable_time', 'pm25', 'temperature', 'target_aqi']].head(10))
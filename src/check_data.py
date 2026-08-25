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

# --- STEP 2: STEP-BY-STEP SAFE DEBUGGING ---
try:
    print("\n[Step A] Testing metadata fetch for all feature groups...")
    fg_metadata = feature_store.get_feature_groups()
    print(f"-> Success! Found {len(fg_metadata)} feature groups in this store.")
    for fg in fg_metadata:
        print(f"   - Name: {fg.name}, Version: {fg.version}")
except Exception as e:
    print(f"-> Failed at metadata fetch: {e}")

try:
    print("\n[Step B] Attempting to retrieve 'karachi_aqi_features' version 5...")
    aqi_fg = feature_store.get_feature_group("karachi_aqi_features", version=5)
    print("-> Success! Feature Group object retrieved.")
    
    print("\n[Step C] Reading data from Feature Group...")
    df = aqi_fg.select_all().read(read_options={"use_arrow_flight": False})
    print(f"-> Success! Data loaded with shape: {df.shape}")

    # Timestamp ko readable date-time mein convert karein
    df['readable_time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df_sorted = df.sort_values(by="timestamp", ascending=False)

    print("\n--- LATEST ROWS WITH EXACT TIMESTAMPS ---")
    print(df_sorted[['timestamp', 'readable_time', 'pm25', 'temperature', 'target_aqi']].head(10))

except Exception as e:
    print(f"-> Failed during Step B or C: {e}")
import os
import hopsworks
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key, host="eu-west.cloud.hopsworks.ai")
feature_store = project.get_feature_store()

# 1. Get Feature Group
aqi_fg = feature_store.get_feature_group("karachi_aqi_features", version=2)
df = aqi_fg.read()

print(f"Total rows before cleaning: {len(df)}")

# 2. Filter out rows jahan target_aqi 5 se kam hai (jaise 3 aur 4 wali rows)
df_clean = df[df['target_aqi'] > 5].copy()

print(f"Total rows after removing low AQI rows: {len(df_clean)}")

# 3. Cleaned dataframe ko wapas Feature Store par overwrite kar dein
aqi_fg.insert(df_clean, overwrite=True)
print("Successfully removed old open-weather test rows and updated Hopsworks feature group!")
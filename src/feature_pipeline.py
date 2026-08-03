import requests
import os
import sys
import pandas as pd
from dotenv import load_dotenv
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile
from pathlib import Path

# load environment variables from .env file
load_dotenv()

# import Data ingestor from src/ingestor.py
from ingestor import AQICNDataIngestor

#1. Configure production Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger(__name__)

# Import Hopsworks
import hopsworks

#2. Define FeaturePipeline class
class AirQualityFeaturePipeline:
   """
    Production Feature Pipeline for the 10Pearls AQI Predictor.
    Transforms raw telemetry into ML-ready features.
    """
   def __init__(self, city: str = "karachi"):
      """
      initializes the pipeline by instantiating the AQICNDataIngestor.
      """
      self.city = city
      self.ingestor = AQICNDataIngestor(city=city)
      logger.info(f"Initialized AirQualityFeaturePipeline for city: {self.city}")

   def extract_time_features(self) -> Dict[str, Any]:
      """
        Algorithm 1: Temporal Feature Extractor.
        Extracts temporal components (hour, day, month, day_of_week)
        required by project specifications.
        """
      now=datetime.now()
      return{
         "hour": now.hour,
         "day": now.day,
         "month":now.month,
         "day_of_week": now.weekday(),          # 0=Monday, 6=Sunday
         "timestamp": int(now.timestamp()*1000)  # Unix epoch time
        }



   def fetch_openweather_telemetry(self) -> Optional[Dict[str, Any]]:
        """
        Fetches real-time dynamic weather and pollution telemetry directly via OpenWeatherMap API.
        """
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            logger.error("OPENWEATHER_API_KEY is missing in .env file.")
            return None

        # Karachi coordinates (Latitude, Longitude)
        lat, lon = 24.8607, 67.0011
        
        pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"

        try:
            p_res = requests.get(pollution_url, timeout=10).json()
            w_res = requests.get(weather_url, timeout=10).json()

            components = p_res["list"][0]["components"]
            main_data = p_res["list"][0]["main"]

            # Direct target AQI calculation as per mentor guidelines (Scale 1-5 mapped to index range)
            raw_aqi_index = main_data.get("aqi", 3)
            target_aqi = float(raw_aqi_index * 30)  # Maps 1->30, 2->60, 3->90, 4->120, 5->150+

            return {
                "pm25": float(components.get("pm2_5", 25.0)),
                "pm10": float(components.get("pm10", 45.0)),
                "temperature": float(w_res["main"]["temp"]),
                "humidity": float(w_res["main"]["humidity"]),
                "aqi": target_aqi
            }
        except Exception as e:
            logger.error(f"OpenWeather Fetch Error: {e}")
            return None
   def calculate_humidex(self, temp_c: Optional[float], humidity: Optional[float]) -> Optional[float]:
    """
    Algorithm 2: Canadian Humidex Domain Calculation.
    Estimates pollution retention potential based on atmospheric moisture.
    Includes defensive fallbacks if values are missing
    """
    if temp_c is None or humidity is None:
        logger.warning("Missing temperature or humidity: skipping humidex calculation.")
        return None
    try:
        # vapor pressure approximation (e)
        e=(6.11*(10**((7.5*temp_c) / (237.7+temp_c))))*(humidity/100.0)
        humidex= temp_c + (5/9) * (e-10)
        return round(humidex, 2)
    except Exception as err:
        logger.error(f"Failed to calculate Humidex due to math exception: {err}")
        return None
   def generate_feature_vector(self) -> Optional [Dict[str, Any]]:
      """
        Step 3 Orchestrator Algorithm: Merges Stage 0 API telemetry with 
        Stage 1 engineered features to construct a standardized, ML-ready 
        feature vector.
        """
      logger.info(f"Generating live feature vector for target city: {self.city}")

      # 1. Ingest raw telemetry (Primary: OpenWeather, Fallback: AQICN)
      metrics = self.fetch_openweather_telemetry()
      if not metrics:
          logger.warning("OpenWeather ingestion failed; falling back to AQICN ingestor...")
          raw_data = self.ingestor.fetch_live_telemetry()
          if not raw_data:
              logger.error("All data ingestion sources failed; feature vector generation aborted.")
              return None
          metrics = self.ingestor.parse_station_metrics(raw_data)

      # 2. Extract temporal parameters
      time_feats=self.extract_time_features()

      #3. Compute domain science metrics (Canadian Humidex)
      humidex=self.calculate_humidex(
         temp_c=metrics.get("temperature"),
         humidity=metrics.get("humidity")
      )

      # 4. Handle null/missing values gracefully for ML compatibility
      pm25_val = metrics.get("pm25")
      pm10_val = metrics.get("pm10")
      target_aqi_val = metrics.get("aqi")

      # Fallback for PM10 if missing (typically ~1.6x to 2.0x of PM2.5 in urban areas)
      if pm10_val is None and pm25_val is not None:
          pm10_val = round(pm25_val * 1.8, 2)
      elif pm10_val is None:
          pm10_val = 40.0 # Safe default value

      # Target Label Validation (Option A: Pure ML Training)
      if target_aqi_val is None:
          logger.error("Target AQI is missing from API telemetry! Skipping record creation to prevent fake training targets.")
          return None

      # 5. Assemble production feature vector
      feature_vector = {
         # Metadata
         "city": self.city,
         "timestamp": time_feats["timestamp"],

         # Cyclic temporal features
         "hour": time_feats["hour"],
         "day": time_feats["day"],
         "month": time_feats["month"],
         "day_of_week": time_feats["day_of_week"],

         # Physical Telemetry features
         "temperature": metrics.get("temperature", 30.0),
         "humidity": metrics.get("humidity", 60.0),   
         "pm25": pm25_val,
         "pm10": pm10_val,

         # Engineered domain Metric
         "humidex": humidex if humidex is not None else 35.0,
         "aqi_change_rate": 0.0,

         # Target Label
         "target_aqi": target_aqi_val
      }
      logger.info("Successfully assembled production feature vector.")
      return feature_vector
   
   def save_to_feature_store(self, feature_vector: Dict[str, Any]) -> bool:
      """
      Algorithm 4: Hopsworks Cloud Feature Store Integration.
      Connects securely to Hopsworks, registers/fetches the feature Group, 
      and pushes the production feature vector using standard HTTP streaming.
      """
      if not feature_vector:
         logger.warning("No feature vector provided; skipping Hopsworks insertion.")
         return False

      # 1. fetch Hopsworks API key from environment variable 
      api_key=os.getenv("HOPSWORKS_API_KEY")
      if not api_key:
         logger.error("HOPSWORKS_API_KEY is missing from environment variables (.env).")
         raise ValueError("HOPSWORKS_API_KEY environment variable not set.")

      try:
         # 2. Login & Access Feature Store
         logger.info("Authenticating with Hopsworks Cloud Feature Store...")
         project = hopsworks.login(
            api_key_value=api_key,
            host="eu-west.cloud.hopsworks.ai",
            cert_folder=tempfile.gettempdir()
            )
         fs = project.get_feature_store()

         # 3. Convert feature vector dictionary into a Pandas DataFrame
         df = pd.DataFrame([feature_vector])
         # Cast missing/nullable numerical columns explicitly to float64 so Hopsworks detects the schema
         for col in ["pm10", "pm25", "temperature", "humidity", "humidex","aqi_change_rate", "target_aqi"]:
             if col in df.columns:
                df[col] = df[col].astype("float64")

         # 4. Get or Create the Feature Group schema
         logger.info("Registering/fetching Hopsworks Feature Group: karachi_aqi_features")
         aqi_fg = fs.get_or_create_feature_group(
            name="karachi_aqi_features",
            version=4,
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=True,
            description="Live weather telemetry & Canadian Humidex domain features for AQI prediction"
         )

         # 5. Insert / Upsert data into Cloud Feature Store
         logger.info("Pushing feature vector to Hopsworks Feature Store...")
         aqi_fg.insert(
             df, 
             write_options={"wait_for_job": True}
            )
         logger.info("Successfully pushed feature vector to Hopsworks Cloud!")
         return True

      except Exception as err:
         logger.error(f"Failed to push feature vector to Hopsworks: {err}")
         return False



if __name__=="__main__":
   pipeline=AirQualityFeaturePipeline(city="karachi")

   # Test Algorithm 1
   time_features=pipeline.extract_time_features()
   print("\n--- ALGORITHM 1: TIME FEATURES ---")
   print (time_features)

   # Test Algorithm 2 (Simulating 32°C and 75% humidity)
   sample_humidex= pipeline.calculate_humidex(temp_c=32.0, humidity=75.0)
   print("\n--- ALGORITHM 2: HUMIDEX CALCULATED ---")
   print(f"Calculated Humidex: {sample_humidex}°C")

   #Test Step 3: End to End Live feature vector generation
   vector=pipeline.generate_feature_vector()
   print("\n==========================================")
   print("      PRODUCTION LIVE FEATURE VECTOR      ")
   print("==========================================")
   import json
   print(json.dumps(vector, indent=4))

   # Test Step 4: Push vector to Hopsworks Cloud Feature Store
   if vector:
      logger.info("\n--- STEP 4: HOPSWORKS CLOUD INTEGRATION ---")
      success = pipeline.save_to_feature_store(vector)
      if success:
         print("\n✅ Step 4 Complete: Live feature vector is now stored in Hopsworks Cloud!")

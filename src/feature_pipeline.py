import logging
from typing import Dict, Any, Optional
from datetime import datetime

# import Data ingestor from src/ingestor.py
from ingestor import AQICNDataIngestor

#1. Configure production Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger(__name__)

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
         "timestamp": int(now.timestamp())  # Unix epoch time
        }


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
      #1. Ingest raw telemetry via AQICNDataIngestor
      raw_data=self.ingestor.fetch_live_telemetry()
      if not raw_data:
         logger.error("Data ingestion failed; feature vector generation aborted.")
         return None
      metrics=self.ingestor.parse_station_metrics(raw_data)

      # 2. Extract temporal parameters
      time_feats=self.extract_time_features()

      #3. Compute domain science metrics (Canadian Humidex)
      humidex=self.calculate_humidex(
         temp_c=metrics.get("temperature"),
         humidity=metrics.get("humidity")
      )

      #4. Assemble production feature vector
      feature_vector={
         #Metadata
         "city": self.city,
         "timestamp": time_feats["timestamp"],

         # Cyclic temporal features
         "hour": time_feats["hour"],
         "day": time_feats["day"],
         "month": time_feats["month"],
         "day_of_week": time_feats["day_of_week"],

         #Physical Telemetry features
         "temperature": metrics.get("temperature"),
         "humidity": metrics.get("humidity"),   
         "pm25": metrics.get("pm25"),
         "pm10": metrics.get("pm10"),

         #Engineered domain Metric
         "humidex": humidex,

         #Target Label
         "target_aqi": metrics.get("aqi")
      }
      logger.info("Successfully assembled production feature vector.")
      return feature_vector



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

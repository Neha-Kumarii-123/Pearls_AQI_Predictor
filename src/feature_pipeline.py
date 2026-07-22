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

if __name__=="__main__":
   pipeline=AirQualityFeaturePipeline(city="karachi")

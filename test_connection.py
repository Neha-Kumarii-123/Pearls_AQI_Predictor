import os
import logging
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Configure production-style logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AQICNDataIngestor:
    """
    Production-ready data ingestor for fetching environmental parameters 
    from the AQICN API service.
    """
    
    def __init__(self, city: str = "karachi"):
        load_dotenv()
        self.api_token = os.getenv("AQICN_TOKEN")
        self.city = city
        self.base_url = "https://api.waqi.info/feed"
        
        if not self.api_token:
            logger.critical("AQICN_TOKEN environment variable is missing from .env!")
            raise ValueError("Authentication token missing. Check your .env file.")

    def fetch_live_telemetry(self) -> Optional[Dict[str, Any]]:
        """
        Queries the AQICN endpoint and returns structured telemetry payload.
        """
        target_url = f"{self.base_url}/{self.city}/?token={self.api_token}"
        
        try:
            logger.info(f"Initiating telemetry request for location: {self.city.capitalize()}")
            response = requests.get(target_url, timeout=10)
            response.raise_for_status()
            
            payload = response.json()
            if payload.get("status") == "ok":
                logger.info("Successfully received telemetry response from remote station.")
                return payload.get("data", {})
            else:
                logger.error(f"API business logic error: {payload.get('data')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network transport error encountered: {e}")
            return None

    def parse_station_metrics(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts and normalizes raw pollutant readings into model inputs.
        """
        iaqi = raw_data.get("iaqi", {})
        
        # Defensive extraction pattern to handle missing parameters gracefully
        extract_val = lambda key: iaqi.get(key, {}).get("v", None)

        return {
            "station_name": raw_data.get("city", {}).get("name", "Unknown"),
            "aqi_target": raw_data.get("aqi", None),
            "dominant_pollutant": raw_data.get("dominentpol", "N/A"),
            "pm25": extract_val("pm25"),
            "temperature": extract_val("t"),
            "humidity": extract_val("h"),
            "pressure": extract_val("p"),
            "wind_speed": extract_val("w"),
            "dew_point": extract_val("dew")
        }


if __name__ == "__main__":
    # Test execution harness
    ingestor = AQICNDataIngestor(city="karachi")
    telemetry = ingestor.fetch_live_telemetry()
    
    if telemetry:
        parsed_metrics = ingestor.parse_station_metrics(telemetry)
        print("\n" + "="*40)
        print(" 📊 PARSED TELEMETRY METRICS (MODEL INPUTS) ")
        print("="*40)
        for key, val in parsed_metrics.items():
            print(f"{key.upper():<20}: {val}")
        print("="*40 + "\n")
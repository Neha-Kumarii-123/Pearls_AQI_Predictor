import os
import requests
from dotenv import load_dotenv

# 1. Load environment variables from the hidden .env file
load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")

# 2. Define the target city endpoint
CITY = "karachi" 
URL = f"https://api.waqi.info/feed/{CITY}/?token={TOKEN}"

def test_api_connection():
    if not TOKEN:
        print("❌ Error: AQICN_TOKEN not found in .env file. Please check your setup!")
        return

    print(f"📡 Requesting real-time air quality data for {CITY.title()}...")
    
    try:
        response = requests.get(URL, timeout=10)
        res_json = response.json()
        
        if res_json.get("status") == "ok":
            data = res_json["data"]
            
            print("=========================================")
            print("✅ Connection Successful via GUI!")
            print(f"📍 Location: {data['city']['name']}")
            print(f"😷 Current AQI: {data['aqi']}")
            print(f"📊 Primary Pollutant: {data.get('dominentpol', 'N/A')}")
            print("=========================================")
            
            print("\n🔍 Raw Pollutant Breakdown (Model Inputs):")
            iaqi = data.get("iaqi", {})
            for pollutant, val in iaqi.items():
                print(f"   - {pollutant.upper()}: {val.get('v')}")
        else:
            print(f"❌ API Error: {res_json.get('data', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_api_connection()
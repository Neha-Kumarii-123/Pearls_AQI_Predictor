from fastapi import FastAPI
import time
from pathlib import Path
from dotenv import load_dotenv

# Import the robust live inference function from predict.py
from src.predict import predict

# Load environment variables from project root
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=REPO_ROOT / ".env")

app = FastAPI(title="Karachi AQI Predictor API", version="1.0")

# ============================================================
# CACHE CONFIGURATION & GLOBAL STATE
# ============================================================
CACHE_TTL = 3600  # Cache valid for 1 hour
_prediction_cache = None
_prediction_cache_time = 0

# ============================================================
# STARTUP WARM-UP EVENT
# ============================================================
@app.on_event("startup")
async def startup_event():
    """
    Pre-fetch live model predictions on server boot so the first user 
    never experiences a cold-start delay.
    """
    global _prediction_cache, _prediction_cache_time
    print("\n--- Warming up FastAPI prediction cache with live model inference ---")
    try:
        result = predict()
        if result and "error" not in result:
            _prediction_cache = result
            _prediction_cache_time = time.time()
            print("--- Live cache warm-up successful ---")
        else:
            print(f"Cache warm-up returned error: {result.get('error', 'Unknown error')}")
    except Exception as exc:
        print(f"Cache warm-up failed: {exc}")

@app.get("/")
def root():
    return {"message": "Karachi AQI Predictor Backend is running successfully with real-time inference."}

@app.get("/predict")
def get_latest_predictions():
    global _prediction_cache, _prediction_cache_time
    now = time.time()
    
    # 1. Return in-memory cached response if valid (Within TTL)
    if _prediction_cache is not None and (now - _prediction_cache_time < CACHE_TTL):
        print("Serving live prediction from in-memory cache.")
        return _prediction_cache

    # 2. Otherwise, run fresh live inference via predict() and update cache
    try:
        result = predict()
        
        if not result or "error" in result:
            return {"error": result.get("error", "Failed to generate live prediction.")}
            
        # Update cache store and timestamp
        _prediction_cache = result
        _prediction_cache_time = now
        
        return result
    except Exception as exc:
        return {"error": str(exc)}
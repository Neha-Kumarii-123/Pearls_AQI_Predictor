from src.predict import predict, get_latest_v6_row, connect_to_hopsworks
from fastapi import FastAPI
from src.predict import predict

app = FastAPI(title="Karachi AQI Predictor")


@app.get("/")
def root():
    return {"message": "Karachi AQI Predictor API is running"}


@app.get("/predict")
def get_prediction():
    return predict()
@app.get("/current")
def get_current():
    project = connect_to_hopsworks()
    feature_row = get_latest_v6_row(project)

    return {
        "timestamp": feature_row["timestamp"].iloc[0],
        "current_aqi": float(feature_row["target_aqi"].iloc[0]),
    }
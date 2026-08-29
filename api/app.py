from fastapi import FastAPI
from src.predict import predict

app = FastAPI(title="Karachi AQI Predictor")


@app.get("/")
def root():
    return {"message": "Karachi AQI Predictor API is running"}


@app.get("/predict")
def get_prediction():
    return predict()
from src.predict import predict, get_latest_v6_row, connect_to_hopsworks
from fastapi import FastAPI
from src.predict import predict
import pandas as pd

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
@app.get("/history")
def get_history():
    project = connect_to_hopsworks()

    fs = project.get_feature_store()

    feature_group = fs.get_feature_group(
        name="karachi_aqi_features",
        version=6,
    )

    now = pd.Timestamp.now(tz="UTC")

    start_time = now - pd.Timedelta(days=30)

    dataframe = feature_group.read(
        start_time=start_time.to_pydatetime(),
        end_time=now.to_pydatetime(),
        dataframe_type="pandas",
    )

    if dataframe is None or dataframe.empty:
        return {
            "data": []
        }

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["timestamp"]
    )

    dataframe = dataframe[
        dataframe["city"].astype(str).str.lower()
        == "karachi"
    ].copy()

    # ---------------------------------------------------------
    # Keep only the actual historical AQI information
    # required by the dashboard.
    # ---------------------------------------------------------

    history_columns = [
        "timestamp",
        "target_aqi",
        "pm25",
        "pm10",
        "ozone",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "carbon_monoxide",
        "temperature",
        "humidity",
    ]

    available_columns = [
        column
        for column in history_columns
        if column in dataframe.columns
    ]

    dataframe = dataframe[
        available_columns
    ].sort_values("timestamp")

    # ---------------------------------------------------------
    # Aggregate hourly observations to daily values.
    #
    # This dramatically reduces the amount of data sent
    # to Streamlit while preserving the historical trend.
    # ---------------------------------------------------------

    dataframe["date"] = (
        dataframe["timestamp"]
        .dt.date
    )

    daily = (
        dataframe
        .groupby("date", as_index=False)
        .agg(
            {
                "target_aqi": "mean",
                "pm25": "mean",
                "pm10": "mean",
                "ozone": "mean",
                "nitrogen_dioxide": "mean",
                "sulphur_dioxide": "mean",
                "carbon_monoxide": "mean",
                "temperature": "mean",
                "humidity": "mean",
            }
        )
    )

    daily["date"] = daily["date"].astype(str)

    daily = daily.where(
        pd.notna(daily),
        None,
    )

    return {
        "data": daily.to_dict(
            orient="records"
        )
    }
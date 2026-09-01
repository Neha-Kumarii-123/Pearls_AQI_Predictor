
# Karachi AQI Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

## Overview / Abstract

This repository implements an end-to-end Air Quality Index (AQI) forecasting and monitoring system for Karachi, designed to predict near-term air quality conditions across three horizons: Day +1, Day +2, and Day +3. The system combines domain-aware feature engineering, a centralized Hopsworks Feature Store, trained machine learning models, a FastAPI inference layer, and a Streamlit monitoring dashboard for operational visibility.

The core product is not a single notebook prototype but a production-oriented MLOps workflow. Feature generation is centralized in the canonical production pipeline, validated against a 100-feature contract, and pushed to the Hopsworks feature group `karachi_aqi_features` version 6. Inference reads the latest available production feature row, loads the corresponding model versions from the Hopsworks Model Registry, and returns a unified AQI forecast with hazard classification logic for high-risk conditions.

The UI exposes the live forecast, current AQI state, and model explanations in a user-friendly dashboard, while the backend exposes structured APIs for prediction and explainability. This makes the project suitable for both operational monitoring and further experimentation in forecasting and early-warning scenarios.

## System Architecture & Workflow

The system follows a modular architecture that separates data acquisition, feature engineering, model serving, and front-end monitoring.

```text
Open-Meteo / AQI Source Data
            │
            ▼
    src/feature_pipeline.py
            │
            │  - fetch recent hourly weather + pollutant signals
            │  - reconstruct historical context
            │  - build canonical 100-feature contract
            ▼
 Hopsworks Feature Store (karachi_aqi_features v6)
            │
            │  - latest complete feature row
            │  - feature validation and historical context
            ▼
   Model Training / Registration
   src/train_horizons_day1.py
   src/train_day2_ridge.py
   src/train_day3_ridge.py
   src/register_models.py
            │
            │  - model artifacts in Hopsworks registry
            ▼
      src/predict.py (inference engine)
            │
            │  - reads latest row from feature store
            │  - validates model feature contract
            │  - loads day1/day2/day3 models
            ▼
   web_app/backend_api.py (FastAPI)
            │
            │  - /predict: live AQI + hazard status
            │  - /explain: SHAP-based feature attribution
            ▼
    web_app/app.py (Streamlit dashboard)
            │
            │  - live forecast cards
            │  - alerting and AQI categories
            │  - EDA visual analytics
            ▼
        End-user monitoring interface
```

### Data Flow

1. The pipeline in `src/feature_pipeline.py` pulls recent environmental data from Open-Meteo and ingests the latest hourly observations into the production feature group.
2. Feature generation is defined in `src/feature_engineering.py` and enforces the canonical contract used across training and inference. This contract includes the 100 engineered features, including lag features, rolling statistics, humidex, temporal indicators, and interaction terms.
3. Model training scripts produce horizon-specific models for Day +1, Day +2, and Day +3. These are registered through `src/register_models.py` into the Hopsworks Model Registry.
4. `src/predict.py` retrieves the newest feature row from Hopsworks, validates schema compatibility, and executes all three models to produce a multi-horizon AQI forecast.
5. The FastAPI service in `web_app/backend_api.py` exposes the inference results via HTTP endpoints and caches responses for efficient repeated access.
6. The Streamlit dashboard in `web_app/app.py` visualizes the predictions, explains feature contribution, and surfaces alert conditions for hazardous AQI levels.

## Key Features

- Automated feature engineering pipeline for hourly AQI and meteorological data.
- Production-ready feature store integration with Hopsworks, versioned feature groups, and model registration.
- Three-horizon forecasting through separate Day +1, Day +2, and Day +3 models.
- Canonical 100-feature contract to keep training and serving strictly aligned.
- Hazard threshold logic that flags severe AQI conditions when observed or forecast AQI exceeds critical ranges.
- Real-time FastAPI inference service with caching and startup warm-up.
- SHAP-based explainability for model-driven factor analysis.
- Interactive EDA dashboard with historical AQI trend, hour-of-day variation, weekday effects, and pollutant relationship views.
- Reproducible training and validation workflow for model version comparison and deployment.

## Tech Stack

### Core Programming & ML

- Python 3.10+
- Pandas and NumPy for feature generation and numerical analysis
- Scikit-learn for regression and pipeline baselines
- XGBoost for the Day +1 forecasting model
- Joblib for model persistence and serialization
- SHAP for explainability and feature attribution

### Feature Store & MLOps

- Hopsworks Feature Store
- Hopsworks Model Registry
- Open-Meteo API data integration
- Feature versioning aligned to `karachi_aqi_features` v6

### Backend & Frontend

- FastAPI for the prediction and explanation APIs
- Streamlit for the interactive AQI dashboard
- Requests for API calls between frontend and backend
- dotenv for environment-driven configuration

### Data Engineering & Utilities

- Python standard library and structured logging
- Temporary certificate handling for cloud authentication
- Production validation logic for time-series continuity and schema integrity

## Project Directory Structure

```text
Pearls_AQI_Predictor-main/
├── README.md
├── Report.md
├── requirements.txt
├── .env
├── .gitignore
├── .python-version
├── karachi_aqi_day1_metrics_v6.json
├── karachi_aqi_day2_metrics_v6.json
├── karachi_aqi_day3_metrics_v6.json
├── notebooks/
│   └── eda_karachi_aqi_features.ipynb
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── feature_pipeline.py
│   ├── gap_fill_pipeline.py
│   ├── backfill_pipeline.py
│   ├── predict.py
│   ├── explainability.py
│   ├── register_models.py
│   ├── train_horizons_day1.py
│   ├── train_day2_ridge.py
│   ├── train_day3_ridge.py
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_live_data.py
│   ├── test_live_feature_compatibility.py
│   ├── test_target_aqi.py
│   ├── test_v5_feature_engineering.py
│   └── test_v6_feature_store.py
├── web_app/
│   ├── app.py
│   ├── backend_api.py
│   └── assets/
│       └── eda/
│           ├── aqi_by_day_of_week.png
│           ├── aqi_by_hour.png
│           ├── aqi_distribution.png
│           ├── aqi_historical_trend.png
│           ├── aqi_vs_pollutants.png
│           └── high_aqi_analysis.png
└── .github/
    └── workflows/
```

## Prerequisites & Environment Setup

### Required Software

- Python 3.10 or above
- Pip package manager
- Access to a valid Hopsworks account and API key
- Internet access for Open-Meteo and Hopsworks connectivity

### Environment Variables

Create a `.env` file in the repository root with the required credentials:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
AQI_API_URL=http://localhost:8000
```

The project also expects valid Hopsworks host configuration and model artifacts to already exist in the configured environment. The inference logic reads the current feature group and the registered model versions from Hopsworks.

## Installation & Running Guide

### 1. Clone the repository

```bash
git clone <repository-url>
cd Pearls_AQI_Predictor-main
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

If no `.env.example` exists, create `.env` manually and set the required variables as described above.

### 5. Start the FastAPI backend

From the repository root:

```bash
uvicorn web_app.backend_api:app --host 0.0.0.0 --port 8000 --reload
```

This starts the API that serves:

- `GET /` for service health
- `GET /predict` for live AQI predictions and alert evaluation
- `GET /explain` for SHAP-based feature explanations

### 6. Launch the Streamlit frontend

In a second terminal:

```bash
export AQI_API_URL=http://localhost:8000
streamlit run web_app/app.py --server.port 8501
```

Then open the Streamlit frontend in your browser at:

```text
http://localhost:8501
```

### Optional: Run feature pipeline manually

```bash
python src/feature_pipeline.py
```

This command updates the production feature group with the latest complete hourly feature representation for Karachi.

## API Endpoints Summary

The backend service in `web_app/backend_api.py` exposes the following endpoints:

### `GET /`

Returns a simple health/status message confirming that the AQI prediction backend is active.

### `GET /predict`

Returns a live predictive payload containing:

- `current_aqi`
- `day1`, `day2`, `day3` forecast values
- model version metadata
- AQI risk category information
- `alert` object with hazard status and message when thresholds are exceeded

The route reuses cached responses for efficiency while ensuring the underlying feature row and inference remain fresh enough for production usage.

### `GET /explain`

Returns SHAP-based explanations for the model output, typically with a structure such as:

- `day1`, `day2`, and `day3` explanation blocks
- `prediction` value
- `base_value`
- ranked `features` array with `feature`, `shap_value`, and `impact`

This helps interpret which environmental and lag variables are contributing most strongly to the forecast.

## Author & Acknowledgments

### Author

- Neha Kumari
- 10Pearls Shine Cohort 9 Internship
- AQI Forecasting and Monitoring Project

### Acknowledgments

This project builds on modern MLOps practices and cloud-native feature engineering workflows. It integrates:

- Hopsworks for feature storage and model registry support
- Open-Meteo for timely environmental data access
- FastAPI and Streamlit for practical operational deployment
- Python data-science tooling for model training, feature creation, and explainability

The repository is structured to support experimentation, reproducibility, and deployment of machine learning forecasting systems in a real-world monitoring context.

---

For deeper implementation details, refer to the source modules under `src/` and the project report in `Report.md`.


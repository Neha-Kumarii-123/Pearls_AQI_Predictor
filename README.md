
# Karachi AQI Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
[![Live App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://pearlsaqipredictor-g5cgabkya5ykm6yx4zm5jp.streamlit.app/)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Deployment-success)

## Overview / Abstract
> 🚀 **Live App Link:** [View Live Streamlit Dashboard](https://pearlsaqipredictor-g5cgabkya5ykm6yx4zm5jp.streamlit.app/)
This repository implements an end-to-end Air Quality Index (AQI) forecasting and monitoring system for Karachi, designed to predict near-term air quality conditions across three horizons: Day +1, Day +2, and Day +3. The system combines domain-aware feature engineering, a centralized Hopsworks Feature Store, trained machine learning models, and a unified Streamlit monitoring dashboard for operational visibility.

The core product is a production-oriented MLOps workflow. Feature generation is centralized in the canonical production pipeline, validated against a 100-feature contract, and pushed to the Hopsworks feature group `karachi_aqi_features` version 6. The unified Streamlit application reads the latest available production feature row, loads the corresponding model versions from the Hopsworks Model Registry, executes live multi-horizon forecasting, performs dynamic SHAP explainability calculations, and applies hazard classification logic for high-risk conditions.

This streamlined, decoupled-to-unified architecture makes the project suitable for seamless, free-tier cloud deployment on Streamlit Cloud without requiring external backend servers or paid infrastructure.

## System Architecture & Workflow

The system follows a streamlined, modular architecture that integrates feature storage, model inference, SHAP explainability, and the front-end dashboard directly into a unified Streamlit application.

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
    web_app/app.py (Unified Streamlit Dashboard & Inference Engine)
            │
            │  - reads latest row from feature store (online=True)
            │  - validates model feature contract & runs Day 1/2/3 models
            │  - computes dynamic SHAP explanations
            │  - live forecast cards, alerting, and EDA visual analytics
            ▼
        End-user monitoring interface (Streamlit Cloud)

```

### Data Flow

1. The pipeline in `src/feature_pipeline.py` pulls recent environmental data from Open-Meteo and ingests the latest hourly observations into the production feature group.
2. Feature generation is defined in `src/feature_engineering.py` and enforces the canonical contract used across training and inference. This contract includes the 100 engineered features, including lag features, rolling statistics, humidex, temporal indicators, and interaction terms.
3. Model training scripts produce horizon-specific models for Day +1, Day +2, and Day +3. These are registered through `src/register_models.py` into the Hopsworks Model Registry.
4. The unified Streamlit application (`web_app/app.py`) retrieves the newest feature row from Hopsworks, validates schema compatibility, and executes all three models to produce a multi-horizon AQI forecast.
5. The dashboard visualizes the predictions, computes SHAP feature contributions dynamically, and surfaces alert conditions for hazardous AQI levels.

## Key Features

* Automated feature engineering pipeline for hourly AQI and meteorological data.
* Production-ready feature store integration with Hopsworks, versioned feature groups, and model registration.
* Three-horizon forecasting through separate Day +1, Day +2, and Day +3 models.
* Canonical 100-feature contract to keep training and serving strictly aligned.
* Hazard threshold logic that flags severe AQI conditions when observed or forecast AQI exceeds critical ranges.
* Real-time in-app inference execution optimized for cloud deployment.
* Dynamic SHAP-based explainability for model-driven factor analysis.
* Interactive EDA dashboard with historical AQI trend, hour-of-day variation, weekday effects, and pollutant relationship views.
* Reproducible training and validation workflow for model version comparison and deployment.

## Tech Stack

### Core Programming & ML

* Python 3.10+
* Pandas and NumPy for feature generation and numerical analysis
* Scikit-learn for regression and pipeline baselines
* XGBoost for the Day +1 forecasting model
* Joblib for model persistence and serialization
* SHAP for explainability and feature attribution

### Feature Store & MLOps

* Hopsworks Feature Store
* Hopsworks Model Registry
* Open-Meteo API data integration
* Feature versioning aligned to `karachi_aqi_features` v6

### Frontend & Deployment

* Streamlit for the interactive unified AQI dashboard and in-app inference engine
* Requests for external data API calls
* dotenv for environment-driven configuration

### Data Engineering & Utilities

* Python standard library and structured logging
* Temporary certificate handling for cloud authentication
* Production validation logic for time-series continuity and schema integrity

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

* Python 3.10 or above
* Pip package manager
* Access to a valid Hopsworks account and API key
* Internet access for Open-Meteo and Hopsworks connectivity

### Environment Variables

Create a `.env` file in the repository root with the required credentials:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key

```

The project expects valid Hopsworks host configuration and model artifacts to already exist in the configured environment. The inference logic reads the current feature group and registered model versions directly from Hopsworks using online feature reading capabilities.

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

Create a `.env` file and set your Hopsworks API key.

### 5. Launch the Streamlit application

Run the unified monitoring app directly:

```bash
streamlit run web_app/app.py --server.port 8501

```

Then open the Streamlit interface in your browser at:

```text
http://localhost:8501

```

### Optional: Run feature pipeline manually

```bash
python src/feature_pipeline.py

```

This command updates the production feature group with the latest complete hourly feature representation for Karachi.

## Author & Acknowledgments

### Author

* Neha Kumari
* 10Pearls Shine Cohort 9 Internship
* AQI Forecasting and Monitoring Project

### Acknowledgments

This project builds on modern MLOps practices and cloud-native feature engineering workflows. It integrates:

* Hopsworks for feature storage and model registry support
* Open-Meteo for timely environmental data access
* Streamlit for practical, self-contained operational deployment
* Python data-science tooling for model training, feature creation, and explainability

The repository is structured to support experimentation, reproducibility, and deployment of machine learning forecasting systems in a real-world monitoring context.

---

For deeper implementation details, refer to the source modules under `src/` and the project report in `Report.md`.




# Pearls AQI Predictor: Serverless MLOps Pipeline

Author: Neha Kumari  
Project: End-to-End AQI Forecasting System  
Program: 10Pearls Shine Internship Capstone  

## System Architecture

```text
[ AQICN API / Open-Meteo ]
          │ (Hourly Telemetry)
          ▼
┌──────────────────────────────────────────────┐
│ Feature Pipeline (src/feature_pipeline.py)   │
│ - UTC Temporal Extraction                    │
│ - Canadian Humidex Domain Calculation        │
│ - AQI Rate-of-Change Computation             │
└──────────────────────┬───────────────────────┘
                       │ Streaming Ingestion (Kafka / Delta Lake)
                       ▼
         ┌────────────────────────────┐
         │ Hopsworks Feature Store    │
         │ (karachi_aqi_features v2)  │
         └─────────────┬──────────────┘
                       │ Feature Views
      ┌────────────────┴────────────────┐
      ▼                                 ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│ Training Pipeline         │     │ Batch Inference & Dashboard │
│ - Model Fine-Tuning       │     │ - Real-time AQI Forecast  │
│ - XAI (SHAP Explanations) │     │ - Hazard Alert Trigger    │
│ - Model Registry Log      │     │ - Streamlit Dashboard     │
└───────────────────────────┘     └───────────────────────────┘

```

## Project Overview

This repository implements a serverless Machine Learning Operations (MLOps) system designed to predict Air Quality Index (AQI) levels for Karachi up to 3 days in advance.

Rather than relying on static notebook scripts or manual data downloads, the system uses an automated pipeline structure that decouples feature engineering, model training, and batch inference through a central feature store.

## Technical Stack

* Language and Runtime: Python 3.10+ in containerized GitHub Codespaces (Linux environment)
* Feature Store and Model Registry: Hopsworks Cloud (Delta Lake, confluent-kafka)
* Data Engineering and ML: Pandas, NumPy, Scikit-Learn, Joblib, SHAP / LIME
* Orchestration and CI/CD: GitHub Actions (Scheduled Cron Workflows)
* Serving Layer: Streamlit Cloud

## Key Engineering Decisions

* Cloud Container Runtime: Developed within GitHub Codespaces to resolve platform-specific C-library and SSL certificate path discrepancies encountered on Windows OS during native SDK streaming.
* UTC Temporal Alignment: Features use UTC Unix timestamps to maintain index consistency across distributed server runners and cloud storage backends.
* Domain Feature Engineering: Integrated the Canadian Humidex formula alongside raw PM2.5 and PM10 metrics to capture atmospheric density and perceived heat impacts on pollutant dispersion.
* Serverless Storage: Replaced raw CSV file storage with Hopsworks Cloud Feature Store to support low-latency streaming writes and consistent offline/online feature parity.

## Current Pipeline Status

* [x] Phase 1: Feature Pipeline and Infrastructure
* Configured containerized cloud environment.
* Built real-time API ingestion script for Karachi weather telemetry.
* Calculated Canadian Humidex and time-based features.
* Successfully registered and streamed live feature vectors into Hopsworks (`karachi_aqi_features` v2).


* [ ] Phase 2: Historical Backfill and Exploratory Data Analysis
* Historical data ingestion and AQI change-rate feature derivation.
* Exploratory data analysis and correlation matrix generation.


* [ ] Phase 3: Model Training and Explainable AI (XAI)
* Model training, evaluation metrics logging, and SHAP model explainability.


* [ ] Phase 4: Automation and Dashboard Deployment
* Automated hourly execution via GitHub Actions and deployment of interactive Streamlit UI.



## Quickstart Guide

1. Repository Setup:

```bash
git clone [https://github.com/Neha-Kumarii-123/Pearls_AQI_Predictor.git](https://github.com/Neha-Kumarii-123/Pearls_AQI_Predictor.git)
cd Pearls_AQI_Predictor

```

2. Environment Configuration:
Create a `.env` file in the root directory:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
AQICN_API_KEY=your_aqicn_api_key

```

3. Run Live Feature Pipeline:

```bash
pip install -r requirements.txt
python src/feature_pipeline.py

```

## Detailed Documentation

For architectural trade-offs, step-by-step pipeline logs, EDA charts, and model evaluation metrics, see the comprehensive project report in [REPORT.md].


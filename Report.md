# Engineering Report: Serverless Air Quality Forecasting System

Author: Neha Kumari  
Role: Lead Engineer & MLOps Developer  
Project: Pearls AQI Predictor (Karachi Air Quality System)  
Context: 10Pearls Shine Internship Capstone  

---

## 1. Executive Summary

This report documents the architectural setup, feature engineering methodology, and production fixes for the Pearls AQI Predictor system. The goal of this project is to shift air quality monitoring in Karachi from reactive reporting to proactive 3-day forecasting. 

Rather than deploying a static machine learning script running on local hardware, this system implements a serverless MLOps architecture. Real-time atmospheric telemetry is extracted from external APIs, transformed into domain-specific features, and ingested into a cloud feature store to ensure zero offline/online data skew during model training and inference.

---

## 2. Infrastructure & Environment Isolation

### Cross-Platform Driver & SSL Resolution
During early development on local Windows environments, native C-library dependencies (confluent-kafka, hops-deltalake, and OpenSSL drivers) triggered runtime path mismatches while establishing secure WebSocket and Kafka streaming connections to Hopsworks Cloud.

To eliminate platform-specific discrepancies, the entire development workspace was migrated to a containerized Linux runtime within GitHub Codespaces. This ensured 100% environment parity with production CI/CD execution nodes running on cloud Linux runners.

### Security & Secret Management
To prevent accidental credential leaks in public version control, all authentication assets (Hopsworks API keys, AQICN API credentials) were decoupled from codebase files and injected dynamically via .env files locally and encrypted environment secrets in GitHub repository settings.

---

## 3. Feature Pipeline Implementation (src/feature_pipeline.py)

The feature pipeline operates as an automated ingestion engine responsible for fetching live atmospheric telemetry, executing feature engineering, and streaming structured vectors into the feature store.

```text
[ AQICN REST API ]
       │
       ├── Telemetry (PM2.5, PM10, Temperature, Humidity)
       ▼
┌────────────────────────────────────────────────────────┐
│ Feature Pipeline Node                                  │
│ 1. Time Normalization (UTC Unix Epoch)                 │
│ 2. Domain Feature: Canadian Humidex Algorithm         │
│ 3. Derived Rate-of-Change Feature (Delta PM2.5)        │
└──────────────────────────┬─────────────────────────────┘
                           │ Low-Latency Streaming Ingestion
                           ▼
          ┌──────────────────────────────────┐
          │ Hopsworks Cloud Feature Store    │
          │ Feature Group: karachi_aqi_v2    │
          └──────────────────────────────────┘

```

### Raw Telemetry Ingestion

The ingestion module connects to the AQICN REST API for the Karachi station, pulling raw telemetry parameters:

* pm25: Fine particulate matter (µg/m³)
* pm10: Coarse particulate matter (µg/m³)
* temperature: Ambient temperature in Celsius
* humidity: Relative atmospheric humidity (%)

### Feature Engineering & Domain Metrics

#### UTC Timestamp Normalization

To prevent server-side timezone mismatches between local testing nodes and cloud runners (GitHub Actions / AWS), all time dimensions are normalized to UTC Unix timestamps. Temporal signals (hour, day, month, day_of_week) are derived directly from the UTC index.

#### Canadian Humidex Model

Raw temperature and humidity alone do not fully account for atmospheric density changes that affect particulate suspension. The pipeline computes the Canadian Humidex to capture perceived heat and vapor pressure:

Humidex = T + (5/9) * (e - 10)

Where e (vapor pressure in mbar) is calculated as:
e = 6.11 * 10^((7.5 * T) / (237.7 + T)) * (H / 100)

* T = Air Temperature (°C)
* H = Relative Humidity (%)

### Feature Store Schema (karachi_aqi_features v2)

Features are registered and stored in the Hopsworks Cloud Feature Store under Feature Group Version 2:

| Feature Name | Data Type | Description |
| --- | --- | --- |
| city | String | Target location identifier (karachi) |
| timestamp | BigInt | Primary key index (UTC Unix timestamp) |
| pm25 | Double | Primary pollutant metric |
| pm10 | Double | Coarse particle metric |
| temperature | Double | Raw ambient temperature |
| humidity | Double | Raw relative humidity |
| humidex | Double | Derived atmospheric density heat metric |
| hour | Int | Hourly temporal signal (0–23) |
| day | Int | Day of month |
| month | Int | Month index (1–12) |
| day_of_week | Int | Day index (0–6) |

---

## 4. Evaluation Criteria Mapping

This project addresses the core requirements set by the industrial evaluation panel:

1. External API Ingestion: Successfully integrated real-time REST API connections pulling weather and pollutant vectors for Karachi.
2. Feature Computation: Engineered temporal signals (hour, day, month) and domain features (humidex). The rate-of-change (Delta AQI) feature is designed for integration into the historical backfill phase.
3. Feature Store Integration: Successfully registered schema and streamed online data into Hopsworks Cloud via Kafka/Delta Lake drivers.

---

## 5. Next Execution Steps

With Phase 1 (Feature Pipeline & Infrastructure Setup) complete and verified on the online cloud dashboard, the immediate development milestones are:

1. Phase 2: Historical Backfill Script (src/backfill_pipeline.py)
* Fetch past 6 months to 2 years of historical Karachi atmospheric telemetry.
* Compute rolling vector differences (aqi_change_rate) across historical sequences.
* Execute Exploratory Data Analysis (EDA) to produce feature correlation matrices and trend graphs.


2. Phase 3: Supervised Model Training & Model Registry
* Train baseline and sequential ML models (Random Forest, LightGBM, LSTM/XGBoost).
* Register optimal model artifacts in the Hopsworks Model Registry.
* Generate SHAP / LIME feature importance plots for model explainability.


3. Phase 4: Serving & Automation
* Deploy real-time interactive UI via Streamlit.
* Configure automated hourly cron execution using GitHub Actions.



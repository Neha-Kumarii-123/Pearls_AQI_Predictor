# Pearls AQI Predictor 📡

Hi! I am Neha Kumari, a final-year software engineering student. This is my very first AI and MLOps project, which I am building as part of the 10Pearls Shine Internship Program. 

Instead of just running a simple data science script on my computer, my goal here is to build a real, working system that automatically updates itself without needing a server (100% serverless MLOps).

---

## 🎯 The Real-World Problem (Why I'm Building This)
Living in Pakistan, we see how drastically air pollution impacts health and daily routines. Most apps only tell us how bad the air *currently* is. That is reactive. 

I want to build a system that looks 3 days ahead. If we can predict a dangerous spike in the Air Quality Index (AQI) beforehand, schools, families, and companies can make smart choices—like switching to remote work or staying indoors—before the smog hits.

---

## 🏗️ How I Am Building This (Step-by-Step Roadmap)
I am breaking this big project down into smaller, logical steps:

1. **Stage 0: Environment & API Connection (Completed) ✅**
   * Set up a clean local Python virtual environment (`venv`).
   * Secured my private API keys inside a hidden `.env` file so they never leak online.
   * Wrote a script to successfully connect to the AQICN API and verified that I can fetch live data.

2. **Phase 1 & 2: Feature Engineering & Historical Data (Next Step) ⏳**
   * I will turn raw data into clean tables using Pandas.
   * Create custom indicators like "how fast the AQI is changing" and gather historical data to train the model.

3. **Phase 3 & 4: Machine Learning & Automation ⏳**
   * Train supervised ML models (like Random Forest) to forecast the index.
   * Build a beautiful, live dashboard using Streamlit.
   * Automate everything using GitHub Actions so the script runs on its own every hour.

---

## 💻 My Stage 0 Verification Log
To prove my setup works perfectly, here is the exact live output I generated directly from my local VS Code terminal when connecting to the API:

```text
📡 Requesting real-time air quality data for Karachi...
=========================================
✅ Connection Successful via GUI!
📍 Location: Karachi US Consulate, Pakistan
😷 Current AQI: 161
📊 Primary Pollutant: pm25
=========================================
🛠️ My Tech Stack
Language: Python

Environment: VS Code, Git, Virtual Environments (venv), Dotenv (.env)

Libraries I will use: Requests (for data fetching), Pandas, Scikit-learn, Streamlit.


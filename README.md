# Nanjing PM2.5 Tomorrow Forecast

This folder contains a standalone Streamlit app for predicting the next-day PM2.5 level in Nanjing.

## Purpose

The original research project explains historical PM2.5 variation. This app turns the work into a usable forecast tool. Its primary website forecast is the next-day PM2.5 value from Open-Meteo's Air Quality forecast endpoint, aggregated from hourly values into a daily mean. The research models are retained for context, uncertainty, and cross-checking.

## What It Shows

- predicted PM2.5 for the next valid forecast date
- readable category label such as Good, Moderate, or Unhealthy for Sensitive Groups
- uncertainty range based on historical holdout error
- recent PM2.5 trend and forecast marker
- cross-check between the main XGBoost model and the conservative one-day-ahead model
- key input variables used by the model

## Forecast Logic

The primary forecast uses the Open-Meteo Air Quality PM2.5 forecast for the target date. This makes the web app operational: it can forecast tomorrow even when the research model's lag features are incomplete.

The app also runs or displays research-model context where available:

- Research model: XGBoost, historical holdout MAE 10.82
- Conservative model: Forecast Random Forest, historical holdout MAE 14.13

The displayed uncertainty range uses the more conservative MAE so the app does not overstate precision.

For live PM2.5 lags, the app first uses Open-Meteo's relative `past_days` / `forecast_days` request format rather than a fixed historical date range. This is important because fixed date-range requests can return stale modeled air-quality archives, while the relative request is better suited for current short-term use.

If the Open-Meteo weather forecast endpoint is temporarily rate-limited, the app can still produce a degraded live forecast by combining live recent PM2.5 data with same-season weather averages from the packaged Nanjing training dataset. The interface labels this clearly as a fallback forecast.

If the Open-Meteo air-quality forecast endpoint itself is temporarily unavailable or rate-limited, the app still returns tomorrow's date using a lower-confidence same-season PM2.5 baseline from the packaged Nanjing dataset. This fallback is clearly labeled and should be interpreted as a continuity mode, not as the best available live forecast.

## How to Run

From the main project folder:

1. Activate the virtual environment: .venv\Scripts\Activate.ps1
2. Run the app: streamlit run tomorrow_pm25_forecaster\app.py
3. Open the local Streamlit URL shown in the terminal.

To validate the packaged app without relying on live network access:

Run: python tomorrow_pm25_forecaster\validate_forecaster.py

## Folder Contents

- app.py: Streamlit dashboard
- forecast_engine.py: data fetching, feature construction, model loading, prediction logic
- validate_forecaster.py: offline validation script
- SOURCES.md: source links for API and calendar references
- models/: copied trained model artifacts
- data/: copied model metrics and modeling dataset for fallback validation
- requirements.txt: Python package requirements

## Data Sources

- Open-Meteo Air Quality API for recent PM2.5 data
- Open-Meteo Weather Forecast API for target-day weather features
- Packaged historical modeling data for offline fallback and validation
- Embedded China public holiday ranges for calendar proxy features

## Important Limitations

This is a data science forecast tool, not an official air-quality advisory. The app depends on modeled Open-Meteo air-quality inputs and weather forecasts. It does not directly observe local traffic, industrial emissions, construction activity, regional transport, or sudden policy changes. Large pollution spikes may therefore be underpredicted.

The app uses the same calendar-proxy logic as the research model for consistency. It marks official public holidays, weekends, nearby holiday days, school-term months, and cold-season months as model features.

## Recommended Use

Use this app as the software extension of the research project. It demonstrates that the project is not only a static analysis, but also a working prediction system with model loading, live data collection, feature engineering, uncertainty communication, and a deployable interface.

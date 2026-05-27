# Deployment Guide

There are two different meanings of "not using PowerShell."

## Option 1: Double-click Local Launcher

Use this if the app only needs to run on this computer.

Double-click start_forecaster.bat.

It will open a command window and run the Streamlit app at:

http://localhost:8502

This avoids typing PowerShell commands, but the app is still running on your own computer. If the command window is closed, the website stops.

## Option 2: Real Public Website

Use this if you want a link that works on other computers without your laptop running.

Recommended platform: Streamlit Community Cloud.

Steps:

1. Create a GitHub repository, for example nanjing-pm25-forecast.
2. Upload the contents of this folder as the repository root. That means app.py, forecast_engine.py, requirements.txt, models, data, and .streamlit should be directly inside the GitHub repository.
3. Go to https://share.streamlit.io.
4. Click Create app.
5. Choose the GitHub repository and branch.
6. Set the app entrypoint file to app.py.
7. Keep Python version as the default unless package installation fails.
8. Click Deploy.

After deployment, Streamlit gives a public URL ending in streamlit.app. That link can be opened from another computer.

## If Uploading the Whole Parent Project Instead

If you upload the full nanjing-air-quality-ds project instead of this folder alone, set the entrypoint file to:

tomorrow_pm25_forecaster/app.py

Streamlit Cloud can read requirements.txt from the same folder as the entrypoint file.

## Files Required for Cloud Deployment

- app.py
- forecast_engine.py
- requirements.txt
- models/best_model.joblib
- models/forecast_model.joblib
- models/feature_columns.json
- models/forecast_feature_columns.json
- data/modeling_dataset.csv
- data/model_metrics.csv
- data/forecast_model_metrics.csv
- .streamlit/config.toml

Do not upload .venv. It is local-only and too large.

## Notes

No API key is required. The app uses public Open-Meteo endpoints.

The first deployment may take a few minutes because packages such as xgboost and lightgbm need to install.

If the cloud app can access recent Open-Meteo PM2.5 data but the weather forecast endpoint is rate-limited, it will produce a degraded live forecast using same-season historical weather averages and label this clearly in the interface.

If the cloud app cannot access recent Open-Meteo PM2.5 data at all, live tomorrow prediction will not be available until the network request succeeds. Turn off live mode only to view the packaged historical validation demo.

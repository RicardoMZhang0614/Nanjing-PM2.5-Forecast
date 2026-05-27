@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment was not found.
  echo Expected: .venv\Scripts\python.exe
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run "tomorrow_pm25_forecaster\app.py" --server.port 8502
pause

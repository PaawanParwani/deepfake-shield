@echo off
title Deepfake Shield - Local PC Server
echo ========================================================
echo   Deepfake Shield - Local PC Host (Full SOTA Models)
echo ========================================================
echo.
cd /d "%~dp0"
set PYTHONPATH=%CD%
echo [INFO] Starting FastAPI + Uvicorn server on http://localhost:8000 ...
echo [INFO] Accessible locally at: http://localhost:8000
echo [INFO] Accessible on your local Wi-Fi network at: http://192.168.0.176:8000
echo.
start "" "http://localhost:8000"
python -m uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000 --reload
pause

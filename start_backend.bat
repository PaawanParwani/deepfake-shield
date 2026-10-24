@echo off
title Deepfake Shield Backend (High Priority, EcoQoS Disabled)
cd /d "%~dp0"
echo ========================================================
echo   Deepfake Shield — Real-Time Inference Backend
echo   Config: Priority=AboveNormal, EcoQoS=Disabled
echo ========================================================
start /AboveNormal "" .venv\Scripts\python.exe -m uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000

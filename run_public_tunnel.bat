@echo off
title Deepfake Shield - Public HTTPS Tunnel
echo ========================================================
echo   Deepfake Shield - Public HTTPS Tunnel (localtunnel)
echo ========================================================
echo.
echo [INFO] Exposing http://localhost:8000 to the public internet...
echo [INFO] Share the generated https://... URL with anyone!
echo.
npx -y localtunnel --port 8000
pause

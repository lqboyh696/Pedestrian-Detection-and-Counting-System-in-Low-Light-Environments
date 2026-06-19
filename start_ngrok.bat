@echo off
title Low-Light Pedestrian Detection - ngrok
cd /d "d:\pythonProject\Pedestrian Detection and Counting System in Low-Light Environments"

echo [1/2] Starting Flask on port 5000...
start "Flask Server" cmd /k "conda activate DarkIR_RealTime && python app.py"

echo        Waiting for models to load...
timeout /t 12 /nobreak >nul

echo [2/2] Starting ngrok tunnel...
echo.
echo   Public:  https://recluse-false-swaddling.ngrok-free.dev
echo   Local:   http://localhost:5000
echo.
echo   Ctrl+C to stop.
echo.

D:\Ngrok\ngrok.exe http --domain=recluse-false-swaddling.ngrok-free.dev 5000
pause

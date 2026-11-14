@echo off
title Phishing Shield AI - Cyber Threat Detector
cls
echo ======================================================================
echo           PHISHING SHIELD AI - CYBER DEFENSE DASHBOARD
echo ======================================================================
echo.
echo [*] Checking Python virtual environment...
if not exist "phishing_env\Scripts\python.exe" (
    echo [!] Virtual environment not found at phishing_env\Scripts\python.exe
    pause
    exit /b 1
)

echo [*] Starting Phishing Detection Backend & Dashboard Server...
echo [*] Hardware Acceleration: CUDA / RTX GPU (if available)
echo [*] Preloading DistilBERT Transformer + URL DNN Models...
echo.

set MPLCONFIGDIR=%TEMP%\mpl
start "Phishing Shield AI - Server" phishing_env\Scripts\python.exe app.py

echo [*] Waiting for server to initialize on http://127.0.0.1:5000...
timeout /t 6 /nobreak >nul

echo [*] Launching Dashboard in default web browser...
start http://127.0.0.1:5000

echo.
echo ======================================================================
echo [OK] Dashboard is running at: http://127.0.0.1:5000
echo.
echo You can keep this window open or minimize it.
echo To stop the server at any time, run stop_app.bat or close the server window.
echo ======================================================================
echo.
pause


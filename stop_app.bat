@echo off
title Stop Phishing Shield AI Server
echo Stopping all running instances of app.py on port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a
    echo Terminated process PID %%a
)
echo Server stopped successfully.
pause


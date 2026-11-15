@echo off
title Push Phishing Shield AI to GitHub
cls
echo ======================================================================
echo                PUSHING PROJECT TO GITHUB REPOSITORY
echo ======================================================================
echo.

set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set NO_PROXY=*

echo [*] Target Remote: https://github.com/vibhuAIML/Phishing-Shield-AI.git
echo [*] Pushing branch: main
echo.
echo If a GitHub browser login window pops up, click "Sign in with your browser".
echo.

git -c http.proxy="" -c https.proxy="" push -u origin main

echo.
echo ======================================================================
if %ERRORLEVEL% EQU 0 (
    echo [OK] SUCCESS! All files have been pushed to GitHub.
    echo Visit: https://github.com/vibhuAIML/Phishing-Shield-AI
) else (
    echo [!] Push encountered an issue. See message above.
)
echo ======================================================================
pause


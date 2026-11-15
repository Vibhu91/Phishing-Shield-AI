@echo off
title Docker & Kubernetes One-Click Automation
cls
echo ======================================================================
echo          PHISHING SHIELD AI - DOCKER & KUBERNETES AUTOMATION
echo ======================================================================
echo.

echo [*] Checking Docker daemon status...
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Docker Desktop is not running yet!
    echo     Please launch Docker Desktop and wait until the whale icon is solid green.
    echo.
    pause
    exit /b 1
)

echo [✓] Docker engine is ACTIVE and ready.
echo.
echo ======================================================================
echo Choose an action:
echo   1. Build Docker image (phishing-shield:latest)
echo   2. Run directly with Docker (http://localhost:5000)
echo   3. Deploy to local Kubernetes (kubectl apply)
echo   4. View Kubernetes status (Pods, Services)
echo   5. Stop all Docker / Kubernetes workloads
echo ======================================================================
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo [*] Building Docker image...
    docker build -t phishing-shield:latest .
    echo.
    echo [✓] Docker image build complete!
    pause
) else if "%choice%"=="2" (
    echo.
    echo [*] Starting Docker container on port 5000...
    docker stop phishing-shield-app >nul 2>&1
    docker rm phishing-shield-app >nul 2>&1
    docker run -d --name phishing-shield-app -p 5000:5000 phishing-shield:latest
    echo [✓] Container running in background!
    echo [✓] Opening http://localhost:5000 in your browser...
    start http://localhost:5000
    pause
) else if "%choice%"=="3" (
    echo.
    echo [*] Applying Kubernetes manifest to cluster...
    kubectl apply -f k8s/local-k8s.yaml
    echo.
    echo [*] Checking pods status (waiting for container initialization)...
    kubectl get pods -l app=phishing-shield
    echo.
    echo [✓] Manifest applied! Open http://localhost:5000 once pods are Running.
    pause
) else if "%choice%"=="4" (
    echo.
    echo [*] Kubernetes Pods:
    kubectl get pods -l app=phishing-shield -o wide
    echo.
    echo [*] Kubernetes Services:
    kubectl get svc phishing-shield-service
    pause
) else if "%choice%"=="5" (
    echo.
    echo [*] Stopping Docker container...
    docker stop phishing-shield-app >nul 2>&1
    docker rm phishing-shield-app >nul 2>&1
    echo [*] Deleting Kubernetes deployment...
    kubectl delete -f k8s/local-k8s.yaml >nul 2>&1
    echo [✓] All clean!
    pause
) else (
    echo Invalid choice.
)

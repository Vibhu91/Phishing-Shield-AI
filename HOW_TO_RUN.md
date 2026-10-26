# 🚀 How to Run the Multimodal Phishing Detection System

This guide provides step-by-step instructions to run the **Cybersecurity Operations Center (SOC) Web Dashboard**, execute **CLI predictions**, run **benchmark evaluations**, and replicate the **data preprocessing and model training pipeline**.

---

## 📋 Table of Contents
1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Quick Start: Launching the SOC Web Dashboard](#2-quick-start-launching-the-soc-web-dashboard)
3. [Running Dual-Engine Predictions via CLI](#3-running-dual-engine-predictions-via-cli)
4. [Running Benchmark Evaluations](#4-running-benchmark-evaluations)
5. [Using the REST API](#5-using-the-rest-api)
6. [Data Preprocessing & Model Training Workflow](#6-data-preprocessing--model-training-workflow)

---

## 1. Prerequisites & Environment Setup

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.10+ (Tested on Python 3.10 / 3.11 / 3.12 / 3.14)
- **Hardware Acceleration (Recommended)**: NVIDIA GPU with CUDA (e.g. RTX 3050 Laptop GPU / RTX 3060+) or CPU fallback.

### Activate the Project Virtual Environment

Open PowerShell or Command Prompt inside the `PhishingDeeplearning` directory:

```powershell
# Navigate to project directory
cd C:\Users\Vibhu-Kavi\Downloads\PhishingDeeplearning

# Activate existing virtual environment
.\phishing_env\Scripts\Activate.ps1
```

*(If you are on Linux / macOS, use: `source phishing_env/bin/activate`)*

---

## 2. Quick Start: Launching the SOC Web Dashboard

Start the integrated Flask web server hosting both the REST API and the interactive dark-mode SOC dashboard:

```powershell
& ".\phishing_env\Scripts\python.exe" app.py
```

### Accessing the Web Interface
Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Dashboard Features:
- **Real-Time Dual-Engine Inference**: Type or paste any email subject, body text, or standalone URL.
- **Embedded Link Detection**: Automatically parses hyperlinks inside the email body and feeds them to the 49-feature URL DNN.
- **Visual Threat Breakdown**: Live animated gauge meters, severity tags (`SAFE`, `LOW`, `MODERATE`, `HIGH`, `CRITICAL`), and feature-level explanations.
- **13 Interactive Preloaded Scenarios**: Click any sample (PayPal suspension, M365 password expiry, Amazon order fraud, benign company newsletters) to audition detection capabilities instantly.

---

## 3. Running Dual-Engine Predictions via CLI

Use `merged_phishing_detector.py` for rapid headless scanning without the web server:

### A. Full Email Multimodal Scan (Subject + Body + Embedded Links)
```powershell
& ".\phishing_env\Scripts\python.exe" merged_phishing_detector.py `
    --subject "Urgent: Your account is locked" `
    --body "Please verify your account credentials immediately at http://login-secure-banking-update.xyz to prevent suspension."
```

### B. Pure Conversational / Business Email Check
```powershell
& ".\phishing_env\Scripts\python.exe" merged_phishing_detector.py `
    --subject "Weekly Team Sync Minutes" `
    --body "Hi team, thanks for attending our weekly engineering sync. The meeting notes have been uploaded to github.com for your review."
```

### C. Standalone URL Inspection (URL DNN Only)
```powershell
& ".\phishing_env\Scripts\python.exe" merged_phishing_detector.py `
    --url "http://paypal-verification-account-login.top/auth/index.php"
```

---

## 4. Running Benchmark Evaluations

### A. Internal Dataset Holdout Evaluation
Evaluate both models on a randomized slice from the active 182,364-email corpus:

```powershell
& ".\phishing_env\Scripts\python.exe" merged_phishing_detector.py --evaluate --samples 500
```

### B. Unseen External Dataset Evaluation
Run the model against fresh external datasets (e.g. HuggingFace / Kaggle unseen sets):

```powershell
& ".\phishing_env\Scripts\python.exe" -c "
import sys; sys.path.insert(0, '.'); from merged_phishing_detector import MergedPhishingDetector
import pandas as pd
detector = MergedPhishingDetector()
df = pd.read_csv('https://huggingface.co/datasets/UniqueData/email-spam-classification/resolve/main/email_spam.csv')
for _, row in df.head(5).iterrows():
    print(detector.predict(subject=str(row['title']), body=str(row['text'])))
"
```

---

## 5. Using the REST API

When `app.py` is running, the REST API endpoints are exposed on `http://127.0.0.1:5000`:

### Health Check
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -Method GET
```

### Analyze Email
```powershell
$payload = @{
    subject = "Account Password Expiration Notice"
    body    = "Your session will expire in 1 hour. Log in now at http://portal-update-m365.xyz"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict/email" -Method POST -Body $payload -ContentType "application/json"
```

### Analyze URL
```powershell
$payload = @{
    url = "http://login-account-verification.xyz/auth"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict/url" -Method POST -Body $payload -ContentType "application/json"
```

---

## 6. Data Preprocessing & Model Training Workflow

If you ever wish to re-merge new data, re-run SMOTE, or retrain the Transformer:

### Step 1: Merge Additional Datasets
```powershell
& ".\phishing_env\Scripts\python.exe" emaildata/merge_datasets.py
```
*(Produces `emaildata/combined_phishing_email_dataset.csv` with 182,364 deduplicated rows).*

### Step 2: Exploratory Data Analysis (EDA)
```powershell
& ".\phishing_env\Scripts\python.exe" "emaildata/explore_dataset email.py" `
    --data "emaildata/combined_phishing_email_dataset.csv" `
    --out "emaildata/eda_output"
```

### Step 3: Preprocessing & SMOTE Balancing
```powershell
& ".\phishing_env\Scripts\python.exe" emaildata/preprocessing_smote.py `
    --input "emaildata/combined_phishing_email_dataset.csv" `
    --out "emaildata/output" `
    --max-features 30000 `
    --k 5
```

### Step 4: Retrain the Email DistilBERT Transformer (GPU)
```powershell
& ".\phishing_env\Scripts\python.exe" emaildata/train_phishingemail.py `
    --data "emaildata/combined_phishing_email_dataset.csv" `
    --model distilbert-base-uncased `
    --epochs 2 `
    --bs 16 `
    --accum 2 `
    --grad_ckpt `
    --out "emaildata/results"
```
*(Automatically saves the fine-tuned model and tokenizer to `emaildata/results/best_distilbert/` for instant deployment).*

---

### 🎉 All Systems Ready!
Launch `app.py` anytime to run the complete defense system.

# 🎓 Teacher Presentation Guide: Phishing Shield AI

Follow this quick guide to run and present the project smoothly in front of your teacher.

---

## 🚀 How to Start the Demo (Choose Either Way)

### Method A: One-Click (Easiest)
Double-click:
👉 **`START_PRESENTATION.bat`**

### Method B: Single Command in Terminal
In your PowerShell or Command Prompt, run:
```powershell
.\phishing_env\Scripts\python.exe presentation_demo.py
```

Both methods will:
1. Load your **DistilBERT Transformer** and **URL DNN** onto the GPU.
2. Start the **Flask Backend Server** on `http://127.0.0.1:5000`.
3. **Automatically open the Frontend Dashboard** in your default web browser.

---

## 🎯 3-Step Presentation Script for Your Teacher

When the browser opens, show these 3 key capabilities:

### 1. Show the Dual Architecture (Tell the Teacher)
> *"Most commercial filters evaluate email text and URLs in silos. In this project, we built a multimodal dual-engine pipeline:*
> * 1. **Email Classifier**: A fine-tuned DistilBERT transformer that detects social-engineering cues (urgency, credential lures, fear of account suspension).
> * 2. **URL Classifier**: A 49-feature Deep Neural Network (DNN) that extracts Shannon entropy, token randomness, domain authority, and TLD risks.
> * 3. **Fusion Layer**: A Bayesian Noisy-OR probability fusion model combining both outputs."*

### 2. Live Demo 1: Preloaded Real Scenarios
1. Go to the **"Demo Scenarios"** panel on the dashboard.
2. Click on **"🚨 PayPal Account Suspension"** or **"🚨 Microsoft 365 Password Expiry"**.
3. Click **"Scan"**:
   * Show that the system flags it as **`PHISHING [CRITICAL]` (100%)**.
   * Point out the **Key Risk Signals** detected (e.g., suspicious TLD `.xyz`, credential keywords, high entropy).

### 3. Live Demo 2: Safe/Legitimate Test
1. Click on **"🛡️ Wikipedia Phishing Article"** or type `https://www.google.com`.
2. Click **"Scan URL"**:
   * Show that the system correctly marks it as **`LEGITIMATE [SAFE]` (2.5% risk)**.

### 4. Live Demo 3: Interactive Custom Input
* Ask your teacher: *"Sir/Ma'am, give me any sample phishing email or URL you want to test."*
* Paste it into the box and click **"Run Dual Scan"** to see live inference on your RTX 3050 GPU in ~50 ms.

---

## ⏹️ How to Stop the Demo After Presentation
* In the black terminal window, press **`Ctrl + C`**, or double-click **`stop_app.bat`**.


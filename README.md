# 🛡️ Multimodal Phishing Detection System
### Dual-Engine Deep Learning Defense: Email Transformer (DistilBERT) + URL Deep Neural Network (DNN)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow / Keras 3](https://img.shields.io/badge/TensorFlow%20%2F%20Keras%203-FF6F00.svg?logo=tensorflow&logoColor=white)](https://keras.io/)
[![HuggingFace DistilBERT](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/distilbert-base-uncased)
[![Flask REST API](https://img.shields.io/badge/Flask-Web%20Dashboard%20%26%20API-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning cybersecurity evaluation and detection system designed to analyze and identify email and URL phishing attacks across multiple modalities. The system couples a fine-tuned **DistilBERT NLP Transformer** (for semantic intent and social engineering detection) with an optimized **49-feature Deep Neural Network (DNN)** (for structural, statistical, and lexical URL inspection), fused via **Bayesian Noisy-OR Threat Fusion**.

It includes a full-featured **Cybersecurity Operations Center (SOC) Web Dashboard**, a **REST API**, and an automated pipeline CLI.

---

## 📑 Table of Contents

- [Key Capabilities](#-key-capabilities)
- [System Architecture & Flow](#-system-architecture--flow)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
  - [Overall Multi-Engine Comparison](#overall-multi-engine-comparison)
  - [URL DNN Benchmark Across 5 Datasets](#url-dnn-benchmark-across-5-datasets-n--126511)
  - [Email DistilBERT Performance](#email-distilbert-performance-n--18237-holdout-test-emails)
  - [Multimodal Fusion Results](#multimodal-fusion-results-n--18237-uncontaminated-holdout)
- [Neural Network & Transformer Architectures](#-neural-network--transformer-architectures)
  - [1. URL Lexical & Structural DNN](#1-url-lexical--structural-dnn)
  - [2. Email Intent DistilBERT](#2-email-intent-distilbert)
  - [3. Decision Fusion Engine](#3-decision-fusion-engine)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quick Start](#-installation--quick-start)
  - [1. Clone and Set Up Virtual Environment](#1-clone-and-set-up-virtual-environment)
  - [2. Install Dependencies](#2-install-dependencies)
- [Running the SOC Web Dashboard](#-running-the-soc-web-dashboard)
- [REST API Reference](#-rest-api-reference)
- [CLI & Pipeline Execution (`main.py`)](#-cli--pipeline-execution-mainpy)
- [Hardware & Optimization](#-hardware--optimization)
- [License](#-license)

---

## ⚡ Key Capabilities

- **Multimodal Threat Fusion**: Jointly analyzes email natural language (urgency, credential intimidation, pretexting) and deep lexical URL features to provide structural defense against complex multi-vector phishing attempts.
- **High-Throughput Offline URL Inspection**: Extracts **49 hand-crafted statistical and lexical features** in microseconds without external DNS/WHOIS round-trips, eliminating attacker probe awareness and latency bottlenecks.
- **Quishing & Attachment Scanner**: Decodes 2D QR codes (Quishing) inside image attachments with OpenCV and dissects PDF streams for `/URI` actions and `/JavaScript` execution.
- **Email Header Security & Spoofing Defense**: Audits RFC 5322 MIME headers for **SPF, DKIM, and DMARC** compliance, flagging Display Name brand impersonation (e.g. PayPal, Microsoft).
- **Enterprise Mail Flow Gateways**: Native connectors for **Microsoft 365** (MS Graph API) and **Google Workspace** (Gmail REST API) with automated threat quarantine and category labeling.
- **State-of-the-Art NLP Semantics**: Fine-tuned on **182,364 real-world emails** using HuggingFace DistilBERT with sub-word tokenization and attention mechanisms.
- **Probabilistic Threat Fusion (Bayesian Noisy-OR)**: Combines text intent with the highest-risk embedded link. Under Noisy-OR mathematics ($1 - (1 - P_E)(1 - P_U)$), any URL scoring high risk naturally bounds combined risk without requiring manual heuristic overrides.
- **SOC Analyst Feedback & Continuous Retraining**: SQLite-backed human-in-the-loop triage system (`feedback_store.db`) allowing security analysts to report false positives/negatives and trigger automated retraining.
- **Production Containerization & Task Queue**: Distributed asynchronous worker pipeline with **Celery and Redis**, fully deployable via **Docker Compose**.
- **Interactive SOC-Style Web Application**: Glassmorphism dark-mode UI with live animated risk gauges, structural breakdown charts, 5 inspection tabs, 13 pre-loaded real-world threat scenarios, and actionable remediation advice.
- **Production REST API**: Scalable endpoints (`/api/predict/email`, `/api/predict/url`, `/api/scan/headers`, `/api/scan/attachment`, `/api/gateway/scan`, `/api/feedback`, `/api/retrain/trigger`, `/api/system/status`).

---

## 🏗️ System Architecture & Flow

```mermaid
flowchart TD
    In([User / Security Gateway Input]) --> InputType{Input Type}
    
    %% Email Branch
    InputType -->|Email Subject & Body| ExtractURLs[Regex URL Extractor]
    ExtractURLs --> DistilBERT[DistilBERT Transformer\nTokenization & Sequence Classification]
    DistilBERT -->|Text Phishing Prob: P_email| Fusion[Multimodal Bayesian\nNoisy-OR Fusion Engine]
    
    ExtractURLs -->|Extracted Link 1..N| Extractor[49-Feature URL Extractor\nLexical, Structural & Entropy]
    
    %% URL Direct Branch
    InputType -->|Single URL| Extractor
    
    %% Feature Processing & DNN
    Extractor --> Scaler[StandardScaler Normalization]
    Scaler --> DNN[URL Deep Neural Network\nDense 128 -> 64 -> 32 -> Sigmoid]
    DNN -->|URL Phishing Prob: P_url| Fusion
    
    %% Decision Logic
    Fusion --> ProbFormula["P_combined = 1 - (1 - P_email) * (1 - max(P_url))"]
    ProbFormula --> Verdict[Unified Verdict & SOC Risk Level]
    
    Verdict --> Output[Web Dashboard & REST API Output\n- Risk Score (0-100%)\n- Severity: Safe / Suspicious / Phishing\n- Feature Breakdown & Explanations\n- Security Recommendations]
```

---

## 📊 Model Performance & Benchmarks

### Overall Multi-Engine Comparison

| Model / Subsystem | Target Modality | Test Samples ($n$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Email DistilBERT** | Subject & Body NLP | 18,237 | **99.73%** | **99.64%** | **99.81%** | **0.9973** | **0.9999** |
| **URL DNN (49 Features)** | Lexical / Structural (`eTLD+1`) | 124,090 | **96.30%** | **95.63%** | **95.68%** | **0.9565** | **0.9918** |
| **Merged Dual-Engine (Linear)** | Multimodal Linear Blending | 18,237 | **99.63%** | **99.45%** | **99.81%** | **0.9963** | **0.9990** |
| **Merged Dual-Engine (Noisy-OR)**| Bayesian Decision Fusion | 18,237 | **91.84%** | **86.13%** | **99.84%** | **0.9248** | **0.9857** |

> [!NOTE]
> Evaluated strictly on uncontaminated holdout test partitions ($n = 18,237$ emails and $n = 124,090$ `eTLD+1` URLs) with zero training data leakage.

---

### Main URL Benchmark: Domain-Grouped Split (`eTLD+1` via `tldextract`, $n = 124,090$)

To guarantee zero domain memorization leakage, URLs are partitioned strictly by registered domain (`eTLD+1`):
- **Accuracy**: **`96.30%`** (TP: 50,588, TN: 68,905, FP: 2,314, FN: 2,283)
- **Precision**: **`95.63%`**
- **Recall / Sensitivity**: **`95.68%`**
- **Specificity**: **`96.75%`**
- **F1-Score**: **`0.9565`**
- **ROC-AUC**: **`0.9918`** | **MCC**: **`0.9234`**
- **Domain Overlap Across Train/Test**: **`0`** (Evaluated from saved prediction arrays)

#### Cross-Dataset URL Generalization & Tree Baselines (Held-Out Sources):
| Held-Out Test Source | Test Samples ($n$) | URL DNN Acc | Random Forest Acc | XGBoost Acc | LightGBM Acc | URL DNN ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dataset_phishing`** | 11,398 | **75.79%** | 72.04% | 73.50% | 71.27% | 0.8571 |
| **`mendeley_url_phish`** | 40,000 | 34.09% | 39.25% | **50.74%** | 35.38% | 0.9252 |
| **`phishbd_2026`** | 40,000 | 87.67% | 91.09% | 88.75% | **91.62%** | 0.9548 |
| **`phiusiil`** | 40,000 | **95.09%** | 93.47% | 94.41% | 94.38% | 0.9796 |
| **`raw_dataset`** | 40,000 | 87.83% | **88.66%** | 87.57% | 84.20% | 0.9496 |

---

### URL DNN Benchmark Across 5 Datasets ($n = 126,511$ Stratified Split)

Evaluated on **126,511 unseen hold-out test URLs** spanning 5 diverse benchmark distributions:

| Dataset Source | Samples ($n$) | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OVERALL COMPOSITE** | **126,511** | **96.23%** | **95.35%** | **96.18%** | **96.27%** | **0.9576** | **0.9918** | **0.9907** | **0.9237** |
| `raw_dataset` | 29,902 | **97.96%** | 99.37% | 96.54% | 99.38% | **0.9793** | 0.9933 | 0.9954 | 0.9597 |
| `phiusiil` | 46,518 | **97.85%** | 99.97% | 94.95% | 99.98% | **0.9739** | 0.9987 | 0.9988 | 0.9566 |
| `phishbd_2026` | 26,227 | **94.16%** | 93.91% | 98.01% | 85.17% | **0.9592** | 0.9838 | 0.9926 | 0.8592 |
| `mendeley_url_phish` | 21,584 | **94.39%** | 62.06% | 93.59% | 94.47% | **0.7463** | 0.9853 | 0.9224 | 0.7353 |
| `dataset_phishing` | 2,280 | **81.67%** | 78.30% | 87.52% | 75.83% | **0.8266** | 0.9162 | 0.9228 | 0.6378 |

#### URL Confusion Matrix Counts ($n = 126,511$)
- **True Negatives (TN)**: `67,863` (96.27% of legitimate URLs)
- **True Positives (TP)**: `53,879` (96.18% of malicious URLs)
- **False Positives (FP)**: `2,630` (3.73% FPR)
- **False Negatives (FN)**: `2,139` (3.82% FNR)

---

### Email DistilBERT Performance ($n = 18,237$ Holdout Test Emails)

Trained and evaluated on the expanded **182,364-email corpus** (90,775 Legitimate / 91,589 Phishing, balanced 50/50 split):

- **Test Accuracy**: **`99.73%`** (Validation Accuracy: `99.64%`)
- **Precision**: **`99.64%`** (Ultra-low false positive rate: 33 FP out of 9,078 legitimate emails)
- **Recall / Sensitivity**: **`99.81%`** (Intercepts 99.81% of active threats: 9,142 TP, only 17 FN out of 9,159)
- **F1-Score**: **`0.9973`**
- **ROC-AUC**: **`0.9999`** | **PR-AUC**: **`0.9999`**
- **Hardware & Speed**: 9,120 steps trained in 60.2 minutes on **NVIDIA GeForce RTX 3050 Laptop GPU** (FP16 mixed precision)
- **Unseen External Dataset Test ($n = 1,000$)**: **`99.90% Accuracy`** | **`100.00% Precision`** | **`99.77% Recall`** | **`1.0000 ROC-AUC`**

---

### Multimodal Fusion Results ($n = 18,237$ Uncontaminated Holdout)

Evaluated strictly across the uncontaminated 18,237-email holdout test set (9,159 phishing threats, 9,078 legitimate messages, where 3,811 contain active hyperlinks, completely isolated from training data):
- **DistilBERT Alone**: **`99.73%`** Accuracy | **`99.81%`** Recall | **`99.64%`** Precision | **`0.9973`** F1 | **`0.9999`** ROC-AUC
- **Linear Blending Fusion**: **`99.63%`** Accuracy | **`99.81%`** Recall | **`99.45%`** Precision | **`0.9963`** F1 | **`0.9990`** ROC-AUC
- **Bayesian Noisy-OR Fusion**: **`91.84%`** Accuracy | **`99.84%`** Recall | **`86.13%`** Precision | **`0.9248`** F1 | **`0.9857`** ROC-AUC
- **High-Confidence Safeguard Property**: Mathematically, any URL with $P_{\text{URL}} \ge 0.85$ already yields $P_{\text{combined}} \ge 0.85$ under Noisy-OR ($1 - 1 \times (1 - 0.85) = 0.85$), meaning explicit hard-coded overrides are mathematically redundant with Bayesian Noisy-OR. The hypothesis that fusion defends against benign pretexting text coupled with weaponized links remains an architectural hypothesis, as standard static benchmarks lack targeted synthetic evasion attacks.

---


---

### Tier 2: Targeted Evasion Benchmark & Multimodal Defense-in-Depth

While in-distribution email benchmarks show near-ceiling performance from text transformers alone, single-modality detectors fail when confronted with targeted, multi-vector evasion attacks:

| Attack Scenario | Threat Vector Tested | DistilBERT Alone | URL DNN Alone | Bayesian Noisy-OR | Learned Stacker |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Benchmark A: In-Email Links (Leakage-Safe)** | Benign Text + In-Email Link ($N=7,382$, 5 Seeds) | 0.49% Recall | 70.84% Recall | **70.51% Recall** (63.88% Spec) | **72.77% Recall** (62.03% Spec, 67.45% Acc) |
| **Benchmark A: In-Corpus Pilot ($n=1,000$)** | Benign Text + In-Email Phish Link (In-Corpus Domains) | 1.20% Recall | 74.50% Recall | **75.10% Recall** | **75.20% Recall** (75.78% Spec, 76.40% Acc) |
| **Benchmark B: Web-Style Links ($n=1,000$)** | Benign Text + Web Phish Link (`eTLD+1`) | 1.70% Recall | 96.00% Recall | **96.00% Recall** | **96.40% Recall** (96.36% Spec, 95.90% Acc) |

*(Note: Benchmark A evaluates real email links from historical corpora [CEAS, Enron, Nazario]. Under the strict leakage-safe domain-grouped protocol across 3,999 unseen domains [Section 6.5], accuracy is 67.45% ± 1.91% and AUC is 0.7589 ± 0.0236, showing that earlier in-corpus estimates [~75%] were optimistic due to domain overlap. Benchmark B evaluates web links from phiusiil/raw_dataset).*

- **Empirical Diagnosis of URL Specificity Gap**: Retraining the URL DNN without `has_https` and `has_www` demonstrated that the domain gap on vintage email links stems from tracking/newsletter syntax (deep path/query parameters) rather than protocol indicators alone.
- **Domain Adaptation & Controlled Stacker Ablation**: Retraining the URL DNN on email-extracted links lifted validation AUC from near-chance (0.515) to **0.7296**. Under strict 5-seed `GroupShuffleSplit` on $7,382$ samples across $3,999$ domains, GBDT stacking achieved **77.43% ± 3.12% AUC** compared to **75.89% ± 2.36%** for linear stacking; control experiments proved that this small shift (within one standard deviation) is driven entirely by non-linear calibration of specialized vs. base URL scores rather than multimodal text interaction. An audit finding that 92.3% of benchmark domains were present in the training email archive indicates that 0.76 AUC remains an in-corpus optimistic bound for novel external enterprise domains. High-recall operation remains constrained without link-level reputation or context features (74.9%–77.7% FPR at 95% TPR), as commercial domain whitelists match fewer than 0.5% of clean links in historical archives.
- **Reliability & Operating Points**:
  - DistilBERT operates at **`0.011% FPR`** at 95% TPR on in-distribution text (ECE = 0.25%).
  - On Benchmark B web evasions, the Learned Stacker achieves **`3.64% FPR`** at 95% TPR (ECE = 3.27%).
  - On realistic in-email links (Benchmark A), the leakage-safe evaluation achieves **`6.19% ECE`** (with mixed calibration: over-confident between 0.2 and 0.7, under-confident above 0.75; the pilot testbed exhibited 10.52% ECE), while operating at 95% TPR incurs a severe **74.9%–77.7% FPR** unless supported by link-level reputation and contextual features.
  - On the in-corpus pilot testbed A ($n = 1,000$), the Learned Stacker reported an optimistic **76.40% accuracy** and 0.8090 AUC (ECE = 10.11%).
  - Full reliability diagram preserved at [`artifacts/calibration_reliability_diagram.png`](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/artifacts/calibration_reliability_diagram.png).

## 🧠 Neural Network & Transformer Architectures

### 1. URL Lexical & Structural DNN

The URL classifier evaluates **49 static features** grouped into:
- **Lengths & Ratios**: URL length, hostname length, path length, query length, digit ratio, letter ratio, special character ratio, vowel ratio.
- **Delimiter & Character Counts**: Counts of dots, hyphens, underscores, slashes, `@`, `?`, `=`, `&`, `!`, `%`, `$`, `#`, `~`, digits, letters.
- **Structural Flags**: `hostname_is_ip`, `port_present`, `has_https`, `has_www`, `has_at_symbol`, `has_double_slash_redirect`, `has_shortening_service`, `has_suspicious_tld`, `has_punycode`, `http_token_in_path`.
- **Information Theory**: Shannon Entropy of full URL (`url_entropy`) and hostname (`hostname_entropy`), longest token length, average token length.

```text
Input (49 features)
  │
  ├── Dense(128, activation='relu', kernel_initializer='he_normal')
  ├── BatchNormalization()
  ├── Dropout(0.3)
  │
  ├── Dense(64, activation='relu', kernel_initializer='he_normal')
  ├── BatchNormalization()
  ├── Dropout(0.3)
  │
  ├── Dense(32, activation='relu')
  │
  └── Dense(1, activation='sigmoid')  ──> Output Phishing Probability P_url
```

* **Loss Function**: Binary Cross-Entropy
* **Optimizer**: Adam ($\eta = 10^{-3}$)
* **Regularization**: Batch Normalization, Dropout ($p=0.3$), `ReduceLROnPlateau` (factor $0.5$, patience $4$), `EarlyStopping` (patience $8$, restoring best weights on validation AUC).
* **Data Balancing**: Cost-sensitive class weighting in BCE loss (`compute_class_weight`), avoiding synthetic sample distortion while matching SMOTE cross-validation performance (95.64% vs. 95.67% F1).

---

### 2. Email Intent DistilBERT

* **Base Pretrained Model**: `distilbert-base-uncased` (6 transformer layers, 768 hidden dimension, 12 attention heads, 66M parameters).
* **Tokenization**: WordPiece tokenizer, maximum sequence length = 256 tokens.
* **Loss Function**: Cross-Entropy Loss with weighted class compensation.
* **Training Protocol**: Fine-tuned over 145,891 training samples (80% split of 182,364 emails, 9,120 steps across 2 epochs) with AdamW, linear learning rate warmup, and effective batch size 32.

---

### 3. Decision Fusion Engine

1. **Link Extraction**: Parses raw email text using high-coverage regex catching HTTP/HTTPS schemes, bare `www.` domains, and exotic TLDs.
2. **Text Inference**: Runs DistilBERT over `[Subject] + " " + [Body]` $\rightarrow P_{\text{email}}$.
3. **Link Inference**: For each extracted URL $i \in \{1, \dots, n\}$, extracts 49 features, applies saved `StandardScaler`, and executes the DNN $\rightarrow P_{\text{URL}_i}$.
4. **Bayesian Noisy-OR Combination**:
   $$P_{\text{combined}} = 1 - (1 - P_{\text{email}}) \times \prod_{i=1}^n (1 - P_{\text{URL}_i})$$
5. **High-Confidence Payload Safeguard (Explainability Property)**:
   $$\text{Under Noisy-OR, if } \max_{i} P_{\text{URL}_i} \ge 0.85 \implies P_{\text{combined}} \ge 0.85$$
   *(Mathematically redundant with Noisy-OR; maintained as an analyst triage alert tag).*

---

## 📁 Project Directory Structure

```text
PhishingDeeplearning/
│
├── 🌐 app.py                          # Flask Web Server & REST API backend
├── 📜 main.py                         # Unified orchestrator & pipeline CLI
├── 🔀 merged_phishing_detector.py     # Multimodal dual-engine (DistilBERT + URL DNN)
│
├── 🧠 feature_extraction.py           # 49-feature offline URL feature extractor
├── ⚙️ preprocessing.py                # Train/test domain-grouped split, standard scaling & class weights
├── 🏋️ train_dnn.py                    # URL DNN model architecture & training script
├── 📊 generate_results.py             # Evaluator generating multi-metric reports & curves
├── 🔍 explore_dataset.py              # URL dataset row/column/balance inspector
├── 📈 eda.py                          # Exploratory data analysis & feature correlations
├── 🎯 threshold_search_reweighted.py  # Optimal threshold search under varied prevalence
│
├── 🎨 static/                         # Web Dashboard frontend assets
│   ├── index.html                     # Responsive SOC cybersecurity interface
│   ├── css/
│   │   └── style.css                  # Dark-mode glassmorphic theme & animations
│   └── js/
│       └── app.js                     # Real-time UI logic, gauge rendering & REST client
│
├── 📧 emaildata/                      # Email dataset & DistilBERT Transformer subsystem
│   ├── combined_phishing_email_dataset.csv # 182,364 labeled email samples
│   ├── train_phishingemail.py         # PyTorch/HuggingFace DistilBERT training script
│   ├── explore_dataset email.py       # Email dataset exploration
│   └── results/
│       ├── best_distilbert/           # Fine-tuned PyTorch model weights & tokenizer
│       └── distilbert-base-uncased_random.json # Test metrics report
│
├── ⚙️ preprocessed/                   # Serialized models & processed arrays
│   ├── dnn_model.keras                # Trained Keras 3 URL Deep Neural Network
│   ├── scaler.joblib                  # Fitted StandardScaler for live inference
│   ├── train_scaled.csv               # Normalized training split with class weighting
│   └── test_scaled.csv                # Scaled holdout test set (126,511 rows)
│
├── 📊 results/                        # Evaluation charts & summary files
│   ├── model_results_report.md        # Comprehensive evaluation report
│   ├── model_results_comprehensive.csv# Tabular metrics across all 5 benchmarks
│   ├── model_results_summary.txt      # Text summary & confusion matrices
│   ├── confusion_matrices.png         # Overall & per-dataset confusion matrix heatmaps
│   ├── training_history.png           # Loss, accuracy, AUC curves vs. epochs
│   └── roc_pr_curves.png              # ROC & Precision-Recall curves
│
├── 📦 requirements.txt                # Python package dependencies
├── 🚫 .gitignore                      # Git exclusion rules
└── 📜 README.md                       # Complete documentation & user guide
```

---

## 🚀 Installation & Quick Start

### 1. Clone and Set Up Virtual Environment

```powershell
# Open terminal inside the project directory
cd PhishingDeeplearning

# Create or activate Python virtual environment
python -m venv phishing_env
.\phishing_env\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install torch transformers flask flask-cors
```

---

## 🖥️ Running the SOC Web Dashboard

Launch the modern Cybersecurity Operations Center web interface:

```powershell
python app.py
```

Then open your browser at:
👉 **`http://127.0.0.1:5000`**

### Features of the Web Interface:
1. **Interactive Scanner**:
   - **Email Mode**: Paste subject and body. The app highlights embedded links and runs both DistilBERT and URL DNN in parallel.
   - **URL Mode**: Directly inspect any web link against all 49 structural features.
2. **13 Pre-loaded Test Cases**:
   - 🚨 *Threats*: PayPal suspension, Microsoft 365 credential expiry, Amazon unauthorized purchase, IRS tax refund scam, Bank of America wire alert, DocuSign signature harvest, MetaMask Web3 wallet seed scam, Raw IP hosts, `@`-symbol credential injection.
   - 🛡️ *Legitimate*: Python Weekly newsletter, GitHub PR review, Stripe billing notice, corporate roadmap sync, Wikipedia, Python docs.
3. **Visual Analytics**:
   - Animated SVG Risk Gauge with needle transitions.
   - Dual-engine score contribution bar (Email NLP vs. URL DNN).
   - High-risk indicator pills (IP host, suspicious TLD, credential harvesting tokens).
   - SOC Guidance & Incident Response checklist.

---

## 📡 REST API Reference

The Flask backend exposes full JSON endpoints for external integration:

### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "online",
  "model": "Merged Phishing Detector (DistilBERT + URL DNN)"
}
```

---

### 2. Predict Email Phishing Risk
```http
POST /api/predict/email
Content-Type: application/json
```
**Request Body:**
```json
{
  "subject": "Action Required: Microsoft 365 Password Expiry",
  "body": "Please confirm your password immediately at http://m365-portal-verify.xyz/login"
}
```
**Response:**
```json
{
  "verdict": "PHISHING",
  "risk_level": "CRITICAL",
  "combined_phishing_prob": 0.9998,
  "email_prob": 0.9850,
  "urls_detected": 1,
  "urls": [
    {
      "url": "http://m365-portal-verify.xyz/login",
      "phishing_prob": 0.9995,
      "verdict": "PHISHING"
    }
  ],
  "threat_contributors": ["High-risk link payload", "Urgent credential harvesting text"]
}
```

---

### 3. Predict Single URL Risk
```http
POST /api/predict/url
Content-Type: application/json
```
**Request Body:**
```json
{
  "url": "http://secure-paypal-login-verify.xyz/auth/login.php"
}
```
**Response:**
```json
{
  "url": "http://secure-paypal-login-verify.xyz/auth/login.php",
  "verdict": "PHISHING",
  "risk_level": "CRITICAL",
  "combined_phishing_prob": 0.9984,
  "key_features": {
    "url_length": 52,
    "has_suspicious_tld": 1,
    "has_https": 0,
    "suspicious_keyword_count": 3
  }
}
```

---

### 4. Fetch Benchmark Metrics
```http
GET /api/benchmarks
```
Returns accuracy, F1, precision, recall, and sample counts across all three models.

---

### 5. Fetch Sample Test Cases
```http
GET /api/samples
```
Returns the catalog of 13 built-in demonstration cases.

---

## 💻 CLI & Pipeline Execution (`main.py`)

The unified orchestrator `main.py` provides command-line control over every pipeline stage:

```powershell
# 1. Dataset exploration (shapes, balance, missing values)
python main.py --explore

# 2. Extract 49 features from raw dataset URLs
python main.py --extract

# 3. Exploratory Data Analysis & correlation heatmaps
python main.py --eda

# 4. Preprocess: Domain-Grouped Split -> StandardScaler -> Class Weights
python main.py --preprocess

# 5. Train the URL Deep Neural Network
python main.py --train

# 6. Evaluate URL DNN & generate full report/plots
python main.py --evaluate

# 7. Predict phishing risk for a single URL
python main.py --predict "http://secure-bank-login-update.top/auth"

# 8. Predict phishing risk for an email (Subject + Body)
python main.py --predict-email "Urgent Account Alert" "Click here to verify: http://paypal-verify.xyz"

# 9. Run interactive multimodal CLI demo
python main.py --merged

# 10. Run multimodal evaluation on test dataset
python main.py --merged-eval

# 11. Run complete pipeline end-to-end
python main.py --all
```

---

## 🖼️ Research Figures & Visual Artifacts

The repository provides five publication-grade figures in [`artifacts/`](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/artifacts):
1. **System Architecture**: [`artifacts/figure_architecture.png`](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/artifacts/figure_architecture.png) (DistilBERT + URL DNN + Auxiliary Scanners + Bayesian Noisy-OR Fusion).
2. **ROC and Precision-Recall Curves**: [`artifacts/figure_roc_pr_curves.png`](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/artifacts/figure_roc_pr_curves.png) (DistilBERT vs. URL DNN vs. Random Forest vs. LightGBM on registered-domain split).
3. **Confusion Matrices**: [`artifacts/figure_confusion_matrices.png`](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/artifacts/figure_confusion_matrices.png) (Holdout evaluations for text and URL classifiers).
4. **Benchmark A vs. B Evasion Comparison**: [`artifacts/figure_benchmark_a_b_comparison.png`](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/artifacts/figure_benchmark_a_b_comparison.png) (Illustrates severe link distribution shift between in-email links and web links).
5. **Reliability & Calibration Plot**: [`artifacts/calibration_reliability_diagram.png`](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/artifacts/calibration_reliability_diagram.png) (Expected calibration error across probability deciles).

---
## 🧪 Empirical Studies & Model Efficiency (Tier 3 Benchmarks)

### 1. Model Latency, Runtime Overhead & Memory Footprint (RSS)
Timing measurements distinguish between direct callable model execution and API dispatch overhead, while memory is evaluated as operating system Resident Set Size (RSS):

| Component / Subsystem | Execution Protocol | Hardware Acceleration | Latency per Sample (Mean) | Process Memory (RSS) |
| :--- | :--- | :---: | :---: | :---: |
| **URL 49-Feature Extraction** | Pure Python lexical parser | CPU (Single Core) | **`0.035 ms`** | Included below |
| **URL DNN (Direct `model()`)** | Single-sample (Direct tensor call) | CPU (Single Core) | **`3.159 ms`** | **`712.80 MB`** |
| **URL DNN (Batched)** | Batch size = 128 | CPU (Single Core) | **`0.030 ms`** | **`712.80 MB`** |
| **URL DNN (Keras `predict()`)** | Single-sample (Includes API overhead) | CPU (Single Core) | **`45.714 ms`** | **`712.80 MB`** |
| **DistilBERT Transformer** | Single-sample (Sequence length 128) | GPU (NVIDIA RTX) | **`3.08 ms`** | **`2,786.34 MB`** |
| **DistilBERT Transformer** | Single-sample (Sequence length 128) | CPU (Fallback Mode) | **`19.58 ms`** | **`2,786.34 MB`** |
| **Header Auth Scanner** | MIME header regex parser | CPU (Single Core) | **`< 0.85 ms`** | Included below |
| **Attachment Scanner** | OpenCV QR + PyPDF stream audit | CPU (Single Core) | **`12.40 ms`** | Included below |

*Runtime Memory Takeaway*: Active deep learning runtime environments (TensorFlow, PyTorch, CUDA libraries) require **~2.78 GB of active process RSS memory**, while achieving high single-sample throughput on commodity hardware.

### 2. Multi-Method Feature Attribution & Explainability
- **TreeSHAP vs. Gradient Attribution**: Evaluated across 2,000 held-out test URLs. LightGBM TreeSHAP highlights `has_https` (1.237), `has_www` (0.889), and `path_length` (0.704) as leading features, while partly disagreeing with DNN gradient rankings (`has_www`, `longest_word_length`, `tld_length`). As confirmed by protocol ablation, boolean flags reflect dataset artifacts, whereas structural path and hostname properties provide consistent threat discriminators.
- **DistilBERT Token Attribution**: Gradient norm attribution aggregated across 50 holdout phishing emails confirms that predictions focus heavily on coercive action verbs (`click`, `urgent`, `verify`) coupled with sensitive assets (`account`, `mail`, `access`).

### 3. Error Analysis on Clean Holdout Emails ($n = 18,237$)
DistilBERT achieves $99.73\%$ accuracy with exactly **33 False Positives** and **17 False Negatives**:
- **False Positives ($n = 33$)**: Driven primarily by promotional retail marketing urgency ($45.5\%$) and automated corporate system notifications ($30.3\%$).
- **False Negatives ($n = 17$)**: Driven by mundane, low-urgency conversational business correspondence lacking explicit threat terms ($47.1\%$) and obscure text formatting ($29.4\%$).

### 4. Functional Testing of Auxiliary Security Scanners ($N = 200$)
- **Header Security & Spoofing Scanner**: Successfully parsed and flagged all $100$ simulated SPF/DKIM/DMARC and display-name spoofing violations while clearing $100$ compliant legitimate headers.
- **Attachment & Quishing QR Scanner**: Successfully parsed and flagged all $50$ synthetic QR code phishing payloads and $50$ PDF `/Launch` action streams while clearing $100$ clean document controls.

---

## ⚙️ Hardware & Optimization

- **Hardware Acceleration**: Uses PyTorch with native CUDA acceleration for DistilBERT and TensorFlow/Keras on CPU/DirectML.
- **Tested Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) + Intel Core i5/i7.
- **Single-Sample Inference**:
  - URL feature extraction + direct DNN tensor call: **~3.2 ms** per URL (or **0.030 ms** batched).
  - DistilBERT transformer inference on GPU: **~3.1 ms** per email (or **19.6 ms** on CPU).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

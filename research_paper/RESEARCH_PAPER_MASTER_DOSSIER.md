# Empirical Limits of Multimodal Threat Fusion Under Link Distribution Shift in Phishing Detection

---

## Abstract
Machine learning systems for phishing detection frequently report near-perfect detection metrics (>99% accuracy) on public benchmark corpora. However, these figures obscure critical distribution shifts and evaluation pitfalls encountered when defending against modern attack vectors. In this paper, we conduct an empirical investigation into multimodal phishing detection, combining a 66M-parameter DistilBERT transformer for natural language email analysis and a 49-feature lexical Deep Neural Network (DNN) for URL inspection. Our findings expose four central bounds:
1. **Random Splits Overstate Generalization**: Randomly partitioned splits mask severe cross-source fragility. Evaluating web URL models across distinct source repositories reveals accuracies ranging from 34.02% to 95.82%, while email transformer performance drops from 99.73% in-distribution to 83.9%–90.8% on unseen corpora. Furthermore, a simple TF-IDF + Linear SVM baseline matches DistilBERT in-distribution (99.69% vs. 99.73%), with the transformer providing only a 1.7 to 3.1 percentage point advantage out-of-distribution.
2. **Failure of Web-Trained URL Models on Email Links**: Lexical URL models trained on public web crawl repositories degrade to near-chance discrimination when applied to in-email links (accuracy 42.04%, ranking AUC ~0.51). Ablation experiments confirm that this failure is driven by complex newsletter, CRM tracking, and redirect query syntax rather than missing HTTPS or WWW indicators.
3. **Stealth Evasion Bounds**: On synthetic stealth evasion emails where weaponized links are placed inside benign, conversational text bodies, text-only transformers miss virtually all attacks (0.49% recall). Under a leakage-safe domain-grouped protocol ($N = 7,382$ across $3,999$ registered domains), multimodal fusion recovers detection to 70.51% ± 6.82% recall at 63.88% ± 10.28% specificity (AUC ~0.76).
4. **Stacker Controls & Calibration**: Non-linear Gradient Boosted Decision Tree (GBDT) stacking yields no reliable improvement over linear stacking (77.43% ± 3.12% vs. 75.89% ± 2.36% AUC, within 1 standard deviation). Control experiments demonstrate that any marginal shift is driven entirely by non-linear URL probability calibration rather than multimodal synergy.
5. **Operational False Positive Penalty**: Operating at high sensitivity (95% TPR) on raw in-email links forces an unmanageable 74.9%–77.7% false-positive rate across all model variants. Commercial tracking domain whitelists matched only 0.33% (12 / 3,691) of clean links in historical corpora, demonstrating that static whitelists do not resolve the operating penalty.

---

## 1. Introduction

Automated email security gateways have largely transitioned from static signature rules toward machine learning pipelines. Typically, modern perimeter architectures employ deep language models (e.g., BERT, DistilBERT) to analyze natural language intent and tree ensembles or neural networks to inspect hyperlink syntax.

While standalone benchmark evaluations report near-flawless performance, operational deployments encounter acute failure modes:
- **Decoupled Pretexting Attacks**: Attackers compose innocuous, non-urgent conversational messages containing an embedded weaponized URL. Because the text carries no threat intent, natural language detectors assign low risk.
- **Link Distribution Shift**: Intra-enterprise email links (newsletters, CRM tracking redirects, webinar invites) exhibit deep paths, parameter tokens, and complex queries that share structural characteristics with web phishing payloads, confusing web-trained classifiers.
- **Evaluation Artifacts**: Randomly shuffling URLs or emails across train and test sets allows lexical memorization of domain and formatting shortcuts, masking real-world generalization failure.

This paper provides an empirical study characterizing these limitations. We evaluate cross-source degradation, construct leakage-safe stealth evasion benchmarks, analyze the true drivers of URL distribution shift, and characterize the operational false alarm trade-offs.

---

## 2. Related Work

- **Lexical URL Detection**: Ma et al. (2009) demonstrated that lexical host and path properties (lengths, token composition, delimiter counts) enable online classification of malicious URLs without live content downloads. Sahoo et al. (2017) surveyed feature engineering across lexical, host-based, and popularity dimensions. Hannousse & Yahiouche (2021) compared machine learning and deep learning models across 287 lexical and structural features on curated web catalogs. These foundational studies evaluated almost exclusively on web crawl data, leaving intra-email redirection syntax unaddressed.
- **Deep NLP for Phishing Email Detection**: Following BERT (Devlin et al. 2019) and DistilBERT (Sanh et al. 2019), transformer architectures achieved high reported accuracy on public email corpora (Alhogail & Alsanie 2021). Roy et al. (2023) evaluated transformer generalization across public corpora, identifying vulnerabilities to cross-dataset distribution shifts and vocabulary artifacts, consistent with our empirical findings where accuracy drops by 10–15 percentage points out-of-domain.
- **Multimodal Decision Fusion in Security**: While multimodal architectures frequently deploy heuristic thresholding or logical OR gates, probabilistic decision fusion via Bayesian Noisy-OR provides a mathematically grounded framework ($1 - (1 - P_E)(1 - P_U)$). Our findings characterize the calibration breakdown of this model when URL inputs experience severe distribution shift.
- **Auxiliary Vectors & Social Engineering**: Vasek et al. (2022) surveyed modern email threats, noting multi-stage delivery and display-name impersonation techniques, underscoring the role of auxiliary MIME header audits and attachment stream parsing.

---

## 3. Datasets & Experimental Protocol

### 3.1. Primary Datasets
1. **Email Corpora ($N = 182,364$)**: Aggregated across six public benchmark archives:
   - CEAS_08 ($38,964$ emails; 55.6% phishing)
   - Enron Corporate ($29,424$ emails; 47.4% phishing)
   - Kaggle Phishing ($17,077$ emails; 50.0% phishing)
   - Apache SpamAssassin ($5,798$ emails; 29.5% phishing)
   - Ling-Spam ($2,859$ emails; 16.0% phishing)
   - Nazario & Nigerian Fraud ($4,840$ phishing emails)
   - Zionia Phishing archive ($81,139$ emails)
2. **Web URL Corpora ($N = 600,000+$)**: Collected from public web phishing and benign feeds (Phiusiil, OpenPhish, PhishTank, URLHaus, Tranco top domains, and `phishbd_2026`).

### 3.2. Splitting Protocols
- **Random Stratified Split**: 80% Train ($145,891$), 10% Validation ($18,236$), 10% Test ($18,237$).
- **Domain-Grouped Split (`eTLD+1`)**: To evaluate lexical URL generalization, URLs were partitioned by registered domain using `GroupShuffleSplit`, ensuring no domain appears in both training and test sets ($124,090$ holdout URLs).
- **Leave-One-Source-Out (LOSO)**: Complete source collections (e.g., all Enron or all CEAS_08) held out entirely to measure out-of-domain transfer.

---

## 4. Methodology: Models & Multimodal Fusion

### 4.1. Natural Language Processing Engine (DistilBERT)
We fine-tune a pre-trained DistilBERT transformer (66M parameters) on the $145,891$-email training set for 4 epochs ($9,120$ optimizer steps) with AdamW ($\text{lr} = 2 \times 10^{-5}$, linear warmup, weight decay 0.01, batch size 64). Sequences are truncated to 128 tokens covering concatenated subject and body text. The model outputs scalar probability $P_{\text{text}} \in [0, 1]$.

### 4.2. Lexical URL Engine (49-Feature Deep Neural Network)
We extract 49 offline numeric features capturing:
- **Structural lengths & counts**: URL length, hostname length, path length, query length, subdirectory depth.
- **Character metrics**: Delimiter counts (`.`, `-`, `_`, `=`, `?`, `&`, `%`, `/`), digit count, special character ratio.
- **Statistical entropy**: Shannon entropy of the full URL, hostname, and path string.
- **Lexical heuristics**: IPv4 host match, presence of known URL shorteners, presence of security/credential keywords, suspicious TLD match.

The classification network comprises 4 fully connected layers:
$$\text{Dense}(128) \to \text{BatchNorm} \to \text{Dropout}(0.3) \to \text{Dense}(64) \to \text{BatchNorm} \to \text{Dropout}(0.3) \to \text{Dense}(32) \to \text{Sigmoid}$$
Trained using Adam ($\text{lr} = 10^{-3}$, cosine decay) with binary cross-entropy and cost-sensitive class re-weighting, outputting $P_{\text{URL}} \in [0, 1]$.

### 4.3. Probabilistic Threat Fusion
Outputs are integrated via Bayesian Noisy-OR:
$$P_{\text{combined}} = 1 - (1 - P_{\text{text}})\prod_{i=1}^{M} (1 - P_{\text{URL}_i})$$

**Monotonic Safeguard Property**: If any link receives high threat confidence ($P_{\text{URL}} \ge 0.85$):
$$P_{\text{combined}} \ge 1 - (1 - 0)(1 - 0.85) = 0.85$$
This mathematically guarantees high threat escalation without requiring ad-hoc manual override branches.

We contrast Noisy-OR against a learned logistic stacker:
$$P_{\text{stacker}} = \sigma\left(\beta_0 + \beta_1 P_{\text{text}} + \beta_2 \max_i P_{\text{URL}_i}\right)$$

---

## 5. Evaluation Pitfalls: Random vs. Grouped vs. Cross-Source Splits

### 5.1. The Illusion of Randomly Shuffled Splits
When evaluated on randomly partitioned test splits, both modalities appear exceptionally accurate:
- **DistilBERT Email Test ($n = 18,237$)**: Accuracy = **99.73%**, Recall = 99.81%, Specificity = 99.64%, ROC-AUC = 0.9999 ($\text{TN} = 9,045$, $\text{FP} = 33$, $\text{FN} = 17$, $\text{TP} = 9,142$).
- **URL DNN Registered-Domain Holdout ($n = 124,090$)**: Accuracy = **96.30%**, Recall = 95.68%, Specificity = 96.75%, ROC-AUC = 0.9918 ($\text{TN} = 68,905$, $\text{FP} = 2,314$, $\text{FN} = 2,283$, $\text{TP} = 50,588$).

### 5.2. Baseline Comparison: Linear Models Match Transformers In-Distribution
Comparing DistilBERT against standard n-gram linear models reveals that deep transformers provide negligible benefit on standard in-distribution test splits:

| Model Pipeline | Test Accuracy | F1-Score | ROC-AUC | Precision | Recall | Inference Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF (50k n-grams) + Logistic Regression** | 99.35% | 0.9935 | 0.9994 | 99.18% | 99.52% | ~12,000 emails/s |
| **TF-IDF (50k n-grams) + Linear SVM** | 99.69% | 0.9969 | 0.9999 | 99.61% | 99.77% | ~7,000 emails/s |
| **DistilBERT Transformer (Fine-Tuned)** | **`99.73%`** | **`0.9973`** | **`0.9999`** | **`99.64%`** | **`99.81%`** | **~574 emails/s (GPU)** |

### 5.3. Out-of-Distribution Degradation (Cross-Source & LOSO)
When evaluated out-of-distribution across unseen datasets, performance drops substantially:
- **Email Transformer Cross-Corpus Degradation**: DistilBERT accuracy falls to **89.62%** on Enron (ROC-AUC 0.9735) and **86.52%** on CEAS_08 (ROC-AUC 0.9576). The transformer holds only a modest 1.7 to 3.1 percentage point advantage over a simple TF-IDF + Linear SVM (which achieves 87.88% on Enron and 83.42% on CEAS_08).
- **URL Cross-Source Fragility**: Evaluating the URL DNN across isolated source repositories shows extreme variance:

| URL Source Repository | Evaluation Partition ($N$) | Accuracy | Recall | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`phiusiil`** | 100,000 | **95.82%** | 94.70% | 96.65% | 0.9525 |
| **`phishbd_2026`** | 26,227 | **94.16%** | 98.01% | 85.17% | 0.9592 |
| **`raw_dataset`** | 100,000 | **89.37%** | 92.20% | 86.82% | 0.8870 |
| **`email_extracted`** | 10,000 | **`42.04%`** | 89.20% | **`3.84%`** | 0.5898 |
| **`internal_val`** | 2,500 | **`34.02%`** | 95.80% | **`1.82%`** | 0.4907 |

---

## 6. The Link Distribution Shift Dilemma

When the web-trained URL model is exposed to links extracted from corporate emails, accuracy collapses to **42.04%** and specificity drops to **3.84%** (flagging 96% of clean email links as phishing).

### Feature Ablation: Syntax vs. Protocols
To identify the root cause of this failure, we ablated protocol flags (`has_https`, `has_www`):
- Removing `has_https` and `has_www` shifts validation AUC by less than 0.002 on web data.
- On email links, the protocol-ablated model still fails completely (accuracy remains <45%).
- **Empirical Cause**: Legitimate enterprise correspondence heavily employs marketing redirect engines, CRM tracking parameters, and unsubscribe endpoints (`http://e.joost.com/I?a=...`). These links feature heavy delimiter density, long paths, and dense query tokens that directly overlap with web phishing feature distributions. The failure is driven by **link syntax**, not protocol indicators.

---

## 7. Stealth Evasion Benchmarks (A vs. B)

To evaluate defense against stealth attacks where text intent is decoupled from link payloads, we created two non-circular benchmarks:
- **Benchmark A (In-Email Links)**: Evaluates real in-email links from historical mail archives.
- **Benchmark B (Web-Style Links)**: Evaluates web-style links drawn from the registered-domain holdout ($n = 1,000$).

```
Figure 4: Benchmark A vs. Benchmark B Stealth Evasion Performance
Panel A (In-Email Links, N=7,382)          Panel B (Web Links, N=1,000)
100% ┌──────────────────────────┐          100% ┌──────────────────────────┐
     │  Rec   Spec              │               │  Rec   Spec              │
 80% │        [70.8]            │           80% │  [96.0][96.0] [96.4][96.4]│
     │  [67.8] [63.7] [72.8]    │               │                          │
 60% │  [59.7]        [62.0]    │           60% │        [50.1]            │
     │                          │               │                          │
 40% │                          │           40% │                          │
     │                          │               │                          │
  0% └──────────────────────────┘            0% └──────────────────────────┘
      URL   Adapted  Linear                      Text    URL   Linear
      Base    URL    Stacker                     Only    Base  Stacker
```

### Side-by-Side Results:

| Model Configuration | Benchmark A Recall | Benchmark A Specificity | Benchmark A Accuracy | Benchmark B Recall | Benchmark B Specificity | Benchmark B Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text Only (DistilBERT)** | **`0.49%`** ($\pm 0.3$) | **`100.0%`** ($\pm 0.0$)* | 50.2% | **`1.70%`** ($\pm 0.8$) | **`50.13%`** ($\pm 3.1$) | 50.25% |
| **Standard URL DNN** | 67.82% ($\pm 2.6$) | 59.69% ($\pm 0.9$) | 63.8% | 96.00% ($\pm 1.2$) | 96.03% ($\pm 1.2$) | 96.40% |
| **Email-Adapted URL DNN** | 70.84% ($\pm 6.5$) | 63.73% ($\pm 10.2$) | 67.4% | 87.70% ($\pm 2.0$) | 42.25% ($\pm 3.1$) | 48.35% |
| **Bayesian Noisy-OR** | **`70.51%`** ($\pm 6.8$) | **`63.88%`** ($\pm 10.3$) | 67.3% | **`96.00%`** ($\pm 1.2$) | **`95.98%`** ($\pm 1.2$) | 95.75% |
| **Linear Stacker** | **`72.77%`** ($\pm 4.8$) | **`62.03%`** ($\pm 8.3$) | **`67.45%`** | **`96.40%`** ($\pm 1.1$) | **`96.36%`** ($\pm 1.1$) | **`95.90%`** |
| **GBDT Stacker (Ablation)** | 66.78% ($\pm 11.0$) | 72.11% ($\pm 7.0$) | 69.25% | — | — | — |

*\*In Benchmark A, candidate bodies were pre-filtered to $P_{\text{text}} < 0.10$ to prevent text label leakage, yielding $100.0\%$ specificity by construction. In Benchmark B, bodies were unconstrained, yielding $50.1\%$ baseline specificity.*

---

## 8. Controlled Ablations & Stacker Analysis (Section 6.5)

To establish rigorous ground truth under zero-leakage conditions, we assembled an expanded Benchmark A ($N = 7,382$ samples across $3,999$ domains), pairing unique held-out email bodies ($P_{\text{text}} < 0.10$) with real in-email links under 5-seed `GroupShuffleSplit` (60% Train, 20% Val, 20% Test). Thresholds were tuned strictly on Validation:

| Model Configuration | Inputs / Features | ROC-AUC (%) | Accuracy (%) | Recall (%) | Specificity (%) | Recall @ 90% Spec (%) | FPR @ 95% TPR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bayesian Noisy-OR** | $P_T, P_U$ (No fitting; $\tau$ on Val) | $75.90 \pm 2.35$ | $67.32 \pm 1.91$ | $70.51 \pm 6.82$ | $63.88 \pm 10.28$ | $41.16 \pm 2.41$ | $77.74 \pm 4.27$ |
| **URL Adapted Alone** | $P_{\text{url\_adapted}}$ | $75.84 \pm 2.35$ | $67.41 \pm 1.93$ | $70.84 \pm 6.49$ | $63.73 \pm 10.18$ | $40.78 \pm 2.30$ | $77.68 \pm 4.33$ |
| **Linear Stacker (Main)** | $P_T, P_U$ | $75.89 \pm 2.36$ | $67.45 \pm 1.91$ | $72.77 \pm 4.83$ | $62.03 \pm 8.33$ | $40.88 \pm 2.16$ | $77.53 \pm 4.01$ |
| **Linear Stacker + Base URL** | $P_T, P_U, P_{\text{base}}$ | $76.48 \pm 2.84$ | $69.15 \pm 2.26$ | $59.78 \pm 2.48$ | $78.46 \pm 3.48$ | $44.99 \pm 4.01$ | $77.23 \pm 2.90$ |
| **GBDT (Full Multimodal)** | $P_T, P_U, P_B, \text{Prod}, \max, \|\text{Diff}\|$ | **`77.43 ± 3.12`** | **`69.25 ± 2.82`** | $66.78 \pm 10.98$ | $72.11 \pm 6.95$ | **`43.27 ± 3.06`** | **`74.94 ± 6.62`** |
| **Control 1: GBDT No Text** | $P_U, P_B, \max, \|\text{Diff}\|$ | $77.56 \pm 3.14$ | $69.34 \pm 2.68$ | $66.07 \pm 5.15$ | $72.42 \pm 7.38$ | $42.52 \pm 3.69$ | $75.69 \pm 4.26$ |
| **Control 2: URL Scores Only** | $P_{\text{url\_adapted}}, P_{\text{url\_base}}$ | $77.59 \pm 3.13$ | $69.74 \pm 3.03$ | $68.76 \pm 3.98$ | $70.65 \pm 4.87$ | $43.58 \pm 4.48$ | $75.49 \pm 5.59$ |
| **Control 3: Shuffled Labels** | Permuted Labels (Leakage Test) | **`48.46 ± 2.87`** | **`50.46 ± 0.97`** | $69.44 \pm 20.28$ | $31.54 \pm 19.84$ | **`7.76 ± 1.94`** | **`95.97 ± 2.27`** |

### Key Experimental Insights:
1. **No Reliable Difference between GBDT and Linear Stacking**: GBDT AUC ($77.43\% \pm 3.12\%$) overlaps with the linear stacker ($75.89\% \pm 2.36\%$) within one standard deviation.
2. **Gain Derived from URL Calibration, Not Multimodal Synergy**: Control 1 (no text, $77.56\%$) and Control 2 (URL scores only, $77.59\%$) match the full GBDT ($77.43\%$). Because the email body is non-malicious by design, the tree model extracts zero predictive signal from text interaction; the marginal shift is driven entirely by non-linear calibration between URL scores.
3. **Leakage-Safe Integrity**: Fitting GBDT on permuted labels collapses test AUC to $48.46\% \pm 2.87\%$, confirming no information leakage across feature pipelines or split boundaries.

---

## 9. Operational Calibration & Deployment Bounds

### 9.1. Expected Calibration Error (ECE)
- **DistilBERT Holdout ($n = 18,237$)**: $\text{ECE} = 0.25\%$ ($9,043$ emails in $[0.0–0.1]$, $9,162$ in $[0.9–1.0]$, only $32$ intermediate emails).
- **Benchmark B (Web Links)**: $\text{ECE} = 1.17\%$.
- **Benchmark A (In-Email Links, $N = 7,382$)**: $\text{ECE} = 6.19\%$ (the pilot testbed exhibited $10.52\%$). Calibration analysis reveals mixed calibration: the model is over-confident between $0.2$ and $0.7$ (observed threat rate is lower than confidence) and under-confident above $0.75$.

### 9.2. Operating False Positive Penalty
In enterprise environments, perimeter filters operate under strict false alarm budgets. When evaluated on in-email links:
- **Achieving 95% Detection Sensitivity (Recall) forces a 74.9% to 77.7% False Positive Rate** across all model variants ($T \approx 0.22$).
- In an operational gateway, four out of five clean emails would be flagged.
- **Whitelist Failure**: Matching clean links against commercial ESP/CRM tracking domains (Mailchimp, SendGrid, HubSpot, Marketo, Salesforce, Constant Contact) matched only **0.33% (12 / 3,691)** of clean links. Static whitelisting does not mitigate this penalty on historical corpora.

---

## 10. Limitations

1. **Synthetic Stealth Pairings**: Because public email archives lack paired benign correspondence with confirmed embedded zero-day exploits, Benchmark A combines real text bodies with real links synthetically.
2. **Single-Seed DistilBERT Retraining**: While URL models were evaluated across 5 seeds and tree baselines, full DistilBERT retraining across all splits was constrained by computational budgets.
3. **Text-Only Corpora & Lack of Rendered HTML**: Feature extraction was performed on raw email strings without rendering dynamic DOM structures, CSS evasion, or live DNS/WHOIS resolution.
4. **Historical Label Noise**: Borderline commercial solicitations in Enron and CEAS_08 introduce historical label ambiguity.
5. **Single-Annotator Error Analysis**: Qualitative error taxonomy was categorized by a single researcher without multi-annotator inter-rater reliability scores.
6. **No Genuine Temporal Evaluation**: Due to public feed constraints, evaluation relied on source-shift splits rather than longitudinal prospective feeds with explicit capture timestamps.
7. **Transformer Retraining Scope**: Full DistilBERT leave-one-source-out retraining was executed on the two largest sources (Enron and CEAS_08, >80% of data), with n-gram SVM baselines reported across the remaining four.
8. **In-Corpus Domain Familiarity**: An audit revealed that 92.3% of the 3,999 benchmark domains appeared in the training archive used to adapt the URL model. While the stacking split is strictly out-of-fold, **0.76 AUC remains an in-corpus optimistic bound**, and true generalization to completely novel external enterprise domains will face additional degradation.

---

## 11. Conclusion

This empirical investigation demonstrates the sharp bounds of multimodal phishing detection when confronted with link distribution shift and stealth pretexting. While multimodal fusion provides vital protection against text-decoupled attacks (rescuing recall from near-zero up to ~71%), machine learning models alone cannot achieve high sensitivity without severe false positive penalties on intra-enterprise tracking links. Practical perimeter security requires combining neural decision fusion with dynamic, link-level reputation and contextual provenance features rather than relying on unassisted classifiers or static domain whitelists.

---

## 12. Publication Figures & Artifacts

All figures are compiled in `research_paper/`:
- **Figure 1**: [figure_architecture.png](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/research_paper/figure_architecture.png) (System architecture and inference pipeline).
- **Figure 2**: [figure_roc_pr_curves.png](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/research_paper/figure_roc_pr_curves.png) (ROC/PR curves with operational inset zoom).
- **Figure 3**: [figure_confusion_matrices.png](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/research_paper/figure_confusion_matrices.png) (Holdout confusion matrices).
- **Figure 4**: [figure_benchmark_a_b_comparison.png](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/research_paper/figure_benchmark_a_b_comparison.png) (Stealth evasion bars with seed error bars).
- **Figure 5**: [calibration_reliability_diagram.png](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/research_paper/calibration_reliability_diagram.png) (Probability calibration curves).
- **Figure 6**: [figure_benchmark_a_ablation.png](file:///c:/Users/Vibhu-Kavi/Downloads/Phishing%20Shield%20AI/research_paper/figure_benchmark_a_ablation.png) (Model ablation: AUC vs. Recall @ 90% Specificity).

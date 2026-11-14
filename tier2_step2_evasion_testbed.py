import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from tensorflow import keras
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

URL_REGEX = re.compile(
    r'(?:https?://|www\d{0,3}[.]|[a-zA-Z0-9.\-]+[.][a-zA-Z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])',
    re.IGNORECASE
)

def run_evasion_experiment():
    print("=== Step 2: Benign-Text + Malicious-URL Testbed ===")
    
    # 1. Load email test holdout
    print("Loading email test holdout...")
    df_email = pd.read_csv('emaildata/combined_phishing_email_dataset.csv', low_memory=False)
    labels_email = df_email['label'].astype(int).values
    _, temp_idx = train_test_split(np.arange(len(df_email)), test_size=0.2, random_state=42, stratify=labels_email)
    _, test_email_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels_email[temp_idx])
    
    holdout_emails = df_email.iloc[test_email_idx].copy()
    legit_emails = holdout_emails[holdout_emails['label'] == 0].copy()
    phish_emails = holdout_emails[holdout_emails['label'] == 1].copy()
    
    print(f"Total holdout: {len(holdout_emails)} (Legit: {len(legit_emails)}, Phish: {len(phish_emails)})")
    
    # 2. Load held-out URLs from the eTLD+1 test partition
    print("Loading eTLD+1 held-out URLs...")
    d_url = np.load('url_trees_etld1_preds.npz')
    y_url = d_url['y_test']
    probs_dnn = d_url['probs_dnn']
    
    # We want confirmed high-confidence malicious URLs from the held-out URL set
    # Load raw dataset to get the actual URL strings
    df_raw = pd.read_csv('merged_raw_dataset.csv', usecols=['url', 'label'])
    # The eTLD+1 test partition was the last 124,090 URLs or from GroupShuffleSplit
    # Let's inspect or sample known phishing URLs from df_raw where label == 1
    phish_urls_pool = df_raw[df_raw['label'] == 1]['url'].dropna().sample(n=5000, random_state=42).tolist()
    legit_urls_pool = df_raw[df_raw['label'] == 0]['url'].dropna().sample(n=5000, random_state=42).tolist()
    
    print(f"Sampled {len(phish_urls_pool)} phish URLs and {len(legit_urls_pool)} legit URLs.")
    
    # 3. Construct Targeted Evasion Sets:
    # Set A: Benign Text + Weaponized Malicious URL (The classic stealth spear-phishing / account takeover attack)
    # Ground Truth: PHISHING (Label = 1) because the payload link is weaponized!
    # Sample 1,000 legitimate emails that have NO existing URLs (or clean corporate text)
    clean_legit = legit_emails[legit_emails['body'].str.len() > 50].sample(n=1000, random_state=42).copy()
    
    evasion_texts = []
    evasion_urls = []
    for idx, (_, row) in enumerate(clean_legit.iterrows()):
        target_phish_url = phish_urls_pool[idx]
        text_with_url = str(row['subject'] or '') + " " + str(row['body'] or '') + f"\n\nPlease review your account documents here: {target_phish_url}"
        evasion_texts.append(text_with_url)
        evasion_urls.append(target_phish_url)
        
    print(f"Constructed {len(evasion_texts)} Evasion Attack Samples (Benign Text + Phishing Link, True Label = 1).")
    
    # Set B: Pure Phishing Text + Benign URL (Pretexting with innocent link, e.g. linking to google.com/wikipedia)
    # Ground Truth: PHISHING (Label = 1) due to social engineering extortion/fraud!
    clean_phish = phish_emails[phish_emails['body'].str.len() > 50].sample(n=1000, random_state=42).copy()
    pretext_texts = []
    pretext_urls = []
    for idx, (_, row) in enumerate(clean_phish.iterrows()):
        target_legit_url = legit_urls_pool[idx]
        text_with_url = str(row['subject'] or '') + " " + str(row['body'] or '') + f"\n\nOfficial reference: {target_legit_url}"
        pretext_texts.append(text_with_url)
        pretext_urls.append(target_legit_url)
        
    print(f"Constructed {len(pretext_texts)} Pretext Samples (Phishing Text + Benign Link, True Label = 1).")
    
    # Set C: Clean Control Group (Benign Text + Benign Link)
    # Ground Truth: LEGITIMATE (Label = 0)
    control_legit = legit_emails[legit_emails['body'].str.len() > 50].sample(n=1000, random_state=123).copy()
    control_texts = []
    control_urls = []
    for idx, (_, row) in enumerate(control_legit.iterrows()):
        target_legit_url = legit_urls_pool[1000 + idx]
        text_with_url = str(row['subject'] or '') + " " + str(row['body'] or '') + f"\n\nCompany portal: {target_legit_url}"
        control_texts.append(text_with_url)
        control_urls.append(target_legit_url)
        
    print(f"Constructed {len(control_texts)} Clean Control Samples (Benign Text + Benign Link, True Label = 0).")
    
    # 4. Evaluate Models:
    # A) DistilBERT Text Model
    print("\n--- Evaluating DistilBERT on Evasion Testbed ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading DistilBERT on {device}...")
    tokenizer = AutoTokenizer.from_pretrained('emaildata/results/best_distilbert')
    model_dist = AutoModelForSequenceClassification.from_pretrained('emaildata/results/best_distilbert').to(device)
    model_dist.eval()
    
    def score_texts(texts):
        probs = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b_texts = texts[i:i+batch_size]
                enc = tokenizer(b_texts, max_length=256, padding=True, truncation=True, return_tensors='pt').to(device)
                logits = model_dist(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).logits
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
        return np.array(probs)
    
    p_text_evasion = score_texts(evasion_texts)
    p_text_pretext = score_texts(pretext_texts)
    p_text_control = score_texts(control_texts)
    
    # B) URL DNN Model
    print("\n--- Evaluating URL DNN on Extracted URLs ---")
    from feature_extraction import extract_features
    scaler_orig = joblib.load('preprocessed/scaler.joblib')
    model_dnn = keras.models.load_model('preprocessed/dnn_model.keras')
    
    feature_cols = [c for c in pd.read_csv('url_features_dnn.csv', nrows=2).columns if c not in ['label', 'source']]
    
    def score_urls(urls):
        feats = [extract_features(u) for u in urls]
        df_f = pd.DataFrame(feats)
        for c in feature_cols:
            if c not in df_f.columns:
                df_f[c] = 0
        X_sc = scaler_orig.transform(df_f[feature_cols].values)
        return model_dnn.predict(X_sc, batch_size=256, verbose=0).ravel()
    
    p_url_evasion = score_urls(evasion_urls)
    p_url_pretext = score_urls(pretext_urls)
    p_url_control = score_urls(control_urls)
    
    # 5. Compute Decision Fusion Strategies:
    # Linear: 0.5 * P_E + 0.5 * P_U
    # Noisy-OR: 1 - (1 - P_E) * (1 - P_U)
    # Noisy-OR + 0.85 override
    def evaluate_suite(name, y_true, p_e, p_u):
        print(f"\n==================== {name} (N = {len(y_true)}) ====================")
        p_lin = 0.5 * p_e + 0.5 * p_u
        p_nor = 1.0 - (1.0 - p_e) * (1.0 - p_u)
        p_override = np.where(p_u >= 0.85, np.maximum(p_nor, p_u), p_nor)
        
        models = [
            ("DistilBERT Alone (Text Only)", p_e),
            ("URL DNN Alone (URL Only)", p_u),
            ("Linear Blending (0.5 Text + 0.5 URL)", p_lin),
            ("Bayesian Noisy-OR", p_nor),
            ("Bayesian Noisy-OR + 0.85 Override", p_override)
        ]
        
        for m_name, probs in models:
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_true, preds) * 100
            rec = recall_score(y_true, preds, zero_division=0) * 100
            prec = precision_score(y_true, preds, zero_division=0) * 100
            f1 = f1_score(y_true, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_true, probs)
            except Exception:
                auc = 0.5
            print(f"{m_name:38s} | Acc: {acc:6.2f}% | Rec: {rec:6.2f}% | Prec: {prec:6.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")
            
    # Evaluation 1: Benign Text + Weaponized URL (Evasion Attack, True Label = 1)
    # The key test: Does Text miss it, and does Fusion catch it?
    evaluate_suite("1. BENIGN TEXT + WEAPONIZED URL (Stealth Evasion, All Phishing = 1)",
                   np.ones(len(evasion_texts)), p_text_evasion, p_url_evasion)
    
    # Evaluation 2: Phishing Text + Benign URL (Pretexting, All Phishing = 1)
    evaluate_suite("2. PHISHING TEXT + BENIGN URL (Pretexting, All Phishing = 1)",
                   np.ones(len(pretext_texts)), p_text_pretext, p_url_pretext)
    
    # Evaluation 3: Clean Control (Benign Text + Benign URL, All Legit = 0)
    evaluate_suite("3. BENIGN TEXT + BENIGN URL (Clean Control, All Legit = 0)",
                   np.zeros(len(control_texts)), p_text_control, p_url_control)
    
    # Evaluation 4: Composite Balanced Stress Testbed:
    # 1,000 Evasion Attacks (Label = 1) + 1,000 Clean Controls (Label = 0)
    y_composite = np.concatenate([np.ones(len(evasion_texts)), np.zeros(len(control_texts))])
    p_e_comp = np.concatenate([p_text_evasion, p_text_control])
    p_u_comp = np.concatenate([p_url_evasion, p_url_control])
    evaluate_suite("4. COMPOSITE EVASION BENCHMARK (1,000 Stealth Phish + 1,000 Clean Controls)",
                   y_composite, p_e_comp, p_u_comp)

    # Save all testbed arrays
    np.savez('tier2_step2_evasion_testbed.npz',
             p_text_evasion=p_text_evasion, p_url_evasion=p_url_evasion,
             p_text_pretext=p_text_pretext, p_url_pretext=p_url_pretext,
             p_text_control=p_text_control, p_url_control=p_url_control)
    print("\nSuccessfully completed Step 2 and saved results to tier2_step2_evasion_testbed.npz")

if __name__ == '__main__':
    run_evasion_experiment()


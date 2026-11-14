import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from feature_extraction import extract_features

URL_REGEX = re.compile(
    r'(?:https?://|www\d{0,3}[.]|[a-zA-Z0-9.\-]+[.][a-zA-Z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])',
    re.IGNORECASE
)

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model

def run_experiment():
    print("--- 1. Loading URL features dataset ---")
    df_url = pd.read_csv('url_features_dnn.csv')
    print(f"Total URL dataset shape: {df_url.shape}")
    
    # Feature columns
    feature_cols_all = [c for c in df_url.columns if c not in ['label', 'source']]
    feature_cols_no_https_www = [c for c in feature_cols_all if c not in ['has_https', 'has_www']]
    
    print(f"Full features: {len(feature_cols_all)}, Reduced features (no https/www): {len(feature_cols_no_https_www)}")
    
    # Stratified 80/20 train/test split on URL dataset
    y_url = df_url['label'].values
    train_idx, test_idx = train_test_split(np.arange(len(df_url)), test_size=0.2, random_state=42, stratify=y_url)
    
    X_train_full = df_url[feature_cols_all].iloc[train_idx].values
    X_test_full = df_url[feature_cols_all].iloc[test_idx].values
    y_train = y_url[train_idx]
    y_test = y_url[test_idx]
    
    X_train_red = df_url[feature_cols_no_https_www].iloc[train_idx].values
    X_test_red = df_url[feature_cols_no_https_www].iloc[test_idx].values
    
    # Scale full
    scaler_full = StandardScaler()
    X_train_full_sc = scaler_full.fit_transform(X_train_full)
    X_test_full_sc = scaler_full.transform(X_test_full)
    
    # Scale reduced
    scaler_red = StandardScaler()
    X_train_red_sc = scaler_red.fit_transform(X_train_red)
    X_test_red_sc = scaler_red.transform(X_test_red)
    
    # Subsample training data to 150k for fast, stable training if needed, or train on full
    print("Training Reduced DNN (without has_https and has_www)...")
    sub_idx = np.random.RandomState(42).choice(len(X_train_red_sc), size=min(150000, len(X_train_red_sc)), replace=False)
    
    model_red = build_model(len(feature_cols_no_https_www))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5),
    ]
    model_red.fit(
        X_train_red_sc[sub_idx], y_train[sub_idx],
        validation_split=0.1,
        epochs=15,
        batch_size=512,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate on URL test split
    preds_red_url = model_red.predict(X_test_red_sc[:20000], batch_size=1024).ravel()
    acc_red_url = accuracy_score(y_test[:20000], (preds_red_url >= 0.5).astype(int))
    auc_red_url = roc_auc_score(y_test[:20000], preds_red_url)
    print(f"Reduced URL DNN on URL test (20k): Acc = {acc_red_url*100:.2f}%, AUC = {auc_red_url:.4f}")
    
    # Now extract features for all unique email URLs in the 18,237 holdout
    print("\n--- 2. Loading Holdout Email URLs ---")
    df_email = pd.read_csv('emaildata/combined_phishing_email_dataset.csv', low_memory=False)
    labels_email = df_email['label'].astype(int).values
    _, temp_idx = train_test_split(np.arange(len(df_email)), test_size=0.2, random_state=42, stratify=labels_email)
    _, test_email_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels_email[temp_idx])
    
    holdout_emails = df_email.iloc[test_email_idx].copy()
    
    email_url_mapping = [] # (email_label, [urls])
    all_unique_urls = set()
    
    for _, row in holdout_emails.iterrows():
        text = str(row['subject'] or '') + ' ' + str(row['body'] or '')
        found = [m.group(0).strip().rstrip('.,;:!?)>"\'') for m in URL_REGEX.finditer(text)]
        valid_urls = [u for u in found if len(u) > 6 and '.' in u]
        email_url_mapping.append((row['label'], valid_urls))
        for u in valid_urls:
            all_unique_urls.add(u)
            
    unique_url_list = list(all_unique_urls)
    print(f"Total unique email URLs to extract & score: {len(unique_url_list)}")
    
    # Feature extraction on unique email URLs
    print("Extracting features for unique email URLs...")
    raw_feats_list = [extract_features(u) for u in unique_url_list]
    df_email_urls = pd.DataFrame(raw_feats_list)
    
    # Ensure all feature columns present
    for c in feature_cols_all:
        if c not in df_email_urls.columns:
            df_email_urls[c] = 0
            
    # Load original full model and scaler
    model_orig = keras.models.load_model('preprocessed/dnn_model.keras')
    scaler_orig = joblib.load('preprocessed/scaler.joblib')
    
    # Score with original model (all 49 features)
    X_email_orig_sc = scaler_orig.transform(df_email_urls[feature_cols_all].values)
    probs_orig_unique = model_orig.predict(X_email_orig_sc, batch_size=1024).ravel()
    
    # Score with reduced model (47 features, no https/www)
    X_email_red_sc = scaler_red.transform(df_email_urls[feature_cols_no_https_www].values)
    probs_red_unique = model_red.predict(X_email_red_sc, batch_size=1024).ravel()
    
    # Map back to unique dict
    url2prob_orig = dict(zip(unique_url_list, probs_orig_unique))
    url2prob_red = dict(zip(unique_url_list, probs_red_unique))
    
    # Compute max URL score per email
    email_labels = []
    max_orig_scores = []
    max_red_scores = []
    
    for lbl, urls in email_url_mapping:
        if urls:
            email_labels.append(lbl)
            max_orig_scores.append(max(url2prob_orig[u] for u in urls))
            max_red_scores.append(max(url2prob_red[u] for u in urls))
            
    email_labels = np.array(email_labels)
    max_orig_scores = np.array(max_orig_scores)
    max_red_scores = np.array(max_red_scores)
    
    print(f"\n--- 3. Results on URL-Containing Email Subset (n = {len(email_labels)}) ---")
    print(f"Class breakdown: {np.sum(email_labels == 1)} Phishing, {np.sum(email_labels == 0)} Legitimate")
    
    # Original model metrics
    preds_orig = (max_orig_scores >= 0.5).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(email_labels, preds_orig).ravel()
    acc_o = accuracy_score(email_labels, preds_orig)
    auc_o = roc_auc_score(email_labels, max_orig_scores)
    prec_o = precision_score(email_labels, preds_orig)
    rec_o = recall_score(email_labels, preds_orig)
    spec_o = tn_o / (tn_o + fp_o)
    
    # Reduced model metrics
    preds_red = (max_red_scores >= 0.5).astype(int)
    tn_r, fp_r, fn_r, tp_r = confusion_matrix(email_labels, preds_red).ravel()
    acc_r = accuracy_score(email_labels, preds_red)
    auc_r = roc_auc_score(email_labels, max_red_scores)
    prec_r = precision_score(email_labels, preds_red)
    rec_r = recall_score(email_labels, preds_red)
    spec_r = tn_r / (tn_r + fp_r)
    
    print("\nORIGINAL MODEL (49 Features, with has_https & has_www):")
    print(f"  Acc: {acc_o*100:.2f}% | AUC: {auc_o:.4f} | Prec: {prec_o*100:.2f}% | Rec: {rec_o*100:.2f}% | Spec: {spec_o*100:.2f}%")
    print(f"  Confusion Matrix: TP={tp_o}, TN={tn_o}, FP={fp_o}, FN={fn_o}")
    
    print("\nREDUCED MODEL (47 Features, WITHOUT has_https & has_www):")
    print(f"  Acc: {acc_r*100:.2f}% | AUC: {auc_r:.4f} | Prec: {prec_r*100:.2f}% | Rec: {rec_r*100:.2f}% | Spec: {spec_r*100:.2f}")
    print(f"  Confusion Matrix: TP={tp_r}, TN={tn_r}, FP={fp_r}, FN={fn_r}")
    
    # Save results to npz
    np.savez('tier2_step1_https_ablation.npz',
             email_labels=email_labels,
             max_orig_scores=max_orig_scores,
             max_red_scores=max_red_scores,
             probs_orig_unique=probs_orig_unique,
             probs_red_unique=probs_red_unique)
    print("\nSaved Step 1 ablation results to tier2_step1_https_ablation.npz")

if __name__ == '__main__':
    run_experiment()


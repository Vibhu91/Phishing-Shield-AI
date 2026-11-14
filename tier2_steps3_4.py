import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

def run_step3_and_step4():
    print("=== Step 3 & Step 4: URL Calibration, Learned Fusion & Operating Points ===")
    
    # Load step 1 and step 2 datasets
    d1 = np.load('tier2_step1_https_ablation.npz')
    email_labels = d1['email_labels'] # n = 3796
    max_orig = d1['max_orig_scores']   # n = 3796
    
    d2 = np.load('tier2_step2_evasion_testbed.npz')
    p_text_evasion = d2['p_text_evasion']
    p_url_evasion = d2['p_url_evasion']
    p_text_control = d2['p_text_control']
    p_url_control = d2['p_url_control']
    p_text_pretext = d2['p_text_pretext']
    p_url_pretext = d2['p_url_pretext']
    
    # -------------------------------------------------------------
    # Step 3: Domain Adaptation & Calibration on Email-Style Links
    # -------------------------------------------------------------
    print("\n--- 3.1 Fitting Platt Scaling (Logistic Calibration) on Email-Style URLs ---")
    # Split email URL data 50/50 for calibration fit and evaluation
    np.random.seed(42)
    val_cal_idx = np.random.choice(len(email_labels), size=len(email_labels)//2, replace=False)
    test_cal_idx = np.setdiff1d(np.arange(len(email_labels)), val_cal_idx)
    
    calibrator = LogisticRegression(C=1.0)
    calibrator.fit(max_orig[val_cal_idx].reshape(-1, 1), email_labels[val_cal_idx])
    
    cal_scores_test = calibrator.predict_proba(max_orig[test_cal_idx].reshape(-1, 1))[:, 1]
    
    # Evaluate calibration on held-out email URLs
    orig_test = max_orig[test_cal_idx]
    y_test_cal = email_labels[test_cal_idx]
    
    preds_orig = (orig_test >= 0.5).astype(int)
    preds_cal = (cal_scores_test >= 0.5).astype(int)
    
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_test_cal, preds_orig).ravel()
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_test_cal, preds_cal).ravel()
    
    print(f"Uncalibrated URL Model on Email Links: Acc={accuracy_score(y_test_cal, preds_orig)*100:.2f}%, Spec={tn_o/(tn_o+fp_o)*100:.2f}%, Rec={tp_o/(tp_o+fn_o)*100:.2f}%, AUC={roc_auc_score(y_test_cal, orig_test):.4f}")
    print(f"Platt-Calibrated URL Model on Email Links: Acc={accuracy_score(y_test_cal, preds_cal)*100:.2f}%, Spec={tn_c/(tn_c+fp_c)*100:.2f}%, Rec={tp_c/(tp_c+fn_c)*100:.2f}%, AUC={roc_auc_score(y_test_cal, cal_scores_test):.4f}")
    
    # Apply calibration to the Evasion Testbed
    print("\n--- 3.2 Testing Calibrated URL Model & Learned Fusion on Step 2 Evasion Testbed ---")
    # Composite: 1,000 Evasion (Label=1) + 1,000 Control (Label=0)
    y_comp = np.concatenate([np.ones(1000), np.zeros(1000)])
    p_e_comp = np.concatenate([p_text_evasion, p_text_control])
    p_u_comp = np.concatenate([p_url_evasion, p_url_control])
    
    # Learned Logistic Stacker trained on a small development set (200 evasion + 200 control)
    np.random.seed(42)
    dev_idx = np.random.choice(1000, size=200, replace=False)
    eval_idx = np.setdiff1d(np.arange(1000), dev_idx) # 800 test samples
    
    dev_y = np.concatenate([np.ones(len(dev_idx)), np.zeros(len(dev_idx))])
    dev_X = np.column_stack([
        np.concatenate([p_text_evasion[dev_idx], p_text_control[dev_idx]]),
        np.concatenate([p_url_evasion[dev_idx], p_url_control[dev_idx]])
    ])
    
    test_y = np.concatenate([np.ones(len(eval_idx)), np.zeros(len(eval_idx))])
    test_X = np.column_stack([
        np.concatenate([p_text_evasion[eval_idx], p_text_control[eval_idx]]),
        np.concatenate([p_url_evasion[eval_idx], p_url_control[eval_idx]])
    ])
    
    stacker = LogisticRegression(C=1.0)
    stacker.fit(dev_X, dev_y)
    
    p_stack_test = stacker.predict_proba(test_X)[:, 1]
    preds_stack = (p_stack_test >= 0.5).astype(int)
    
    # Also evaluate DistilBERT, URL DNN, and Noisy-OR on the exact same test subset (N=1600)
    p_e_eval = test_X[:, 0]
    p_u_eval = test_X[:, 1]
    p_nor_eval = 1.0 - (1.0 - p_e_eval) * (1.0 - p_u_eval)
    
    for m_name, probs in [
        ("DistilBERT Alone", p_e_eval),
        ("URL DNN Alone", p_u_eval),
        ("Bayesian Noisy-OR", p_nor_eval),
        ("Learned Stacker (Trained with Evasions)", p_stack_test)
    ]:
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(test_y, preds) * 100
        rec = recall_score(test_y, preds) * 100
        spec = precision_score(test_y == 0, preds == 0) * 100
        f1 = f1_score(test_y, preds)
        auc = roc_auc_score(test_y, probs)
        print(f"  {m_name:40s} | Acc: {acc:6.2f}% | Rec: {rec:6.2f}% | Spec: {spec:6.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    print("Stacker Weights:", stacker.coef_, "Intercept:", stacker.intercept_)
    
    # -------------------------------------------------------------
    # Step 4: Expected Calibration Error (ECE) & Operating Points
    # -------------------------------------------------------------
    print("\n--- 4.1 Expected Calibration Error (ECE) Analysis ---")
    def compute_ece(y_true, y_probs, n_bins=10):
        bin_limits = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_data = []
        for i in range(n_bins):
            in_bin = (y_probs >= bin_limits[i]) & (y_probs < bin_limits[i+1])
            if np.sum(in_bin) > 0:
                bin_acc = np.mean(y_true[in_bin])
                bin_conf = np.mean(y_probs[in_bin])
                bin_prop = np.sum(in_bin) / len(y_probs)
                ece += np.abs(bin_acc - bin_conf) * bin_prop
                bin_data.append((bin_limits[i], bin_limits[i+1], bin_acc, bin_conf, np.sum(in_bin)))
        return ece, bin_data
    
    # 1. DistilBERT on 18,237 clean holdout
    d_f = np.load('scratch_fusion_182k.npz')
    y_holdout = d_f['y_test']
    p_dist_holdout = d_f['distilbert_probs']
    ece_dist, dist_bins = compute_ece(y_holdout, p_dist_holdout)
    
    # 2. URL DNN on eTLD+1 holdout (124,090)
    d_u = np.load('url_trees_etld1_preds.npz')
    y_url_holdout = d_u['y_test']
    p_dnn_holdout = d_u['probs_dnn']
    ece_dnn, dnn_bins = compute_ece(y_url_holdout, p_dnn_holdout)
    
    # 3. Noisy-OR Fusion on Evasion Benchmark
    ece_nor, nor_bins = compute_ece(test_y, p_nor_eval)
    # 4. Learned Stacker on Evasion Benchmark
    ece_stack, stack_bins = compute_ece(test_y, p_stack_test)
    
    print(f"DistilBERT ECE (18,237 Holdout Emails):        {ece_dist*100:.2f}%")
    print(f"URL DNN ECE (124,090 eTLD+1 Holdout URLs):     {ece_dnn*100:.2f}%")
    print(f"Noisy-OR Fusion ECE (Evasion Benchmark):       {ece_nor*100:.2f}%")
    print(f"Learned Stacker ECE (Evasion Benchmark):       {ece_stack*100:.2f}%")
    
    # -------------------------------------------------------------
    # Step 4: Operating Points (FPR at Fixed TPR: 95% and 99%)
    # -------------------------------------------------------------
    print("\n--- 4.2 Operating Points: False Positive Rate (FPR) at Fixed TPR ---")
    from sklearn.metrics import roc_curve
    
    def get_fpr_at_tpr(y_true, y_probs, target_tprs=[0.95, 0.99]):
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        results = {}
        for target_tpr in target_tprs:
            # find index where tpr >= target_tpr
            idx = np.where(tpr >= target_tpr)[0][0]
            results[target_tpr] = (fpr[idx], thresholds[idx])
        return results
    
    res_dist = get_fpr_at_tpr(y_holdout, p_dist_holdout)
    res_dnn = get_fpr_at_tpr(y_url_holdout, p_dnn_holdout)
    res_nor = get_fpr_at_tpr(test_y, p_nor_eval)
    res_stack = get_fpr_at_tpr(test_y, p_stack_test)
    
    print(f"{'Model / Modality':40s} | {'FPR @ 95% TPR':15s} | {'FPR @ 99% TPR':15s}")
    print(f"{'-'*40} | {'-'*15} | {'-'*15}")
    print(f"{'DistilBERT (In-Distribution Text)':40s} | {res_dist[0.95][0]*100:6.3f}% (thr={res_dist[0.95][1]:.3f}) | {res_dist[0.99][0]*100:6.3f}% (thr={res_dist[0.99][1]:.3f})")
    print(f"{'URL DNN (eTLD+1 Registered Domains)':40s} | {res_dnn[0.95][0]*100:6.3f}% (thr={res_dnn[0.95][1]:.3f}) | {res_dnn[0.99][0]*100:6.3f}% (thr={res_dnn[0.99][1]:.3f})")
    print(f"{'Bayesian Noisy-OR (Evasion Testbed)':40s} | {res_nor[0.95][0]*100:6.3f}% (thr={res_nor[0.95][1]:.3f}) | {res_nor[0.99][0]*100:6.3f}% (thr={res_nor[0.99][1]:.3f})")
    print(f"{'Learned Stacker (Evasion Testbed)':40s} | {res_stack[0.95][0]*100:6.3f}% (thr={res_stack[0.95][1]:.3f}) | {res_stack[0.99][0]*100:6.3f}% (thr={res_stack[0.99][1]:.3f})")
    
    # Save Step 3 & 4 results
    np.savez('tier2_step3_step4_results.npz',
             ece_dist=ece_dist, ece_dnn=ece_dnn, ece_nor=ece_nor, ece_stack=ece_stack,
             dist_bins=dist_bins, dnn_bins=dnn_bins,
             res_dist_95=res_dist[0.95], res_dist_99=res_dist[0.99],
             res_dnn_95=res_dnn[0.95], res_dnn_99=res_dnn[0.99],
             res_nor_95=res_nor[0.95], res_nor_99=res_nor[0.99],
             res_stack_95=res_stack[0.95], res_stack_99=res_stack[0.99])
    print("\nSaved Tier 2 Steps 3 & 4 results to tier2_step3_step4_results.npz")

if __name__ == '__main__':
    run_step3_and_step4()


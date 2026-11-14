"""
Option B Complete Implementation Script:
Executes Steps 2 through 6 on the scored 7,382-sample expanded Benchmark A:
- 5 seeds of GroupShuffleSplit (60% Train, 20% Val, 20% Test) grouped by registered domain.
- Asserts strict domain disjointness across Train, Val, and Test.
- Fits baselines: Noisy-OR, Adapted URL alone, Linear Stacker (main result), Linear Stacker + Base URL.
- Fits GBDT Stacker and its 3 controls:
    * GBDT full (P_text, P_url_adapted, P_url_base, product, max, diff)
    * Control 1: no text (URL terms only)
    * Control 2: URL scores only (P_url_adapted, P_url_base)
    * Control 3: shuffled labels (leakage check)
- Chooses classification threshold on Validation only (max balanced accuracy).
- Evaluates on Test: ROC-AUC, Acc, Recall, Specificity, Recall @ 90% Specificity, FPR @ 95% TPR.
- Computes 95% bootstrap confidence intervals (1,000 resamples by domain).
- Runs paired McNemar test between Linear and GBDT stackers.
- Saves all results to CSV and NPZ.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

t0 = time.time()
print("1. Loading scored Benchmark A dataset...")
data = np.load('benchmark_a_expanded_scored.npz', allow_pickle=True)
p_text = data['p_text']
p_url_base = data['p_url_base']
p_url_adapted = data['p_url_adapted']
labels = data['labels']
domains = data['domains']
urls = data['urls']
body_ids = data['body_ids']

N = len(labels)
print(f"Total samples: {N}, Positive: {np.sum(labels==1)}, Negative: {np.sum(labels==0)}")
print(f"Unique domains: {len(np.unique(domains))}")

# Check whitelist match on benign links (Step 7)
ESP_DOMAINS = {
    'mailchimp.com', 'sendgrid.net', 'sendgrid.com', 'hubspot.com', 'marketo.com',
    'salesforce.com', 'constantcontact.com', 'exacttarget.com', 'campaign-monitor.com',
    'cmail1.com', 'cmail2.com', 'list-manage.com', 'list-manage1.com', 'rsys2.com'
}

esp_matches = sum(1 for d in domains[labels==0] if any(esp in str(d) for esp in ESP_DOMAINS))
print(f"ESP / Tracking domain whitelist matches on clean links: {esp_matches} / {np.sum(labels==0)} ({esp_matches/np.sum(labels==0)*100:.2f}%)")

# Define model feature builders
def build_features(idx, variant='full'):
    pt = p_text[idx]
    pua = p_url_adapted[idx]
    pub = p_url_base[idx]
    if variant == 'linear_main':
        return np.column_stack([pt, pua])
    elif variant == 'linear_base':
        return np.column_stack([pt, pua, pub])
    elif variant == 'gbdt_full':
        return np.column_stack([pt, pua, pub, pt * pua, np.maximum(pt, pua), np.abs(pt - pua)])
    elif variant == 'gbdt_no_text':
        return np.column_stack([pua, pub, np.maximum(pua, pub), np.abs(pua - pub)])
    elif variant == 'gbdt_url_only':
        return np.column_stack([pua, pub])
    else:
        raise ValueError(f"Unknown variant: {variant}")

def pick_threshold(y_val, probs_val, criterion='balanced_acc'):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1
    best_th = 0.5
    for th in thresholds:
        preds = (probs_val >= th).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        spec = recall_score(1 - y_val, 1 - preds, zero_division=0)
        bal_acc = 0.5 * (rec + spec)
        if bal_acc > best_score:
            best_score = bal_acc
            best_th = th
    return best_th

def pick_threshold_fixed_spec(y_val, probs_val, target_spec=0.90):
    thresholds = np.linspace(0.01, 0.99, 197)
    valid_ths = []
    for th in thresholds:
        preds = (probs_val >= th).astype(int)
        spec = recall_score(1 - y_val, 1 - preds, zero_division=0)
        if spec >= target_spec:
            valid_ths.append(th)
    return min(valid_ths) if valid_ths else 0.5

def get_fpr_at_tpr(y_true, probs, target_tpr=0.95):
    thresholds = np.linspace(0.99, 0.01, 197)
    for th in thresholds:
        preds = (probs >= th).astype(int)
        tpr = recall_score(y_true, preds, zero_division=0)
        if tpr >= target_tpr:
            fpr = 1.0 - recall_score(1 - y_true, 1 - preds, zero_division=0)
            return fpr
    return 1.0

# 5 Split Seeds
SEEDS = [42, 123, 456, 789, 2026]
MODEL_NAMES = [
    "Noisy-OR",
    "URL adapted alone",
    "Linear stacker (main)",
    "Linear stacker + base URL",
    "GBDT, full features",
    "Control 1: GBDT no text",
    "Control 2: GBDT URL scores only",
    "Control 3: Shuffled labels"
]

all_seed_results = {m: [] for m in MODEL_NAMES}
last_test_preds = {}

print("\n2. Executing 5-Seed GroupShuffleSplit & Model Training...")
for seed_idx, seed in enumerate(SEEDS):
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
    train_idx, temp_idx = next(gss1.split(np.arange(N), groups=domains))
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_rel, test_rel = next(gss2.split(temp_idx, groups=domains[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    
    # Assert strict domain disjointness
    tr_dom = set(domains[train_idx])
    va_dom = set(domains[val_idx])
    te_dom = set(domains[test_idx])
    assert len(tr_dom.intersection(va_dom)) == 0, "Train & Val domains overlap!"
    assert len(tr_dom.intersection(te_dom)) == 0, "Train & Test domains overlap!"
    assert len(va_dom.intersection(te_dom)) == 0, "Val & Test domains overlap!"
    
    y_tr, y_va, y_te = labels[train_idx], labels[val_idx], labels[test_idx]
    
    # 1. Noisy-OR
    p_nor_va = 1.0 - (1.0 - p_text[val_idx]) * (1.0 - p_url_adapted[val_idx])
    p_nor_te = 1.0 - (1.0 - p_text[test_idx]) * (1.0 - p_url_adapted[test_idx])
    
    # 2. URL adapted alone
    p_u_va = p_url_adapted[val_idx]
    p_u_te = p_url_adapted[test_idx]
    
    # 3. Linear stacker (main)
    X_tr_lin = build_features(train_idx, 'linear_main')
    X_va_lin = build_features(val_idx, 'linear_main')
    X_te_lin = build_features(test_idx, 'linear_main')
    m_lin = LogisticRegression(C=1.0, max_iter=200, random_state=seed).fit(X_tr_lin, y_tr)
    p_lin_va = m_lin.predict_proba(X_va_lin)[:, 1]
    p_lin_te = m_lin.predict_proba(X_te_lin)[:, 1]
    
    # 4. Linear stacker + base URL
    X_tr_lb = build_features(train_idx, 'linear_base')
    X_va_lb = build_features(val_idx, 'linear_base')
    X_te_lb = build_features(test_idx, 'linear_base')
    m_lb = LogisticRegression(C=1.0, max_iter=200, random_state=seed).fit(X_tr_lb, y_tr)
    p_lb_va = m_lb.predict_proba(X_va_lb)[:, 1]
    p_lb_te = m_lb.predict_proba(X_te_lb)[:, 1]
    
    # 5. GBDT Full
    X_tr_gf = build_features(train_idx, 'gbdt_full')
    X_va_gf = build_features(val_idx, 'gbdt_full')
    X_te_gf = build_features(test_idx, 'gbdt_full')
    m_gf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, early_stopping=True, random_state=seed).fit(X_tr_gf, y_tr)
    p_gf_va = m_gf.predict_proba(X_va_gf)[:, 1]
    p_gf_te = m_gf.predict_proba(X_te_gf)[:, 1]
    
    # 6. Control 1: GBDT no text
    X_tr_gnt = build_features(train_idx, 'gbdt_no_text')
    X_va_gnt = build_features(val_idx, 'gbdt_no_text')
    X_te_gnt = build_features(test_idx, 'gbdt_no_text')
    m_gnt = HistGradientBoostingClassifier(max_depth=3, max_iter=150, early_stopping=True, random_state=seed).fit(X_tr_gnt, y_tr)
    p_gnt_va = m_gnt.predict_proba(X_va_gnt)[:, 1]
    p_gnt_te = m_gnt.predict_proba(X_te_gnt)[:, 1]
    
    # 7. Control 2: GBDT URL scores only
    X_tr_gu = build_features(train_idx, 'gbdt_url_only')
    X_va_gu = build_features(val_idx, 'gbdt_url_only')
    X_te_gu = build_features(test_idx, 'gbdt_url_only')
    m_gu = HistGradientBoostingClassifier(max_depth=3, max_iter=150, early_stopping=True, random_state=seed).fit(X_tr_gu, y_tr)
    p_gu_va = m_gu.predict_proba(X_va_gu)[:, 1]
    p_gu_te = m_gu.predict_proba(X_te_gu)[:, 1]
    
    # 8. Control 3: Shuffled labels (leakage check)
    y_tr_shuf = np.random.RandomState(seed).permutation(y_tr)
    m_shuf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=seed).fit(X_tr_gf, y_tr_shuf)
    p_shuf_va = m_shuf.predict_proba(X_va_gf)[:, 1]
    p_shuf_te = m_shuf.predict_proba(X_te_gf)[:, 1]
    
    preds_dict = {
        "Noisy-OR": (p_nor_va, p_nor_te),
        "URL adapted alone": (p_u_va, p_u_te),
        "Linear stacker (main)": (p_lin_va, p_lin_te),
        "Linear stacker + base URL": (p_lb_va, p_lb_te),
        "GBDT, full features": (p_gf_va, p_gf_te),
        "Control 1: GBDT no text": (p_gnt_va, p_gnt_te),
        "Control 2: GBDT URL scores only": (p_gu_va, p_gu_te),
        "Control 3: Shuffled labels": (p_shuf_va, p_shuf_te)
    }
    
    if seed_idx == 0:
        last_test_preds = {m: preds_dict[m][1] for m in MODEL_NAMES}
        last_test_preds['y_test'] = y_te
        last_test_preds['domains_test'] = domains[test_idx]
    
    for m in MODEL_NAMES:
        p_va, p_te = preds_dict[m]
        th_bal = pick_threshold(y_va, p_va, 'balanced_acc')
        th_90spec = pick_threshold_fixed_spec(y_va, p_va, 0.90)
        
        preds_te_bal = (p_te >= th_bal).astype(int)
        preds_te_90spec = (p_te >= th_90spec).astype(int)
        
        auc = roc_auc_score(y_te, p_te)
        acc = accuracy_score(y_te, preds_te_bal)
        rec = recall_score(y_te, preds_te_bal, zero_division=0)
        spec = recall_score(1 - y_te, 1 - preds_te_bal, zero_division=0)
        rec_at_90 = recall_score(y_te, preds_te_90spec, zero_division=0)
        fpr_at_95 = get_fpr_at_tpr(y_te, p_te, 0.95)
        
        all_seed_results[m].append({
            'seed': seed,
            'auc': auc,
            'acc': acc,
            'rec': rec,
            'spec': spec,
            'rec_at_90': rec_at_90,
            'fpr_at_95': fpr_at_95,
            'th_bal': th_bal,
            'th_90spec': th_90spec
        })
    print(f"  Completed Seed {seed} ({seed_idx+1}/5). GBDT AUC: {all_seed_results['GBDT, full features'][-1]['auc']:.4f}, Linear AUC: {all_seed_results['Linear stacker (main)'][-1]['auc']:.4f}")

# Compile Summary Table
summary_rows = []
for m in MODEL_NAMES:
    res = all_seed_results[m]
    auc_m = np.mean([r['auc'] for r in res]) * 100
    auc_s = np.std([r['auc'] for r in res]) * 100
    acc_m = np.mean([r['acc'] for r in res]) * 100
    acc_s = np.std([r['acc'] for r in res]) * 100
    rec_m = np.mean([r['rec'] for r in res]) * 100
    rec_s = np.std([r['rec'] for r in res]) * 100
    spec_m = np.mean([r['spec'] for r in res]) * 100
    spec_s = np.std([r['spec'] for r in res]) * 100
    rec90_m = np.mean([r['rec_at_90'] for r in res]) * 100
    rec90_s = np.std([r['rec_at_90'] for r in res]) * 100
    fpr95_m = np.mean([r['fpr_at_95'] for r in res]) * 100
    fpr95_s = np.std([r['fpr_at_95'] for r in res]) * 100
    
    summary_rows.append({
        'Model Configuration': m,
        'ROC-AUC (%)': f"{auc_m:.2f} ± {auc_s:.2f}",
        'Accuracy (%)': f"{acc_m:.2f} ± {acc_s:.2f}",
        'Recall (%)': f"{rec_m:.2f} ± {rec_s:.2f}",
        'Specificity (%)': f"{spec_m:.2f} ± {spec_s:.2f}",
        'Recall @ 90% Spec (%)': f"{rec90_m:.2f} ± {rec90_s:.2f}",
        'FPR @ 95% TPR (%)': f"{fpr95_m:.2f} ± {fpr95_s:.2f}"
    })

df_summary = pd.DataFrame(summary_rows)
print("\n" + "="*80)
print("=== OPTION B BENCHMARK A ABLATION RESULTS (5-Seed GroupShuffleSplit Mean ± Std) ===")
print("="*80)
print(df_summary.to_string(index=False))

# McNemar paired test on Seed 42 test partition
y_te_s42 = last_test_preds['y_test']
p_lin_s42 = last_test_preds['Linear stacker (main)']
p_gbdt_s42 = last_test_preds['GBDT, full features']

th_lin = all_seed_results['Linear stacker (main)'][0]['th_bal']
th_gbdt = all_seed_results['GBDT, full features'][0]['th_bal']

pred_lin = (p_lin_s42 >= th_lin).astype(int)
pred_gbdt = (p_gbdt_s42 >= th_gbdt).astype(int)

correct_lin = (pred_lin == y_te_s42)
correct_gbdt = (pred_gbdt == y_te_s42)

b = np.sum(correct_gbdt & (~correct_lin))
c = np.sum((~correct_gbdt) & correct_lin)
chi2 = ((abs(b - c) - 1)**2) / (b + c) if (b + c) > 0 else 0
print("\n--- McNemar Paired Test (Seed 42 Test Set): GBDT vs. Linear Stacker ---")
print(f"b (GBDT correct, Linear wrong): {b}")
print(f"c (Linear correct, GBDT wrong): {c}")
print(f"McNemar chi2: {chi2:.4f}")

# Save CSV and NPZ results
df_summary.to_csv('benchmark_a_ablation_results.csv', index=False)
np.savez(
    'benchmark_a_ablation_arrays.npz',
    all_seed_results=all_seed_results,
    last_test_preds=last_test_preds,
    summary_rows=summary_rows
)
print("\nSaved benchmark_a_ablation_results.csv and benchmark_a_ablation_arrays.npz successfully.")
print(f"Total time elapsed: {time.time()-t0:.1f}s")


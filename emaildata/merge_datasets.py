"""
Merge additional email datasets into combined_phishing_email_dataset.csv.

Sources integrated:
1. Original dataset (83,948 emails)
2. Kaggle Phishing Email dataset (kaggle_phishing_email.csv)
3. Zionia Phishing Benchmark Corpus (zionia_train.parquet, zionia_test.parquet)
4. DarkKnight25 Targeted Phishing dataset (darkknight25_phishing.jsonl)

Pipeline:
1. Create backup of combined_phishing_email_dataset.csv
2. Standardize columns to [id, subject, body, label, source, sender, receiver, date, urls]
3. Deduplicate across all sets
4. Save merged CSV
"""
import os
import re
import json
import shutil
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\Vibhu-Kavi\Downloads\PhishingDeeplearning\emaildata"
COMBINED_CSV = os.path.join(DATA_DIR, "combined_phishing_email_dataset.csv")
BACKUP_CSV = os.path.join(DATA_DIR, "combined_phishing_email_dataset_backup.csv")

RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)

def normalize_text(t):
    if not isinstance(t, str):
        return ""
    return " ".join(t.lower().split())

def make_dedup_key(subject, body):
    s_norm = normalize_text(subject)
    b_norm = normalize_text(body)
    return f"{s_norm}|||{b_norm[:500]}"

def extract_subject_body(text):
    if not isinstance(text, str):
        return "", ""
    text = text.strip()
    m = re.match(r"^(?:re\s*:\s*|subject\s*:\s*)([^\r\n]+)(?:[\r\n]+([\s\S]*))?$", text, re.I)
    if m:
        subj = m.group(1).strip()
        body = (m.group(2) or "").strip()
        if not body:
            body = text
        return subj, body
    lines = text.split("\n", 1)
    if len(lines) == 2 and len(lines[0]) < 120 and any(lines[0].lower().startswith(p) for p in ["re:", "fw:", "fwd:", "re :"]):
        return lines[0].strip(), lines[1].strip()
    return "", text

def main():
    print("=" * 70)
    print("STEP 1: Backing up existing dataset...")
    print("=" * 70)
    if not os.path.exists(BACKUP_CSV):
        shutil.copyfile(COMBINED_CSV, BACKUP_CSV)
        print(f"  Backup created: {BACKUP_CSV} ({os.path.getsize(BACKUP_CSV):,} bytes)")
    else:
        print(f"  Backup already exists: {BACKUP_CSV}")

    print("\n" + "=" * 70)
    print("STEP 2: Loading existing dataset...")
    print("=" * 70)
    df_exist = pd.read_csv(COMBINED_CSV)
    print(f"  Loaded existing dataset: {len(df_exist):,} rows")

    seen_keys = set()
    for s, b in zip(df_exist["subject"].fillna(""), df_exist["body"].fillna("")):
        seen_keys.add(make_dedup_key(s, b))
    print(f"  Unique existing dedup signatures: {len(seen_keys):,}")

    new_rows = []

    # ---------------------------------------------------------
    # Source 1: Kaggle Phishing Email
    # ---------------------------------------------------------
    kaggle_csv = os.path.join(DATA_DIR, "kaggle_phishing_email.csv")
    if os.path.exists(kaggle_csv):
        print("\n" + "=" * 70)
        print("STEP 3: Processing Kaggle Phishing Email dataset...")
        print("=" * 70)
        df_k = pd.read_csv(kaggle_csv)
        label_map = {"Safe Email": 0, "Phishing Email": 1}
        k_count = 0
        for _, row in df_k.iterrows():
            raw_text = str(row["Email Text"]) if pd.notna(row["Email Text"]) else ""
            if not raw_text.strip() or len(raw_text.strip()) < 10:
                continue
            lbl = label_map.get(row.get("Email Type"))
            if lbl is None:
                continue
            
            subj, body = extract_subject_body(raw_text)
            key = make_dedup_key(subj, body)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            has_url = 1 if RE_URL.search(raw_text) else 0
            new_rows.append({
                "subject": subj,
                "body": body,
                "label": lbl,
                "source": "Kaggle_Phishing_Email",
                "sender": np.nan,
                "receiver": np.nan,
                "date": np.nan,
                "urls": has_url
            })
            k_count += 1
        print(f"  Added from Kaggle: {k_count:,} unique emails")

    # ---------------------------------------------------------
    # Source 2: DarkKnight25 Phishing Dataset
    # ---------------------------------------------------------
    dk_path = os.path.join(DATA_DIR, "darkknight25_phishing.jsonl")
    if os.path.exists(dk_path):
        print("\n" + "=" * 70)
        print("STEP 4: Processing DarkKnight25 Targeted Phishing dataset...")
        print("=" * 70)
        dk_count = 0
        with open(dk_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                s = item.get("subject", "").strip()
                b = item.get("body", "").strip()
                if not b and not s:
                    continue
                lbl = 1 if item.get("label") == "phishing" else 0
                key = make_dedup_key(s, b)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                has_url = 1 if RE_URL.search(f"{s} {b}") else 0
                new_rows.append({
                    "subject": s,
                    "body": b,
                    "label": lbl,
                    "source": "DarkKnight25_Phishing",
                    "sender": item.get("spoofed_sender", np.nan),
                    "receiver": np.nan,
                    "date": np.nan,
                    "urls": has_url
                })
                dk_count += 1
        print(f"  Added from DarkKnight25: {dk_count:,} unique emails")

    # ---------------------------------------------------------
    # Source 3: Zionia Benchmark Corpus (train + test)
    # ---------------------------------------------------------
    z_train = os.path.join(DATA_DIR, "zionia_train.parquet")
    z_test = os.path.join(DATA_DIR, "zionia_test.parquet")
    if os.path.exists(z_train):
        print("\n" + "=" * 70)
        print("STEP 5: Processing Zionia Phishing Benchmark Corpus...")
        print("=" * 70)
        z_dfs = [pd.read_parquet(z_train)]
        if os.path.exists(z_test):
            z_dfs.append(pd.read_parquet(z_test))
        df_z = pd.concat(z_dfs, ignore_index=True)
        z_count = 0
        for _, row in df_z.iterrows():
            raw_text = str(row["text_combined"]) if pd.notna(row["text_combined"]) else ""
            if not raw_text.strip() or len(raw_text.strip()) < 10:
                continue
            lbl = int(row["label"])
            subj, body = extract_subject_body(raw_text)
            key = make_dedup_key(subj, body)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            has_url = 1 if RE_URL.search(raw_text) else 0
            new_rows.append({
                "subject": subj,
                "body": body,
                "label": lbl,
                "source": "Zionia_Phishing",
                "sender": np.nan,
                "receiver": np.nan,
                "date": np.nan,
                "urls": has_url
            })
            z_count += 1
        print(f"  Added from Zionia: {z_count:,} unique emails")

    # ---------------------------------------------------------
    # Merge and Save
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Merging and formatting...")
    print("=" * 70)
    df_new = pd.DataFrame(new_rows)
    print(f"  Total newly added rows: {len(df_new):,}")

    cols = ["subject", "body", "label", "source", "sender", "receiver", "date", "urls"]
    df_exist_subset = df_exist[[c for c in cols if c in df_exist.columns]]
    df_final = pd.concat([df_exist_subset, df_new[cols]], ignore_index=True)

    # Re-assign clean 1-indexed sequential IDs
    df_final.insert(0, "id", range(1, len(df_final) + 1))

    print(f"  Final merged dataset rows: {len(df_final):,}")
    print("\nBreakdown by Source and Label:")
    breakdown = pd.crosstab(df_final["source"], df_final["label"], margins=True, margins_name="Total")
    breakdown.columns = ["Legitimate (0)", "Phishing (1)", "Total"]
    print(breakdown.to_string())

    print("\n" + "=" * 70)
    print("STEP 7: Writing to combined_phishing_email_dataset.csv...")
    print("=" * 70)
    df_final.to_csv(COMBINED_CSV, index=False)
    file_size_mb = os.path.getsize(COMBINED_CSV) / (1024 * 1024)
    print(f"  SUCCESS! Saved {len(df_final):,} rows to {COMBINED_CSV} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()

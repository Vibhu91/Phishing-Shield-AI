import pandas as pd
import numpy as np
import re
import tldextract
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv('emaildata/combined_phishing_email_dataset.csv', low_memory=False)
labels = df['label'].astype(int).values

train_idx, temp_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Check candidate bodies in test + val holdouts
for name, idxs in [("Test", test_idx), ("Val", val_idx)]:
    sub_df = df.iloc[idxs]
    valid_sources = ['Enron', 'CEAS_08', 'Ling', 'SpamAssassin']
    cand_bodies = sub_df[(sub_df['label'] == 0) & (sub_df['source'].isin(valid_sources))].copy()
    cand_bodies['word_count'] = cand_bodies['body'].fillna('').apply(lambda x: len(str(x).split()))
    cand_bodies = cand_bodies[cand_bodies['word_count'] >= 30]
    print(f"[{name}] Candidate clean bodies (>= 30 words): {len(cand_bodies)}")

URL_REGEX = re.compile(
    r'(?:https?://|www\d{0,3}[.]|[a-zA-Z0-9.\-]+[.][a-zA-Z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:\'\".,<>?])',
    re.IGNORECASE
)

# Extract URLs from phishing vs clean emails across full df
print("\nExtracting URLs...")
urls_clean = []
urls_phish = []

for idx, r in df.iterrows():
    text = str(r['subject'] or '') + ' ' + str(r['body'] or '')
    found = [m.group(0).strip().rstrip('.,;:!?)>"\'') for m in URL_REGEX.finditer(text)]
    valid = [u for u in found if len(u) > 7 and '.' in u]
    if not valid:
        continue
    lbl = int(r['label'])
    src = str(r['source'])
    for u in valid:
        if lbl == 0 and src in ['Enron', 'CEAS_08', 'Ling', 'SpamAssassin']:
            urls_clean.append((idx, src, u))
        elif lbl == 1 and src in ['Nazario', 'Nigerian_Fraud', 'Zionia_Phishing', 'Kaggle_Phishing_Email']:
            urls_phish.append((idx, src, u))

print(f"Total clean URL instances extracted: {len(urls_clean)}")
print(f"Total phishing URL instances extracted: {len(urls_phish)}")

# Deduplicate by URL
df_clean_urls = pd.DataFrame(urls_clean, columns=['email_id', 'source', 'url']).drop_duplicates(subset=['url'])
df_phish_urls = pd.DataFrame(urls_phish, columns=['email_id', 'source', 'url']).drop_duplicates(subset=['url'])

def get_domain(u):
    ext = tldextract.extract(u)
    return f"{ext.domain}.{ext.suffix}".lower()

df_clean_urls['domain'] = df_clean_urls['url'].apply(get_domain)
df_phish_urls['domain'] = df_phish_urls['url'].apply(get_domain)

print(f"Unique clean URLs: {len(df_clean_urls)} across {df_clean_urls['domain'].nunique()} domains")
print(f"Unique phish URLs: {len(df_phish_urls)} across {df_phish_urls['domain'].nunique()} domains")


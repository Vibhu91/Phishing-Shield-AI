"""
Fast Candidate Extraction Script for Benchmark A Expansion (Pickle format)
"""
import re
import time
import tldextract
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

t0 = time.time()
print("1. Loading dataset...")
df = pd.read_csv('emaildata/combined_phishing_email_dataset.csv', low_memory=False)
labels = df['label'].astype(int).values
print(f"Total dataset rows: {len(df)} in {time.time()-t0:.1f}s")

# Recreate the exact stratified split
print("2. Re-creating holdout splits (80% train, 10% val, 10% test)...")
train_idx, temp_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

holdout_idxs = np.concatenate([val_idx, test_idx])
print(f"Train size: {len(train_idx)}, Holdout size (Val+Test): {len(holdout_idxs)}")

URL_REGEX = re.compile(
    r'https?://[^\s()<>]+|www\.[^\s()<>]+\.[a-zA-Z]{2,4}[^\s()<>]*',
    re.IGNORECASE
)

# Extract clean bodies from holdout only (Enron, CEAS_08, Ling, SpamAssassin)
print("3. Filtering clean candidate bodies from holdout...")
valid_clean_sources = {'Enron', 'CEAS_08', 'Ling', 'SpamAssassin'}
holdout_df = df.iloc[holdout_idxs].copy()

cand_bodies = []
for idx, r in holdout_df.iterrows():
    if r['label'] != 0 or r['source'] not in valid_clean_sources:
        continue
    body = str(r['body'] or '').strip()
    subj = str(r['subject'] or '').strip()
    if not body:
        continue
    # Strip existing URLs
    cleaned_body = URL_REGEX.sub('', body).strip()
    # Check word count
    words = cleaned_body.split()
    if len(words) >= 30:
        cand_bodies.append({
            'body_id': idx,
            'source': r['source'],
            'subject': subj,
            'body': cleaned_body,
            'word_count': len(words)
        })

df_cand_bodies = pd.DataFrame(cand_bodies)
print(f"Extracted {len(df_cand_bodies)} clean candidate bodies (>= 30 words) from holdout.")
print(df_cand_bodies['source'].value_counts())

# Extract URLs from full corpus
print("4. Extracting real URLs from phishing and clean emails...")
urls_attack = []
urls_clean = []

# Include all phishing email sources that have genuine phishing links
valid_phish_sources = {'Nazario', 'Nigerian_Fraud', 'Zionia_Phishing', 'Kaggle_Phishing_Email', 'CEAS_08', 'Enron', 'SpamAssassin'}

for idx, r in df.iterrows():
    body = str(r['body'] or '')
    subj = str(r['subject'] or '')
    text = subj + ' ' + body
    if 'http' not in text and 'www.' not in text:
        continue
    found = URL_REGEX.findall(text)
    if not found:
        continue
    lbl = int(r['label'])
    src = str(r['source'])
    for u in found:
        u_clean = u.strip().rstrip('.,;:!?)>"\'')
        if len(u_clean) > 8 and '.' in u_clean:
            if lbl == 1 and src in valid_phish_sources:
                urls_attack.append({'email_id': idx, 'source': src, 'url': u_clean})
            elif lbl == 0 and src in valid_clean_sources:
                urls_clean.append({'email_id': idx, 'source': src, 'url': u_clean})

df_u_attack = pd.DataFrame(urls_attack).drop_duplicates(subset=['url'])
df_u_clean = pd.DataFrame(urls_clean).drop_duplicates(subset=['url'])

print(f"Extracted unique attack URLs: {len(df_u_attack)}")
print(f"Extracted unique clean URLs: {len(df_u_clean)}")

def get_etld1(u):
    try:
        ext = tldextract.extract(u)
        d = f"{ext.domain}.{ext.suffix}".lower()
        return d if ext.domain and ext.suffix else "unknown"
    except Exception:
        return "unknown"

df_u_attack['domain'] = df_u_attack['url'].apply(get_etld1)
df_u_clean['domain'] = df_u_clean['url'].apply(get_etld1)

df_u_attack = df_u_attack[df_u_attack['domain'] != 'unknown']
df_u_clean = df_u_clean[df_u_clean['domain'] != 'unknown']

print(f"Attack URLs: {len(df_u_attack)} across {df_u_attack['domain'].nunique()} registered domains")
print(f"Clean URLs:  {len(df_u_clean)} across {df_u_clean['domain'].nunique()} registered domains")

# Save as pickle
df_cand_bodies.to_pickle('cand_bodies_holdout.pkl')
df_u_attack.to_pickle('cand_attack_urls.pkl')
df_u_clean.to_pickle('cand_clean_urls.pkl')

print("Saved caches successfully to pkl files.")
print(f"Total time elapsed: {time.time()-t0:.1f}s")

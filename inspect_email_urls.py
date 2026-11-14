import re
import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit, train_test_split

URL_REGEX = re.compile(
    r'(?:https?://|www\d{0,3}[.]|[a-zA-Z0-9.\-]+[.][a-zA-Z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])',
    re.IGNORECASE
)

print("Loading 182k email dataset...")
df = pd.read_csv('emaildata/combined_phishing_email_dataset.csv', low_memory=False)
labels = df['label'].astype(int).values

# Get the train (80%) vs holdout test (10%) splits
train_idx, temp_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

print(f"Train emails: {len(train_idx)}, Val emails: {len(val_idx)}, Test emails: {len(test_idx)}")

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

def extract_urls_from_df(sub_df, name="split"):
    records = []
    for idx, r in sub_df.iterrows():
        text = str(r['subject'] or '') + ' ' + str(r['body'] or '')
        found = [m.group(0).strip().rstrip('.,;:!?)>"\'') for m in URL_REGEX.finditer(text)]
        valid = [u for u in found if len(u) > 6 and '.' in u]
        for u in valid:
            records.append({'email_id': idx, 'source': r['source'], 'label': r['label'], 'url': u})
    df_res = pd.DataFrame(records)
    print(f"[{name}] Extracted {len(df_res)} URL instances ({df_res['url'].nunique() if len(df_res)>0 else 0} unique).")
    if len(df_res) > 0:
        print(f"[{name}] Phishing URLs: {sum(df_res['label']==1)}, Legit URLs: {sum(df_res['label']==0)}")
    return df_res

train_urls_df = extract_urls_from_df(train_df, "Train")
test_urls_df = extract_urls_from_df(test_df, "Test")

# Deduplicate URLs by URL string per split while keeping label
train_urls_unique = train_urls_df.drop_duplicates(subset=['url']).copy()
test_urls_unique = test_urls_df.drop_duplicates(subset=['url']).copy()

# Add registered domain (eTLD+1)
def get_domain(u):
    ext = tldextract.extract(u)
    return f"{ext.domain}.{ext.suffix}".lower()

train_urls_unique['domain'] = train_urls_unique['url'].apply(get_domain)
test_urls_unique['domain'] = test_urls_unique['url'].apply(get_domain)

print(f"Unique Train URLs: {len(train_urls_unique)} across {train_urls_unique['domain'].nunique()} domains")
print(f"Unique Test URLs: {len(test_urls_unique)} across {test_urls_unique['domain'].nunique()} domains")

# Check domain overlap across train and test email URLs
train_domains = set(train_urls_unique['domain'])
test_domains = set(test_urls_unique['domain'])
overlap = train_domains.intersection(test_domains)
print(f"Domain overlap between train and test email URLs: {len(overlap)}")


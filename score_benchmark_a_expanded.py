"""
Extract 49 Features and Score Benchmark A Manifest
Outputs:
- p_text (DistilBERT on the full synthesized email)
- p_url_base (Standard URL DNN on URL)
- p_url_adapted (Email-adapted URL DNN on URL)
- label, domain, body_id
Saves to 'benchmark_a_expanded_scored.npz'
"""
import time
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow import keras
from feature_extraction import build_feature_dataframe

t0 = time.time()
print("1. Loading Benchmark A manifest...")
df = pd.read_pickle('benchmark_a_expanded_manifest.pkl')
print(f"Total samples: {len(df)}")

# 1. DistilBERT text scoring
print("\n2. Scoring full synthesized email texts with DistilBERT...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading DistilBERT on {device}...")
tokenizer = AutoTokenizer.from_pretrained('emaildata/results/best_distilbert')
model_dist = AutoModelForSequenceClassification.from_pretrained('emaildata/results/best_distilbert').to(device)
model_dist.eval()

texts = df['full_text'].tolist()
p_text_list = []
batch_size = 64

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_token_type_ids=False, return_tensors='pt').to(device)
        logits = model_dist(**inputs).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        p_text_list.extend(probs)
        if (i // batch_size) % 25 == 0:
            print(f"  Scored {min(i+batch_size, len(texts))}/{len(texts)} texts...")

p_text = np.array(p_text_list, dtype=np.float32)
print(f"DistilBERT scoring complete. Mean P_text = {p_text.mean():.4f}")

# 2. Extract 49 URL features
print("\n3. Extracting 49 features for each URL...")
urls = df['url'].tolist()
df_urls = pd.DataFrame({'url': urls})
t_feat = time.time()
df_feats = build_feature_dataframe(df_urls, 'url')
drop_cols = [c for c in ['url', 'label', 'source'] if c in df_feats.columns]
if drop_cols:
    df_feats = df_feats.drop(columns=drop_cols)

X_raw = df_feats.values.astype(np.float32)
print(f"Feature extraction complete in {time.time()-t_feat:.1f}s. Matrix shape: {X_raw.shape}")

# 3. Standard URL DNN scoring
print("\n4. Scoring with Standard URL DNN...")
scaler_base = joblib.load('preprocessed/scaler.joblib')
model_base = keras.models.load_model('preprocessed/dnn_model.keras')

X_scaled_base = scaler_base.transform(X_raw)
p_url_base = model_base.predict(X_scaled_base, batch_size=256, verbose=0).flatten().astype(np.float32)
print(f"Standard URL DNN scoring complete. Mean P_url_base = {p_url_base.mean():.4f}")

# 4. Email-Adapted URL DNN scoring
print("\n5. Scoring with Email-Adapted URL DNN...")
scaler_adapted = joblib.load('preprocessed/scaler_email_adapted.joblib')
model_adapted = keras.models.load_model('preprocessed/dnn_email_adapted.keras')

X_scaled_adapted = scaler_adapted.transform(X_raw)
p_url_adapted = model_adapted.predict(X_scaled_adapted, batch_size=256, verbose=0).flatten().astype(np.float32)
print(f"Email-Adapted URL DNN scoring complete. Mean P_url_adapted = {p_url_adapted.mean():.4f}")

# 5. Save all scored arrays
labels = df['label'].values.astype(np.int32)
domains = df['domain'].values
body_ids = df['body_id'].values.astype(np.int32)
urls_arr = np.array(urls)

np.savez(
    'benchmark_a_expanded_scored.npz',
    p_text=p_text,
    p_url_base=p_url_base,
    p_url_adapted=p_url_adapted,
    labels=labels,
    domains=domains,
    body_ids=body_ids,
    urls=urls_arr
)

print(f"\nSaved all scored arrays to benchmark_a_expanded_scored.npz in {time.time()-t0:.1f}s")


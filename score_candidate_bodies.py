"""
Score Candidate Clean Bodies with DistilBERT on GPU
Ensures P_text < 0.10 for all candidate bodies so text does not leak the label.
"""
import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

t0 = time.time()
print("1. Loading candidate bodies...")
df_bodies = pd.read_pickle('cand_bodies_holdout.pkl')
print(f"Total candidate bodies to score: {len(df_bodies)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading DistilBERT checkpoint on {device}...")
tokenizer = AutoTokenizer.from_pretrained('emaildata/results/best_distilbert')
model = AutoModelForSequenceClassification.from_pretrained('emaildata/results/best_distilbert').to(device)
model.eval()

texts = [str(r['subject'] or '') + " " + str(r['body'] or '') for _, r in df_bodies.iterrows()]

batch_size = 64
probs_list = []

print("Scoring in batches...")
with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_token_type_ids=False, return_tensors='pt').to(device)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs_list.extend(probs)
        if (i // batch_size) % 20 == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} bodies...")

df_bodies['p_text'] = probs_list
print(f"Scoring complete in {time.time()-t0:.1f}s.")
print(f"P_text summary:")
print(df_bodies['p_text'].describe())

df_clean_strict = df_bodies[df_bodies['p_text'] < 0.10].copy()
print(f"Bodies passing strict P_text < 0.10 filter: {len(df_clean_strict)} (out of {len(df_bodies)})")

df_clean_strict.to_pickle('cand_bodies_scored_clean.pkl')
print("Saved to cand_bodies_scored_clean.pkl")


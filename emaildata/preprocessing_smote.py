"""
Preprocessing + SMOTE balancing for the combined phishing email dataset.

Pipeline
--------
1. Load CSV, drop exact duplicates / empty emails
2. Build one text field (subject + body) and clean it
3. Add a few simple numeric features (URL count, length, ...)
4. Stratified train / test split          <-- BEFORE SMOTE (avoids data leakage)
5. TF-IDF fitted on the TRAIN set only
6. SMOTE applied to the TRAIN set only (test set stays untouched / real)
7. Save everything to ./output

Usage
-----
    python preprocessing_smote.py
    python preprocessing_smote.py --input path/to/file.csv --max-features 30000 --k 5

Requires: pandas, numpy, scipy, scikit-learn, joblib
Optional: imbalanced-learn (used automatically if installed, otherwise a built-in
          SMOTE implementation for sparse TF-IDF matrices is used)
"""
import argparse
import html
import os
import re

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

# --------------------------------------------------------------------------- #
# 1. Text cleaning
# --------------------------------------------------------------------------- #
MAX_CHARS = 20_000  # a few emails are millions of chars long; cut them

RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_NUMBER = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
RE_NON_ALPHA = re.compile(r"[^a-z\s]")
RE_SPACES = re.compile(r"\s+")
STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """lowercase -> strip html -> url/email/number tokens -> remove punctuation
    -> remove stop-words and very short tokens."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text[:MAX_CHARS])
    text = RE_HTML_TAG.sub(" ", text)
    text = text.lower()
    text = RE_URL.sub(" urltoken ", text)
    text = RE_EMAIL.sub(" emailtoken ", text)
    text = RE_NUMBER.sub(" numtoken ", text)
    text = RE_NON_ALPHA.sub(" ", text)
    tokens = [t for t in RE_SPACES.split(text) if len(t) > 1 and t not in STOPWORDS]
    return " ".join(tokens)


def extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Small numeric features computed on the RAW text."""
    raw = df["subject"].fillna("") + " " + df["body"].fillna("").str.slice(0, MAX_CHARS)
    out = pd.DataFrame(index=df.index)
    out["n_urls"] = raw.str.count(RE_URL.pattern)
    out["n_emails"] = raw.str.count(RE_EMAIL.pattern)
    out["n_exclaim"] = raw.str.count("!")
    out["n_dollar"] = raw.str.count(r"\$")
    out["text_len"] = np.log1p(raw.str.len())
    out["upper_ratio"] = raw.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    out["subject_len"] = df["subject"].fillna("").str.len()
    out["has_html"] = raw.str.contains(RE_HTML_TAG.pattern, regex=True).astype(int)
    return out


# --------------------------------------------------------------------------- #
# 2. SMOTE (imblearn if available, otherwise a sparse-friendly fallback)
# --------------------------------------------------------------------------- #
def smote_fallback(X, y, k=5, random_state=42, chunk=200):
    """Minimal SMOTE for sparse matrices.
    For each synthetic sample: pick a random minority sample, pick one of its k
    nearest minority neighbours (cosine), and interpolate between the two."""
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority, majority_n = classes[counts.argmin()], counts.max()
    n_new = majority_n - counts.min()
    if n_new <= 0:
        return X, y

    X = sp.csr_matrix(X)
    X_min = X[np.where(y == minority)[0]]
    n_min = X_min.shape[0]

    seeds = rng.randint(0, n_min, size=n_new)          # seeds drawn with replacement
    uniq = np.unique(seeds)
    neighbours = {}
    for start in range(0, len(uniq), chunk):           # only compute kNN for chosen seeds
        idx = uniq[start:start + chunk]
        sims = (X_min[idx] @ X_min.T).toarray()        # TF-IDF rows are L2-normalised -> cosine
        sims[np.arange(len(idx)), idx] = -1            # exclude self
        nn = np.argpartition(-sims, k, axis=1)[:, :k]
        for i, row in zip(idx, nn):
            neighbours[i] = row

    nn_pick = np.array([rng.choice(neighbours[s]) for s in seeds])
    gap = rng.rand(n_new, 1)
    base, other = X_min[seeds], X_min[nn_pick]
    X_new = base + sp.csr_matrix(gap) .multiply(other - base)
    X_new = sp.csr_matrix(X_new)

    X_res = sp.vstack([X, X_new], format="csr")
    y_res = np.concatenate([y, np.full(n_new, minority)])
    return X_res, y_res


def apply_smote(X, y, k, seed):
    try:
        from imblearn.over_sampling import SMOTE
        print("  using imblearn.SMOTE")
        return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    except ImportError:
        print("  imbalanced-learn not installed -> using built-in SMOTE")
        return smote_fallback(X, y, k=k, random_state=seed)


# --------------------------------------------------------------------------- #
# 3. Main
# --------------------------------------------------------------------------- #
def main(args):
    os.makedirs(args.out, exist_ok=True)

    print("[1/6] Loading data ...")
    df = pd.read_csv(args.input)
    print(f"  raw shape: {df.shape}")
    print("  class counts:\n", df["label"].value_counts().to_string())

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df = df.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)

    print("[2/6] Cleaning text ...")
    df["text"] = (df["subject"] + " " + df["body"]).map(clean_text)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"  after cleaning: {df.shape}")

    feats = extra_features(df)
    df[["id", "label", "source", "text"]].to_csv(
        os.path.join(args.out, "cleaned_emails.csv"), index=False)

    print("[3/6] Train / test split (stratified) ...")
    idx_train, idx_test = train_test_split(
        df.index, test_size=args.test_size, stratify=df["label"], random_state=args.seed)
    y_train, y_test = df.loc[idx_train, "label"].values, df.loc[idx_test, "label"].values
    print("  train:", np.bincount(y_train), " test:", np.bincount(y_test))

    print("[4/6] TF-IDF (fit on train only) ...")
    tfidf = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2),
                            min_df=3, max_df=0.9, sublinear_tf=True, dtype=np.float32)
    Xt_train = tfidf.fit_transform(df.loc[idx_train, "text"])
    Xt_test = tfidf.transform(df.loc[idx_test, "text"])

    scaler = MaxAbsScaler()                                  # keeps sparsity
    F_train = scaler.fit_transform(feats.loc[idx_train].values.astype(np.float32))
    F_test = scaler.transform(feats.loc[idx_test].values.astype(np.float32))

    X_train = sp.hstack([Xt_train, sp.csr_matrix(F_train)], format="csr")
    X_test = sp.hstack([Xt_test, sp.csr_matrix(F_test)], format="csr")
    print(f"  X_train {X_train.shape}  X_test {X_test.shape}")

    print("[5/6] SMOTE on the training set ...")
    print("  before:", dict(zip(*np.unique(y_train, return_counts=True))))
    X_res, y_res = apply_smote(X_train, y_train, k=args.k, seed=args.seed)
    print("  after: ", dict(zip(*np.unique(y_res, return_counts=True))))

    print("[6/6] Saving ...")
    sp.save_npz(os.path.join(args.out, "X_train_smote.npz"), X_res)
    np.save(os.path.join(args.out, "y_train_smote.npy"), y_res)
    sp.save_npz(os.path.join(args.out, "X_test.npz"), X_test)
    np.save(os.path.join(args.out, "y_test.npy"), y_test)
    joblib.dump(tfidf, os.path.join(args.out, "tfidf_vectorizer.joblib"))
    joblib.dump(scaler, os.path.join(args.out, "feature_scaler.joblib"))
    print(f"Done. Files written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="combined_phishing_email_dataset.csv")
    p.add_argument("--out", default="output")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--k", type=int, default=5, help="SMOTE k_neighbors")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

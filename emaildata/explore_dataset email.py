"""
Explore combined_phishing_email_dataset.csv

Usage:
    python explore_dataset.py --data combined_phishing_email_dataset.csv
    python explore_dataset.py --data combined_phishing_email_dataset.csv --out eda_output

Creates in --out:
    eda_summary.txt            full text report
    01_class_balance.png       overall label counts
    02_source_by_label.png     rows per source, split by label
    03_length_distribution.png words per email, phishing vs legitimate
    04_length_by_source.png    median email length per source
    05_top_words.png           most distinctive words per class
    06_missing_values.png      missing data per column
Requires: pandas, numpy, matplotlib, scikit-learn
"""
import argparse
import io
import os
import re
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

LABEL_NAME = {0: "Legitimate (0)", 1: "Phishing/Spam (1)"}
COLORS = {0: "#2a9d8f", 1: "#e76f51"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="eda_output")
    return p.parse_args()


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["words"] = df["body"].str.split().str.len()
    df["chars"] = df["body"].str.len()

    buf = io.StringIO()
    with redirect_stdout(buf):
        section("1. OVERVIEW")
        print(f"Rows: {len(df):,}    Columns: {df.shape[1]}")
        print("Columns:", list(df.columns))
        print("\nLabel counts:")
        vc = df["label"].value_counts().sort_index()
        for k, v in vc.items():
            print(f"  {LABEL_NAME.get(k, k)}: {v:,}  ({v / len(df):.1%})")

        section("2. SOURCES x LABEL")
        ct = pd.crosstab(df["source"], df["label"].map(LABEL_NAME))
        ct["total"] = ct.sum(axis=1)
        print(ct.sort_values("total", ascending=False).to_string())
        single = [s for s, r in pd.crosstab(df["source"], df["label"]).iterrows() if (r > 0).sum() == 1]
        print("\nSources containing only ONE class (risk of the model learning source style):")
        print("  ", single if single else "none")

        section("3. MISSING VALUES")
        miss = pd.DataFrame({"missing": df.isna().sum(),
                             "empty_string": [(df[c].astype(str).str.strip() == "").sum() for c in df.columns]})
        miss["missing_%"] = (miss["missing"] / len(df) * 100).round(2)
        print(miss.to_string())

        section("4. EMAIL LENGTH (words in body)")
        print(df["words"].describe(percentiles=[.25, .5, .75, .9, .95, .99]).round(1).to_string())
        print("\nMedian words by label:")
        print(df.groupby("label")["words"].median().rename(LABEL_NAME).to_string())
        print("\nMedian words by source:")
        print(df.groupby("source")["words"].median().sort_values().to_string())
        print(f"\nEmails longer than 256 words: {(df['words'] > 256).mean():.1%}")
        print(f"Emails longer than 512 words: {(df['words'] > 512).mean():.1%}")
        print(f"Very short emails (<5 words): {(df['words'] < 5).sum():,}")

        section("5. DUPLICATES")
        key = df["subject"].str.lower().str.strip() + "||" + df["body"].str.lower().str.strip()
        print(f"Exact duplicate (subject+body) rows: {key.duplicated().sum():,}")
        print(f"Duplicate bodies only: {df['body'].str.lower().str.strip().duplicated().sum():,}")

        section("6. URLS (if present)")
        if "urls" in df.columns and df["urls"].notna().any():
            u = pd.to_numeric(df["urls"], errors="coerce")
            print("Share of rows with url flag/count > 0, by label:")
            print((u.fillna(0) > 0).groupby(df["label"]).mean().rename(LABEL_NAME).round(3).to_string())
        else:
            print("No usable 'urls' column.")
        has_http = df["body"].str.contains(r"https?://|www\.", case=False, regex=True)
        print("\nBodies containing a link, by label:")
        print(has_http.groupby(df["label"]).mean().rename(LABEL_NAME).round(3).to_string())

        section("7. TEXT STYLE SIGNALS (share of emails, by label)")
        sig = pd.DataFrame({
            "has_exclamation": df["body"].str.contains("!", regex=False),
            "has_dollar": df["body"].str.contains(r"\$", regex=True),
            "has_urgent_words": df["body"].str.contains(r"urgent|immediately|verify|suspend|confirm|password|account",
                                                         case=False, regex=True),
            "mostly_uppercase_subject": df["subject"].apply(lambda s: len(s) > 5 and s.isupper()),
        })
        print(sig.groupby(df["label"]).mean().rename(index=LABEL_NAME).round(3).T.to_string())

        section("8. MOST DISTINCTIVE WORDS PER CLASS")
        sample = df.sample(n=min(30000, len(df)), random_state=42)
        cv = CountVectorizer(stop_words="english", max_features=20000, min_df=5, token_pattern=r"[a-zA-Z]{3,}")
        X = cv.fit_transform(sample["body"].str.slice(0, 3000))
        vocab = np.array(cv.get_feature_names_out())
        y = sample["label"].values
        c1 = np.asarray(X[y == 1].sum(axis=0)).ravel() + 1
        c0 = np.asarray(X[y == 0].sum(axis=0)).ravel() + 1
        logodds = np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())
        top1 = vocab[np.argsort(-logodds)[:20]]
        top0 = vocab[np.argsort(logodds)[:20]]
        print("Most phishing/spam-like:", ", ".join(top1))
        print("Most legitimate-like:  ", ", ".join(top0))

        section("9. SAMPLE EMAILS (random, truncated)")
        for lab in (1, 0):
            print(f"\n--- {LABEL_NAME[lab]} ---")
            for _, r in df[df["label"] == lab].sample(3, random_state=1).iterrows():
                body = re.sub(r"\s+", " ", r["body"])[:220]
                print(f"[{r['source']}] Subject: {r['subject'][:80]}\n    {body}...")

    report = buf.getvalue()
    print(report)
    with open(os.path.join(args.out, "eda_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # ---------------- charts ----------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df["label"].value_counts().sort_index()
    ax.bar([LABEL_NAME[i] for i in vc.index], vc.values, color=[COLORS[i] for i in vc.index])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("Class balance"); ax.set_ylabel("Emails")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "01_class_balance.png"), dpi=150); plt.close(fig)

    piv = pd.crosstab(df["source"], df["label"]).reindex(columns=[0, 1], fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(piv.index, piv[0], color=COLORS[0], label=LABEL_NAME[0])
    ax.barh(piv.index, piv[1], left=piv[0], color=COLORS[1], label=LABEL_NAME[1])
    ax.set_title("Emails per source, by label"); ax.set_xlabel("Emails"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "02_source_by_label.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    clip = df["words"].clip(upper=1500)
    bins = np.linspace(0, 1500, 61)
    for lab in (0, 1):
        ax.hist(clip[df["label"] == lab], bins=bins, alpha=0.6, color=COLORS[lab], label=LABEL_NAME[lab])
    ax.set_title("Email length (words, capped at 1500)"); ax.set_xlabel("Words"); ax.set_ylabel("Emails")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "03_length_distribution.png"), dpi=150); plt.close(fig)

    med = df.groupby("source")["words"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(med.index, med.values, color="#457b9d")
    ax.set_title("Median email length by source"); ax.set_xlabel("Words")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "04_length_by_source.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, lab, top in ((axes[0], 1, np.argsort(-logodds)[:15]), (axes[1], 0, np.argsort(logodds)[:15])):
        vals = np.abs(logodds[top])[::-1]
        ax.barh(vocab[top][::-1], vals, color=COLORS[lab])
        ax.set_title(f"Most distinctive: {LABEL_NAME[lab]}"); ax.set_xlabel("log-odds vs other class")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "05_top_words.png"), dpi=150); plt.close(fig)

    m = (df.isna().mean() * 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(m.index, m.values, color="#8d99ae"); ax.set_ylabel("% missing")
    ax.set_title("Missing values per column"); plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "06_missing_values.png"), dpi=150); plt.close(fig)

    print(f"\nSaved report and 6 charts to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

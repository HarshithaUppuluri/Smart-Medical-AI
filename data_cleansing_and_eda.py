"""Cleans the 4 raw datasets and produces EDA plots for the report.

Outputs:
  data/processed/classification.csv  (one-hot symptoms -> disease)
  data/processed/nlp.csv             (cleaned symptom text -> disease)
  data/processed/sentiment.csv       (cleaned + balanced drug review -> sentiment)
  reports_assets/*.png               (EDA charts)
"""
import html
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import RAW_DIR, PROCESSED_DIR, ASSETS_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid")


def savefig(name):
    path = ASSETS_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------- classification
def process_classification():
    print("\n[1/3] Classification dataset (symptoms -> disease)")
    df = pd.read_csv(RAW_DIR / "classification" / "dataset.csv")
    severity = pd.read_csv(RAW_DIR / "classification" / "Symptom-severity.csv")

    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    df[symptom_cols] = df[symptom_cols].apply(lambda c: c.str.strip())
    df["Disease"] = df["Disease"].str.strip()

    vocab = sorted(severity["Symptom"].str.strip().unique())
    print(f"  {len(df)} rows, {df['Disease'].nunique()} diseases, {len(vocab)} known symptoms")

    # one-hot encode: 1 if symptom present anywhere in the row's symptom list
    row_symptom_sets = df[symptom_cols].apply(lambda r: set(r.dropna()), axis=1)
    onehot = pd.DataFrame(
        {sym: row_symptom_sets.apply(lambda s: int(sym in s)) for sym in vocab}
    )
    # NOTE: rows are intentionally repeated in the source data (each disease has
    # several valid symptom-set presentations) -- kept as-is so every disease has
    # enough rows for a meaningful stratified train/test split.
    clean = pd.concat([df["Disease"], onehot], axis=1).reset_index(drop=True)
    clean.to_csv(PROCESSED_DIR / "classification.csv", index=False)
    print(f"  saved data/processed/classification.csv {clean.shape}")

    plt.figure(figsize=(10, 10))
    clean["Disease"].value_counts().plot(kind="barh")
    plt.title("Classification: cases per disease")
    plt.xlabel("count")
    savefig("classification_class_balance.png")

    plt.figure(figsize=(8, 8))
    onehot.sum().sort_values(ascending=False).head(20).plot(kind="barh")
    plt.title("Classification: 20 most common symptoms")
    plt.xlabel("count")
    savefig("classification_top_symptoms.png")


# ---------------------------------------------------------------- nlp
def process_nlp():
    print("\n[2/3] NLP dataset (free-text symptoms -> disease)")
    df = pd.read_csv(RAW_DIR / "nlp" / "Symptom2Disease.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["text"] = df["text"].str.strip().str.lower()
    df["label"] = df["label"].str.strip()
    before = len(df)
    df = df.drop_duplicates(subset="text").dropna().reset_index(drop=True)
    print(f"  {before} -> {len(df)} rows after dedup, {df['label'].nunique()} classes")

    df["text_len"] = df["text"].str.split().apply(len)
    df.to_csv(PROCESSED_DIR / "nlp.csv", index=False)
    print(f"  saved data/processed/nlp.csv {df.shape}")

    plt.figure(figsize=(8, 8))
    df["label"].value_counts().plot(kind="barh")
    plt.title("NLP: symptom descriptions per disease label")
    savefig("nlp_class_balance.png")

    plt.figure(figsize=(6, 4))
    df["text_len"].hist(bins=20)
    plt.title("NLP: symptom description length (words)")
    plt.xlabel("word count")
    savefig("nlp_text_length.png")


# ---------------------------------------------------------------- sentiment
CLEAN_RE = re.compile(r"<[^>]+>")


def clean_review(text: str) -> str:
    text = html.unescape(str(text))
    text = CLEAN_RE.sub(" ", text)
    text = text.replace('"', "").replace("&#039;", "'")
    return re.sub(r"\s+", " ", text).strip()


def rating_to_sentiment(rating: int) -> str:
    if rating >= 7:
        return "Positive"
    if rating >= 5:
        return "Neutral"
    return "Negative"


def process_sentiment():
    print("\n[3/3] Sentiment dataset (drug reviews -> Positive/Neutral/Negative)")
    train = pd.read_csv(RAW_DIR / "sentiment" / "drugsComTrain_raw.csv")
    test = pd.read_csv(RAW_DIR / "sentiment" / "drugsComTest_raw.csv")
    df = pd.concat([train, test], ignore_index=True)
    print(f"  {len(df)} raw reviews")

    df["review"] = df["review"].apply(clean_review)
    df = df[df["review"].str.split().apply(len) >= 3]
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    df = df.drop_duplicates(subset="review").dropna(subset=["review"]).reset_index(drop=True)
    print(f"  {len(df)} rows after cleaning/dedup")
    print(f"  class balance before undersampling:\n{df['sentiment'].value_counts()}")

    # balance classes by undersampling the majority classes so training is fast
    # and the model isn't just learning to predict "Positive" every time
    n_per_class = min(6000, df["sentiment"].value_counts().min())
    parts = [
        df[df["sentiment"] == cls].sample(n=n_per_class, random_state=RANDOM_STATE)
        for cls in df["sentiment"].unique()
    ]
    balanced = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    keep_cols = ["drugName", "condition", "review", "rating", "sentiment"]
    balanced = balanced[keep_cols]
    balanced.to_csv(PROCESSED_DIR / "sentiment.csv", index=False)
    print(f"  saved data/processed/sentiment.csv {balanced.shape} ({n_per_class}/class)")

    plt.figure(figsize=(5, 4))
    df["rating"].value_counts().sort_index().plot(kind="bar")
    plt.title("Sentiment: raw star-rating distribution")
    savefig("sentiment_rating_distribution.png")

    plt.figure(figsize=(5, 4))
    balanced["sentiment"].value_counts().plot(kind="bar", color=["#2a9d8f", "#e9c46a", "#e76f51"])
    plt.title("Sentiment: balanced class distribution (used for training)")
    savefig("sentiment_class_balance.png")


if __name__ == "__main__":
    process_classification()
    process_nlp()
    process_sentiment()
    print("\nDone. Processed files in data/processed/, EDA charts in reports_assets/")

import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

for d in [DATA_DIR, RAW_DIR, PROC_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\-\'\"]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# DATASETS
# =========================
def load_coaid() -> pd.DataFrame:
    base_dir = RAW_DIR / "CoAID-master"
    if not base_dir.exists():
        raise FileNotFoundError(f"CoAID-master not found in: {RAW_DIR}")

    fake_files = list(base_dir.rglob("NewsFakeCOVID-19.csv"))
    real_files = list(base_dir.rglob("NewsRealCOVID-19.csv"))

    if not fake_files or not real_files:
        raise RuntimeError("Could not locate CoAID NewsFake/NewsReal CSV files.")

    def read_files(paths, label: int) -> pd.DataFrame:
        frames = []
        for p in paths:
            tmp = pd.read_csv(p)
            cols = {c.lower(): c for c in tmp.columns}
            if "content" in cols:
                text_col = cols["content"]
            elif "title" in cols:
                text_col = cols["title"]
            else:
                text_col = tmp.columns[0]

            frames.append(pd.DataFrame({
                "text": tmp[text_col].astype(str).apply(clean_text),
                "label": label
            }))

        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all[df_all["text"].str.len() > 20].copy()
        df_all["crisis_type"] = "health"
        df_all["source"] = "CoAID"
        return df_all

    df_fake = read_files(fake_files, 1)
    df_real = read_files(real_files, 0)
    df = pd.concat([df_fake, df_real], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def load_kaggle_disaster_tweets() -> pd.DataFrame:
    path = RAW_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError("Kaggle train.csv not found in data/raw/")

    df = pd.read_csv(path)
    df_out = pd.DataFrame()
    df_out["text"] = df["text"].astype(str).apply(clean_text)
    df_out["is_disaster"] = df["target"].astype(int)

    df_out = df_out[df_out["is_disaster"] == 1].copy()
    df_out = df_out.drop(columns=["is_disaster"])

    df_out["label"] = 0
    df_out["crisis_type"] = "climate"
    df_out["source"] = "Kaggle_Disaster_Tweets"

    df_out = df_out[df_out["text"].str.len() > 20].copy()
    df_out = df_out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df_out


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]
    n = min(len(df0), len(df1))

    df0 = df0.sample(n=n, random_state=RANDOM_SEED)
    df1 = df1.sample(n=n, random_state=RANDOM_SEED)

    df_bal = pd.concat([df0, df1], ignore_index=True)
    df_bal = df_bal.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_bal


# =========================
# TRAINING
# =========================
def train_model(df: pd.DataFrame) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, Any]]:
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED
    )

    tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    metrics = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "classification_report_text": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "dataset_size": len(df),
        "seed": RANDOM_SEED
    }
    return tfidf, clf, metrics


def main():
    print("[1] Loading datasets...")
    df_health = load_coaid()
    df_climate = load_kaggle_disaster_tweets()

    df = pd.concat([df_health, df_climate], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("[2] Balancing dataset...")
    df = balance_dataset(df)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC_DIR / "combined_dataset_balanced.csv", index=False, encoding="utf-8")

    print("[3] Training model...")
    tfidf, clf, metrics = train_model(df)

    # Save model artifacts
    joblib.dump(tfidf, MODELS_DIR / "tfidf.joblib")
    joblib.dump(clf, MODELS_DIR / "classifier.joblib")

    save_json(metrics, RESULTS_DIR / "training_metrics.json")

    print("\n[OK] Training complete.")
    print("[OK] Saved model files:")
    print(f" - {MODELS_DIR / 'tfidf.joblib'}")
    print(f" - {MODELS_DIR / 'classifier.joblib'}")
    print("\n[METRICS]\n")
    print(metrics["classification_report_text"])


if __name__ == "__main__":
    main()

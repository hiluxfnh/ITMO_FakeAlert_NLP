import os
import re
import json
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# CONFIG
# ----------------------------
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

# Label: 0=credible, 1=fake/misleading
# crisis_type: health, climate


# ----------------------------
# HELPERS
# ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def download_file(url: str, out_path: Path):
    """
    Download a file using urllib (no extra dependencies).
    """
    import urllib.request
    print(f"[DOWNLOAD] {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[OK] Saved to: {out_path}")


# ----------------------------
# DATASET 1: CoAID (health crisis)
# Source: https://github.com/cuilimeng/CoAID
# ----------------------------
def load_coaid() -> pd.DataFrame:
    """
    Load CoAID dataset using the explicit files:
    - NewsFakeCOVID-19.csv -> label = 1
    - NewsRealCOVID-19.csv -> label = 0

    We combine across all date folders available in CoAID.
    """
    base_dir = RAW_DIR / "CoAID-master"

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted CoAID folder at: {base_dir}\n"
            f"Make sure CoAID zip was extracted into data/raw/."
        )

    fake_files = list(base_dir.rglob("NewsFakeCOVID-19.csv"))
    real_files = list(base_dir.rglob("NewsRealCOVID-19.csv"))

    if len(fake_files) == 0 or len(real_files) == 0:
        raise RuntimeError("Could not find CoAID NewsFake/NewsReal CSV files.")

    def read_and_normalize(paths, label_value: int) -> pd.DataFrame:
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

            df_part = pd.DataFrame()
            df_part["text"] = tmp[text_col].astype(str).apply(clean_text)
            df_part["label"] = label_value
            frames.append(df_part)

        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all[df_all["text"].str.len() > 20].copy()
        return df_all

    df_fake = read_and_normalize(fake_files, 1)
    df_real = read_and_normalize(real_files, 0)

    df = pd.concat([df_fake, df_real], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df["crisis_type"] = "health"
    df["source"] = "CoAID"

    return df


# ----------------------------
# DATASET 2: CrisisMMD (climate/disaster crisis)
# Source: CrisisNLP: https://crisisnlp.qcri.org/crisismmd.html
# Note: Direct downloads might require acceptance / terms.
# For 2-day project, use a public benchmark repository or text-only subset.
# We'll use CrisisBench repository which aggregates crisis datasets.
# Source: https://github.com/firojalam/crisis_datasets_benchmarks
# ----------------------------
def load_kaggle_disaster_tweets() -> pd.DataFrame:
    """
    Kaggle Disaster Tweets dataset:
    - text column: 'text'
    - target column: 'target' (1=disaster, 0=not disaster)

    We will treat:
    - disaster-related messages as credible crisis alerts (label=0)
    because Kaggle target is not a misinformation label.

    Purpose:
    Provide a real climate/disaster crisis text domain for the system.
    """
    path = RAW_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at: {path}\n"
            f"Download train.csv from Kaggle competition 'NLP Getting Started' and put it in data/raw/"
        )

    df = pd.read_csv(path)

    if "text" not in df.columns or "target" not in df.columns:
        raise RuntimeError("Kaggle train.csv must contain columns: text, target")

    df_out = pd.DataFrame()
    df_out["text"] = df["text"].astype(str).apply(clean_text)

    # Keep only disaster-related messages (crisis alerts)
    df_out["is_disaster"] = df["target"].astype(int)
    df_out = df_out[df_out["is_disaster"] == 1].copy()

    # Kaggle dataset is not fake-vs-real, so label as credible crisis alerts
    df_out["label"] = 0
    df_out["crisis_type"] = "climate"
    df_out["source"] = "Kaggle_Disaster_Tweets"

    df_out = df_out[df_out["text"].str.len() > 20].copy()
    df_out = df_out.drop(columns=["is_disaster"])
    df_out = df_out.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df_out

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance dataset by downsampling the majority class (label=0).
    This prevents the model from predicting 'CREDIBLE' for everything.
    """
    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]

    n = min(len(df0), len(df1))
    df0_sample = df0.sample(n=n, random_state=RANDOM_SEED)
    df1_sample = df1.sample(n=n, random_state=RANDOM_SEED)

    df_bal = pd.concat([df0_sample, df1_sample], ignore_index=True)
    df_bal = df_bal.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_bal


# ----------------------------
# TRAIN BASELINE MODEL
# ----------------------------
def train_baseline(df: pd.DataFrame) -> Tuple[TfidfVectorizer, LogisticRegression, dict]:
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y if len(set(y)) > 1 else None
    )

    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    print("\n[METRICS] Classification Report (Baseline)")
    print(classification_report(y_test, y_pred))

    print("[METRICS] Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))


    return tfidf, clf, {"report": report, "confusion_matrix": cm}


def predict_alert(text: str, tfidf, clf):
    t = clean_text(text)
    vec = tfidf.transform([t])
    pred = int(clf.predict(vec)[0])
    proba = float(clf.predict_proba(vec)[0][pred])
    label = "FAKE/MISLEADING" if pred == 1 else "CREDIBLE"
    return {"prediction": label, "confidence": round(proba, 3)}


# ----------------------------
# MAIN
# ----------------------------
def main():
    print("==============================================")
    print("ITMO Research Internship: Real Dataset Training")
    print("Fake Alerts Detection (Climate + Health)")
    print("==============================================\n")

    # Load datasets
    print("[STAGE 1] Loading CoAID (health misinformation dataset)")
    df_health = load_coaid()
    print(f"[OK] CoAID samples: {len(df_health)}")

    print("\n[STAGE 2] Loading disaster/crisis dataset (Kaggle Disaster Tweets)")
    df_climate = load_kaggle_disaster_tweets()
    print(f"[OK] Climate samples: {len(df_climate)}")

    # Combine
    df = pd.concat([df_health, df_climate], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print("\n[STAGE] Balancing dataset (downsampling majority class)")
    df = balance_dataset(df)
    print(df["label"].value_counts())


    # Save processed dataset
    processed_path = PROC_DIR / "combined_dataset_balanced.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8")

    print(f"\n[OK] Combined dataset saved: {processed_path}")
    print(df["label"].value_counts())

    # Train baseline
    print("\n[STAGE 3] Training baseline model (TF-IDF + Logistic Regression)")
    tfidf, clf, metrics = train_baseline(df)

    # Save results
    save_json(metrics["report"], RESULTS_DIR / "baseline_report.json")
    save_json({"confusion_matrix": metrics["confusion_matrix"]}, RESULTS_DIR / "baseline_confusion_matrix.json")

    print("[OK] Baseline training complete.")
    print("[OK] Saved results to /results.")

    print("\n[DEMO] Showing real examples from dataset (credible vs fake)\n")

    sample_fake = df[df["label"] == 1].sample(2, random_state=RANDOM_SEED)["text"].tolist()
    sample_real = df[df["label"] == 0].sample(2, random_state=RANDOM_SEED)["text"].tolist()

    demo_texts = []
    for t in sample_fake:
        demo_texts.append(("FAKE example (from dataset)", t))
    for t in sample_real:
        demo_texts.append(("CREDIBLE example (from dataset)", t))

    # Demo predictions
    print("\n[DEMO] Real-time text classification demo\n")
    for tag, t in demo_texts:
        out = predict_alert(t, tfidf, clf)
        print(f"{tag}: {t[:200]}...")
        print(f"Prediction: {out['prediction']} | Confidence: {out['confidence']}")
        print("-" * 70)


    for t in demo_texts:
        out = predict_alert(t, tfidf, clf)
        print(f"Text: {t}")
        print(f"Prediction: {out['prediction']} | Confidence: {out['confidence']}")
        print("-" * 70)

    print("\n[FINISHED] Real-data pipeline completed successfully.")
    print("Next: If time allows, we add DistilBERT fine-tuning.\n")


if __name__ == "__main__":
    main()

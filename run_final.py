import re
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

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


# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\-\'\"]+", " ", text)  # keep basic punctuation
    text = re.sub(r"\s+", " ", text)
    return text


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# DATASET 1: CoAID (Health Crisis, Fake vs Real)
# =========================
def load_coaid() -> pd.DataFrame:
    """
    CoAID dataset: explicit labeling based on file names.
    - NewsFakeCOVID-19.csv -> label=1
    - NewsRealCOVID-19.csv -> label=0
    """
    base_dir = RAW_DIR / "CoAID-master"
    if not base_dir.exists():
        raise FileNotFoundError(
            f"CoAID folder not found: {base_dir}\n"
            f"Ensure CoAID-master folder exists in data/raw/"
        )

    fake_files = list(base_dir.rglob("NewsFakeCOVID-19.csv"))
    real_files = list(base_dir.rglob("NewsRealCOVID-19.csv"))

    if not fake_files or not real_files:
        raise RuntimeError("CoAID files not found. Expected NewsFakeCOVID-19.csv and NewsRealCOVID-19.csv")

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

            df_part = pd.DataFrame({
                "text": tmp[text_col].astype(str).apply(clean_text),
                "label": label
            })
            frames.append(df_part)

        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all[df_all["text"].str.len() > 20].copy()
        return df_all

    df_fake = read_files(fake_files, 1)
    df_real = read_files(real_files, 0)

    df = pd.concat([df_fake, df_real], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df["crisis_type"] = "health"
    df["source"] = "CoAID"
    return df


# =========================
# DATASET 2: Kaggle Disaster Tweets (Climate/Disaster Crisis Domain)
# =========================
def load_kaggle_disaster_tweets() -> pd.DataFrame:
    """
    Kaggle Disaster Tweets:
    - 'text' column contains tweets
    - 'target' indicates disaster-related tweet (1) vs non-disaster (0)

    We keep only disaster-related messages and treat them as credible crisis alerts (label=0).
    """
    path = RAW_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at: {path}\n"
            f"Download Kaggle 'NLP Getting Started' train.csv and place it in data/raw/"
        )

    df = pd.read_csv(path)
    if "text" not in df.columns or "target" not in df.columns:
        raise RuntimeError("Kaggle train.csv missing expected columns: text, target")

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


# =========================
# BALANCING
# =========================
def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downsample majority class to create a balanced dataset.
    """
    df0 = df[df["label"] == 0].copy()
    df1 = df[df["label"] == 1].copy()

    n = min(len(df0), len(df1))
    df0 = df0.sample(n=n, random_state=RANDOM_SEED)
    df1 = df1.sample(n=n, random_state=RANDOM_SEED)

    df_bal = pd.concat([df0, df1], ignore_index=True)
    df_bal = df_bal.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_bal


# =========================
# TRAINING
# =========================
def train_baseline(df: pd.DataFrame) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, Any]]:
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y
    )

    tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Class-weight helps robustness even after balancing
    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    report_text = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "classification_report_text": report_text,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "test_size": len(y_test),
        "seed": RANDOM_SEED,
        "tfidf_max_features": 12000,
        "tfidf_ngram_range": [1, 2]
    }

    return tfidf, clf, metrics


def predict_alert(text: str, tfidf: TfidfVectorizer, clf: LogisticRegression) -> Dict[str, Any]:
    t = clean_text(text)
    vec = tfidf.transform([t])
    pred = int(clf.predict(vec)[0])
    proba = float(clf.predict_proba(vec)[0][pred])
    label = "FAKE/MISLEADING" if pred == 1 else "CREDIBLE"
    return {"prediction": label, "confidence": round(proba, 3)}


def top_features(tfidf: TfidfVectorizer, clf: LogisticRegression, top_n: int = 20) -> Dict[str, Any]:
    """
    Extract top weighted features for interpretability.
    This is valuable for the report.
    """
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]  # binary classifier

    top_fake_idx = np.argsort(coefs)[-top_n:]       # highest -> fake
    top_real_idx = np.argsort(coefs)[:top_n]        # lowest -> credible

    return {
        "top_fake_terms": feature_names[top_fake_idx][::-1].tolist(),
        "top_credible_terms": feature_names[top_real_idx].tolist()
    }


# =========================
# MAIN
# =========================
def main():
    print("==============================================")
    print("ITMO Research Internship: Final Pipeline")
    print("Fake Alerts Detection (Climate + Health)")
    print("==============================================\n")

    print("[STAGE 1] Loading CoAID (health misinformation)")
    df_health = load_coaid()
    print(f"[OK] CoAID samples: {len(df_health)}")

    print("\n[STAGE 2] Loading Kaggle Disaster Tweets (climate/disaster domain)")
    df_climate = load_kaggle_disaster_tweets()
    print(f"[OK] Climate samples: {len(df_climate)}")

    # Combine
    df = pd.concat([df_health, df_climate], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("\n[STAGE 3] Balancing dataset")
    df_bal = balance_dataset(df)
    print(df_bal["label"].value_counts())

    # Save processed dataset
    processed_path = PROC_DIR / "combined_dataset_balanced.csv"
    df_bal.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"\n[OK] Saved balanced dataset: {processed_path}")

    # Train model
    print("\n[STAGE 4] Training baseline model (TF-IDF + Logistic Regression)")
    tfidf, clf, metrics = train_baseline(df_bal)

    # Save results
    save_json(metrics, RESULTS_DIR / "baseline_metrics.json")
    print("\n[METRICS] Classification Report\n")
    print(metrics["classification_report_text"])
    print("[METRICS] Confusion Matrix")
    print(np.array(metrics["confusion_matrix"]))

    # Feature interpretability
    feats = top_features(tfidf, clf, top_n=20)
    save_json(feats, RESULTS_DIR / "top_features.json")
    print("\n[INTERPRETABILITY] Top FAKE terms:", feats["top_fake_terms"][:10])
    print("[INTERPRETABILITY] Top CREDIBLE terms:", feats["top_credible_terms"][:10])

    # Demo predictions using real samples
    print("\n[DEMO] Real dataset samples (credible vs fake)\n")
    sample_fake = df_bal[df_bal["label"] == 1].sample(2, random_state=RANDOM_SEED)["text"].tolist()
    sample_real = df_bal[df_bal["label"] == 0].sample(2, random_state=RANDOM_SEED)["text"].tolist()

    demo_texts = [("FAKE example", t) for t in sample_fake] + [("CREDIBLE example", t) for t in sample_real]

    for tag, text in demo_texts:
        out = predict_alert(text, tfidf, clf)
        print(f"{tag}: {text[:200]}...")
        print(f"Prediction: {out['prediction']} | Confidence: {out['confidence']}")
        print("-" * 70)

    print("\n[FINISHED] Final pipeline completed successfully.")
    print("Outputs created:")
    print(f"- {processed_path}")
    print(f"- {RESULTS_DIR / 'baseline_metrics.json'}")
    print(f"- {RESULTS_DIR / 'top_features.json'}")
    print("\nNext: DistilBERT fine-tuning (optional improvement).")


if __name__ == "__main__":
    main()

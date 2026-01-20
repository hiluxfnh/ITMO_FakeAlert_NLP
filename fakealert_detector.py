import re
import numpy as np
import joblib
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


# -------------------------
# Text cleaning
# -------------------------
def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# -------------------------
# Rule-based threat signals
# -------------------------
RISK_PATTERNS = [
    (r"\burgent\b", 2.0, "Uses urgency trigger word 'urgent'"),
    (r"\bbreaking\b", 1.5, "Uses sensational trigger word 'breaking'"),
    (r"\bshare\b|\bforward\b|\bsend to\b", 2.5, "Encourages forwarding/sharing (viral manipulation pattern)"),
    (r"\bsecret\b|\bhidden info\b", 2.5, "Claims secret/hidden information (common hoax pattern)"),
    (r"\bhurry\b|\bact now\b|\bimmediately\b", 2.0, "Pressures immediate action (panic engineering)"),
    (r"\bhit\b|\battack\b|\bstarts tomorrow\b", 1.5, "Predicts dramatic near-future event without evidence"),
    (r"\bhide\b|\brun\b|\bevacuate\b", 2.0, "Strong imperative action commands (panic engineering)"),
    (r"\bunder your beds\b", 3.0, "Highly irrational instruction (strong hoax signature)"),
    (r"\bno one will tell you\b|\bthey won’t tell you\b", 2.5, "Conspiracy framing (authority distrust pattern)"),
]


def rule_risk_score(raw_text: str):
    """
    Return risk score and matched signals based on security heuristics.
    """
    t = clean_text(raw_text)
    score = 0.0
    signals = []

    for pattern, weight, reason in RISK_PATTERNS:
        if re.search(pattern, t):
            score += weight
            signals.append(reason)

    # Missing source hint: no URL, no agency names => suspicious
    has_url = bool(re.search(r"http\S+|www\.\S+", raw_text.lower()))
    has_authority = bool(re.search(r"\b(who|ministry|government|agency|meteorological|health authority)\b", t))

    if (not has_url) and (not has_authority):
        score += 1.5
        signals.append("No credible reference (no link/agency mentioned)")

    return score, signals


# -------------------------
# ML explanation
# -------------------------
def explain_ml(text: str, tfidf, clf, top_k: int = 8):
    t = clean_text(text)
    vec = tfidf.transform([t])

    pred = int(clf.predict(vec)[0])
    proba = float(clf.predict_proba(vec)[0][pred])

    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    vec_coo = vec.tocoo()
    contributions = []
    for idx, val in zip(vec_coo.col, vec_coo.data):
        contributions.append((feature_names[idx], float(val * coefs[idx])))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:top_k]

    reasons = []
    for term, score in top:
        if score > 0:
            reasons.append(f"'{term}' pushes ML toward FAKE")
        else:
            reasons.append(f"'{term}' pushes ML toward CREDIBLE")

    ml_label = "FAKE/MISLEADING" if pred == 1 else "CREDIBLE"
    ml_confidence = proba

    return ml_label, ml_confidence, reasons


# -------------------------
# Hybrid decision
# -------------------------
def hybrid_decision(text: str, tfidf, clf):
    ml_label, ml_conf, ml_reasons = explain_ml(text, tfidf, clf, top_k=6)
    risk_score, risk_signals = rule_risk_score(text)

    # Convert ML output into "fake probability"
    # If ML says fake => fake_prob = ml_conf
    # If ML says credible => fake_prob = 1 - ml_conf
    fake_prob_ml = ml_conf if ml_label == "FAKE/MISLEADING" else (1 - ml_conf)

    # Risk score normalization: cap at 10
    risk_score_capped = min(risk_score, 10.0)

    # Combine
    # 70% ML + 30% rule signals (security hybrid)
    hybrid_fake_prob = 0.7 * fake_prob_ml + 0.3 * (risk_score_capped / 10.0)

    # Decide
    final_label = "FAKE/MISLEADING" if hybrid_fake_prob >= 0.55 else "CREDIBLE"
    credibility_score = 1 - hybrid_fake_prob  # higher = more credible

    return {
        "final_label": final_label,
        "credibility_score": round(float(credibility_score), 3),
        "fake_probability": round(float(hybrid_fake_prob), 3),
        "ml_label": ml_label,
        "ml_confidence": round(float(ml_conf), 3),
        "rule_risk_score": round(float(risk_score), 2),
        "rule_signals": risk_signals[:6],
        "ml_reasons": ml_reasons[:6],
    }


def main():
    tfidf_path = MODELS_DIR / "tfidf.joblib"
    clf_path = MODELS_DIR / "classifier.joblib"

    if not tfidf_path.exists() or not clf_path.exists():
        print("Model files not found. Train first:")
        print("  python train_model.py")
        return

    tfidf = joblib.load(tfidf_path)
    clf = joblib.load(clf_path)

    print("==============================================")
    print(" Hybrid Fake Alert Detector (NLP + Security Rules)")
    print(" Paste a message and press Enter.")
    print(" Type 'exit' to quit.")
    print("==============================================\n")

    while True:
        user_text = input("Enter crisis alert text > ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Exiting detector.")
            break
        if len(user_text) < 10:
            print("Please enter a longer message.\n")
            continue

        res = hybrid_decision(user_text, tfidf, clf)

        print("\n--- RESULT ---")
        print(f"Final decision     : {res['final_label']}")
        print(f"Credibility score  : {res['credibility_score']}  (higher=more credible)")
        print(f"Fake probability   : {res['fake_probability']}")
        print("\n[ML component]")
        print(f"ML label           : {res['ml_label']}")
        print(f"ML confidence      : {res['ml_confidence']}")

        print("\n[Rule-based signals]")
        print(f"Risk score         : {res['rule_risk_score']}")
        if res["rule_signals"]:
            for s in res["rule_signals"]:
                print(f" - {s}")
        else:
            print(" - No high-risk patterns detected")

        print("\n[Top ML reasons]")
        for r in res["ml_reasons"]:
            print(f" - {r}")

        print("--------------\n")


if __name__ == "__main__":
    main()

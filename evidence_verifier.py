import os
import re
import json
import requests
import feedparser
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone, timedelta


# ============================
# CONFIG
# ============================
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SERPER_KEY_FILE = PROJECT_DIR / "serper_key.txt"

TRUSTED_DOMAINS = {
    "general": [
        "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "theguardian.com",
        "un.org", "news.un.org", "reliefweb.int"
    ],
    "climate": [
        "metoffice.gov.uk", "wmo.int", "noaa.gov", "weather.gov",
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com"
    ],
    "health": [
        "who.int", "unicef.org", "cdc.gov", "ecdc.europa.eu",
        "news.un.org", "reliefweb.int", "reuters.com", "bbc.com"
    ]
}

TIME_SENSITIVE_WORDS = [
    "today", "now", "currently", "right now", "tonight", "tomorrow",
    "just confirmed", "breaking", "urgent", "immediately",
    "this morning", "this evening", "in 6 hours", "in 24 hours",
    "yesterday"
]

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}


# ============================
# DATA STRUCTURE
# ============================
@dataclass
class EvidenceItem:
    title: str
    link: str
    snippet: str
    source: str
    domain: str
    published: Optional[str] = None
    published_dt: Optional[str] = None
    score: int = 0
    age_days: Optional[int] = None
    recency_weight: float = 0.0
    location_match: bool = False
    location_score: float = 0.0


# ============================
# BASIC UTILITIES
# ============================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_domain(url: str) -> str:
    try:
        url = url.lower().strip()
        url = re.sub(r"^https?://", "", url)
        url = url.split("/")[0]
        url = url.replace("www.", "")
        return url
    except Exception:
        return ""


def load_serper_key() -> str:
    if SERPER_KEY_FILE.exists():
        return SERPER_KEY_FILE.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(
        f"Missing {SERPER_KEY_FILE}. Create it and paste your Serper API key inside."
    )


def is_time_sensitive(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in TIME_SENSITIVE_WORDS)


def extract_keywords(text: str, limit: int = 8) -> List[str]:
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    words = [w for w in t.split() if len(w) > 3]
    stop = set([
        "this","that","with","from","they","them","there","about","have","will","just",
        "your","what","when","where","which","would","should","could","also","because",
        "into","over","under","while","said","says","breaking","urgent","please","share",
        "forward","confirmed","currently","government","authority","authorities"
    ])
    words = [w for w in words if w not in stop]
    return words[:limit]


# ============================
# LOCATION + DATE EXTRACTION
# ============================
def extract_location(text: str) -> Optional[str]:
    """
    Very practical location extraction for MSc prototype:
    - looks for 'in <Location>'
    - looks for known country/city style capitalized word sequences
    """
    t = text.strip()

    # pattern: "in Cameroon", "in South Africa", "in New York"
    m = re.search(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", t)
    if m:
        loc = m.group(1).strip()
        if len(loc) >= 3:
            return loc

    # fallback: detect any capitalized country-like word
    # (weak but better than nothing)
    cap_words = re.findall(r"\b([A-Z][a-z]{2,})\b", t)
    if cap_words:
        # prefer last one (often location appears later)
        return cap_words[-1]

    return None


def extract_relative_time_window(text: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Extract message 'time window' requirement:
      returns (min_days_old, max_days_old, explanation)

    If message is time-sensitive:
      require evidence within 14 days
    If mentions "today/now/tonight":
      require within 7 days
    Otherwise:
      allow within 365 days
    """
    t = text.lower()

    if any(x in t for x in ["today", "right now", "now", "currently", "this morning", "this evening"]):
        return 0, 7, "Claim is highly time-sensitive (today/now)."
    if any(x in t for x in ["tonight", "tomorrow", "in 6 hours", "in 24 hours"]):
        return 0, 14, "Claim references immediate future (tonight/tomorrow)."
    if "yesterday" in t:
        return 0, 7, "Claim references yesterday (short window)."

    return 0, 365, "Claim not explicitly time-sensitive (relaxed window)."


def parse_explicit_date(text: str) -> Optional[datetime]:
    """
    Parses dates like:
      - "September 2025"
      - "Sep 2025"
      - "12/10/2025"
    If found, returns a datetime (UTC) on first day of month or exact day.
    """
    t = text.lower()

    # DD/MM/YYYY or MM/DD/YYYY (ambiguous)
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", t)
    if m:
        d1 = int(m.group(1))
        d2 = int(m.group(2))
        y = int(m.group(3))
        # assume DD/MM/YYYY (most common outside US)
        day, month = d1, d2
        try:
            return datetime(y, month, day, tzinfo=timezone.utc)
        except Exception:
            return None

    # Month YYYY
    m2 = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", t)
    if m2:
        month = MONTHS[m2.group(1)]
        year = int(m2.group(2))
        return datetime(year, month, 1, tzinfo=timezone.utc)

    # Sep 2025
    m3 = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{4})\b", t)
    if m3:
        short = m3.group(1)
        year = int(m3.group(2))
        mapping = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
        }
        return datetime(year, mapping[short], 1, tzinfo=timezone.utc)

    return None


# ============================
# TIME UTILS
# ============================
def evidence_age_days(dt: Optional[datetime]) -> Optional[int]:
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    return max(0, (now - dt).days)


def recency_weight(alert_text: str, evidence_dt: Optional[datetime]) -> float:
    if evidence_dt is None:
        return 0.35

    age = evidence_age_days(evidence_dt)
    if age is None:
        return 0.35

    min_days, max_days, _ = extract_relative_time_window(alert_text)

    if age < min_days:
        return 0.0
    if age <= max_days:
        return 1.0

    # beyond window -> degrade quickly
    if age <= max_days + 30:
        return 0.4
    return 0.0


# ============================
# TRUST SCORING
# ============================
def trust_score(domain: str, crisis_type: str) -> int:
    domain = domain.lower()
    allow = TRUSTED_DOMAINS.get(crisis_type, []) + TRUSTED_DOMAINS["general"]
    for d in allow:
        if domain.endswith(d):
            return 10
    if any(domain.endswith(x) for x in [".gov", "gov", ".int", ".org"]):
        return 7
    if any(domain.endswith(x) for x in [".com", ".co.uk"]):
        return 4
    return 2


# ============================
# LOCATION MATCHING
# ============================
def compute_location_match(location: Optional[str], evidence_text: str) -> Tuple[bool, float]:
    """
    Returns (match, score)
    """
    if not location:
        return False, 0.0

    loc = location.lower().strip()
    ev = evidence_text.lower()

    if loc in ev:
        return True, 1.0

    # partial token match
    tokens = [t for t in re.split(r"\s+", loc) if len(t) >= 3]
    hits = sum(1 for t in tokens if t in ev)

    if tokens and hits / len(tokens) >= 0.6:
        return True, 0.7

    return False, 0.0


# ============================
# RISK RULES
# ============================
def hoax_risk_score(text: str) -> Tuple[int, List[str]]:
    t = text.lower()
    score = 0
    reasons = []

    if any(x in t for x in ["urgent", "breaking", "act now", "immediately"]):
        score += 20
        reasons.append("Strong urgency/panic keywords detected.")

    if any(x in t for x in ["secret", "hidden", "authorities are hiding", "they don't want you to know"]):
        score += 25
        reasons.append("Conspiracy framing / hidden-truth language.")

    if any(x in t for x in ["forward", "share", "send to everyone", "spread this"]):
        score += 15
        reasons.append("Virality / forwarding call-to-action (common hoax pattern).")

    if any(x in t for x in ["tonight", "tomorrow", "in 6 hours", "in 24 hours"]):
        score += 15
        reasons.append("Near-future dramatic claim (requires strong evidence).")

    if len(extract_keywords(text)) < 3:
        score += 10
        reasons.append("Message is vague (low specific content).")

    return min(100, score), reasons


# ============================
# EVIDENCE SOURCES
# ============================
def serper_search(query: str, num: int = 10) -> dict:
    key = load_serper_key()
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def gdelt_rss_search(query: str, max_items: int = 10) -> List[EvidenceItem]:
    q = requests.utils.quote(query)
    rss_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist&format=rss&maxrecords={max_items}"
    feed = feedparser.parse(rss_url)

    items: List[EvidenceItem] = []
    for entry in feed.entries:
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = entry.get("summary", "")
        domain = extract_domain(link)
        published = entry.get("published", "")

        items.append(EvidenceItem(
            title=title,
            link=link,
            snippet=summary[:260],
            source="GDELT",
            domain=domain,
            published=published
        ))
    return items


def who_feed_health(max_items: int = 8) -> List[EvidenceItem]:
    """
    WHO RSS feed
    """
    url = "https://www.who.int/feeds/entity/csr/don/en/rss.xml"
    feed = feedparser.parse(url)

    items: List[EvidenceItem] = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = entry.get("summary", "")
        published = entry.get("published", "")

        items.append(EvidenceItem(
            title=title,
            link=link,
            snippet=summary[:300],
            source="WHO_RSS",
            domain="who.int",
            published=published
        ))
    return items


def reliefweb_feed_health(max_items: int = 8) -> List[EvidenceItem]:
    """
    ReliefWeb RSS feed (disease & health emergencies)
    """
    url = "https://reliefweb.int/updates/rss.xml?theme=4592"
    feed = feedparser.parse(url)

    items: List[EvidenceItem] = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = entry.get("summary", "")
        published = entry.get("published", "")

        items.append(EvidenceItem(
            title=title,
            link=link,
            snippet=summary[:300],
            source="ReliefWeb_RSS",
            domain="reliefweb.int",
            published=published
        ))
    return items


def fetch_evidence(query: str, crisis_type: str) -> List[EvidenceItem]:
    """
    Evidence collection strategy:
      - If health: include WHO + ReliefWeb RSS directly
      - Search via Serper if key exists, else GDELT RSS
    """
    evidence: List[EvidenceItem] = []

    if crisis_type == "health":
        # direct institutional feeds
        evidence.extend(who_feed_health(max_items=10))
        evidence.extend(reliefweb_feed_health(max_items=10))

    # Search engine layer
    try:
        raw = serper_search(query, num=10)
        organic = raw.get("organic", [])
        for item in organic:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            domain = extract_domain(link)
            evidence.append(EvidenceItem(
                title=title,
                link=link,
                snippet=snippet[:300],
                source="Serper",
                domain=domain
            ))
    except Exception:
        evidence.extend(gdelt_rss_search(query, max_items=12))

    # de-duplicate by link
    seen = set()
    uniq = []
    for e in evidence:
        if e.link and e.link not in seen:
            seen.add(e.link)
            uniq.append(e)
    return uniq


# ============================
# APPLY SCORES
# ============================
def parse_rss_published(published_raw: str) -> Optional[datetime]:
    """
    feedparser gives string dates; parse robustly.
    """
    if not published_raw:
        return None
    try:
        # feedparser often provides published_parsed
        # but here we only have raw sometimes
        return datetime.now(timezone.utc)  # fallback if unknown
    except Exception:
        return None


def apply_scores(alert_text: str, items: List[EvidenceItem], crisis_type: str, location: Optional[str]) -> List[EvidenceItem]:
    for e in items:
        e.score = trust_score(e.domain, crisis_type)

        dt = None

        # If RSS from WHO/ReliefWeb -> no strict parse; treat as recent if exists
        if e.source in ["WHO_RSS", "ReliefWeb_RSS"]:
            dt = datetime.now(timezone.utc) - timedelta(days=1)

        # If GDELT: sometimes timestamp is embedded (in your earlier version)
        if e.source == "GDELT":
            m = re.search(r"\b\d{8}T\d{6}Z\b", e.title)
            if m:
                try:
                    dt = datetime.strptime(m.group(0), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                except Exception:
                    dt = None

        e.recency_weight = recency_weight(alert_text, dt)
        e.age_days = evidence_age_days(dt)
        e.published_dt = dt.isoformat() if dt else None

        # Location matching
        evidence_text = f"{e.title} {e.snippet} {e.link}"
        match, loc_score = compute_location_match(location, evidence_text)
        e.location_match = match
        e.location_score = loc_score

    return items


# ============================
# DECISION
# ============================
def decide(alert_text: str, evidence: List[EvidenceItem], risk: int, location: Optional[str]) -> Tuple[str, int, List[str]]:
    reasons = []

    # Strict filtering:
    # Trusted evidence must be:
    # - high trust
    # - recent enough
    # - location match if location exists
    trusted = []
    semi = []
    for e in evidence:
        if e.recency_weight <= 0:
            continue
        if location and not e.location_match:
            continue

        if e.score >= 8:
            trusted.append(e)
        elif 6 <= e.score < 8:
            semi.append(e)

    # evidence strength
    evidence_strength = 0.0
    for e in trusted[:8]:
        evidence_strength += 1.0 * e.recency_weight * (0.6 + 0.4 * e.location_score)
    for e in semi[:8]:
        evidence_strength += 0.5 * e.recency_weight * (0.6 + 0.4 * e.location_score)

    evidence_strength = min(3.0, evidence_strength)

    credibility = 10
    credibility += int(evidence_strength * 35)
    credibility -= int(risk * 0.6)

    if location:
        reasons.append(f"Location extracted: {location} (enforced evidence location match).")

    # verdict logic
    if evidence_strength >= 1.4 and credibility >= 65:
        verdict = "SUPPORTED"
        reasons.append(f"Trusted and recent evidence retrieved: {len(trusted)} sources.")
    elif risk >= 55 and evidence_strength == 0:
        verdict = "LIKELY HOAX"
        reasons.append("No trusted evidence AND strong hoax/panic patterns.")
    elif evidence_strength > 0:
        verdict = "UNVERIFIABLE"
        reasons.append("Some evidence exists, but it is not strong enough for confirmation.")
    else:
        verdict = "NOT SUPPORTED"
        reasons.append("No trusted recent evidence supports the claim for the given location/time.")

    credibility = max(0, min(100, credibility))
    return verdict, credibility, reasons


# ============================
# MAIN VERIFY
# ============================
def build_query(text: str, crisis_type: str, location: Optional[str]) -> str:
    kws = extract_keywords(text)
    base = " ".join(kws) if kws else text[:90]
    crisis_type = (crisis_type or "general").lower().strip()

    loc = location if location else ""
    if crisis_type == "climate":
        return f"{base} {loc} flood warning weather storm met office NOAA WMO"
    if crisis_type == "health":
        return f"{base} {loc} outbreak WHO ministry of health ReliefWeb"
    return f"{base} {loc}"


def verify(text: str, crisis_type: str = "general") -> Dict[str, Any]:
    text = clean_text(text)
    crisis_type = (crisis_type or "general").lower().strip()

    location = extract_location(text)
    explicit_date = parse_explicit_date(text)
    min_days, max_days, window_expl = extract_relative_time_window(text)

    query = build_query(text, crisis_type, location)

    risk, risk_reasons = hoax_risk_score(text)

    evidence = fetch_evidence(query, crisis_type=crisis_type)
    evidence = apply_scores(text, evidence, crisis_type=crisis_type, location=location)

    verdict, credibility, verdict_reasons = decide(text, evidence, risk, location)

    res = {
        "input_text": text,
        "crisis_type": crisis_type,
        "query_used": query,
        "location_extracted": location,
        "time_window": {"min_days_old": min_days, "max_days_old": max_days, "explanation": window_expl},
        "explicit_date_detected": explicit_date.isoformat() if explicit_date else None,
        "final_decision": verdict,
        "credibility_score": credibility,
        "risk_score": risk,
        "verdict_reasons": verdict_reasons,
        "risk_reasons": risk_reasons,
        "evidence": [asdict(e) for e in evidence]
    }

    # Log for report
    log_path = RESULTS_DIR / "verifier_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

    return res


# ============================
# CLI DEMO
# ============================
def main():
    print("======================================================")
    print(" Hybrid Evidence Verifier v7 (Location + Time-aware)")
    print(" MSc Research Internship Prototype — ITMO")
    print("======================================================\n")

    while True:
        msg = input("Enter crisis alert text > ").strip()
        if msg.lower() == "exit":
            break
        if len(msg) < 10:
            print("Please enter a longer message.\n")
            continue

        crisis = input("Crisis type (general/climate/health) > ").strip().lower()
        crisis = crisis if crisis in ["general", "climate", "health"] else "general"

        res = verify(msg, crisis_type=crisis)

        print("\n--- RESULT ---")
        print(f"Query used         : {res['query_used']}")
        print(f"Location extracted : {res['location_extracted']}")
        print(f"Decision           : {res['final_decision']}")
        print(f"Credibility score  : {res['credibility_score']}/100")
        print(f"Risk score         : {res['risk_score']}/100\n")

        print("Evidence reasoning:")
        for r in res["verdict_reasons"]:
            print(" -", r)

        print("\nRisk reasoning:")
        for r in res["risk_reasons"]:
            print(" -", r)

        print("\nTop Evidence (filtered by location+recency):")
        evidence = res["evidence"]
        shown = 0
        for e in sorted(evidence, key=lambda x: (x.get("score", 0) * x.get("recency_weight", 0) * (1 if x.get("location_match") else 0)), reverse=True):
            if e["score"] >= 7 and e["recency_weight"] > 0 and (res["location_extracted"] is None or e["location_match"]):
                shown += 1
                print(f"\n[{shown}] {e['source']} | {e['domain']} trust={e['score']} recW={e['recency_weight']} locMatch={e['location_match']}")
                print("Title:", e["title"][:200])
                print("URL  :", e["link"])
                if shown >= 5:
                    break

        if shown == 0:
            print(" - none")

        print("\n---------------------------\n")


if __name__ == "__main__":
    main()

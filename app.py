import streamlit as st
import json
import datetime
from collections import deque
from typing import Dict, Any, List, Optional

from evidence_verifier import verify

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Fake Alert Verifier — ITMO",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# SESSION STATE
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=30)

if "selected_result" not in st.session_state:
    st.session_state.selected_result = None

if "prefill_text" not in st.session_state:
    st.session_state.prefill_text = ""


# ======================================================
# STYLES (KEEP STREAMLIT HEADER + FIX HEADER OVERLAP)
# ======================================================
CUSTOM_CSS = """
<style>
/* IMPORTANT:
   We DO NOT hide Streamlit header.
   We just offset our header so it doesn't sit behind it.
*/

/* Add top spacing so our ITMO header never gets hidden */
.block-container {
  padding-top: 4.2rem !important;  /* this is the key fix */
  padding-bottom: 2rem;
}

hr {margin: 1.2rem 0;}

/* Header bar */
.headerbar {
  display:flex; align-items:center; gap:18px;
  padding:14px 18px;
  border:1px solid rgba(0,0,0,0.08);
  margin-bottom:18px;
  background: white;
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.04);
}
.itmo-logo {
  width:60px; height:60px;
  border-radius:14px;
  background:#111827;
  color:white;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:900;
  font-size:18px;
  letter-spacing:1px;
}
.header-title {font-size:22px; font-weight:900; margin:0;}
.header-subtitle {font-size:13px; color:#6b7280; margin-top:3px;}

/* Titles */
.h-title {font-size: 34px; font-weight: 900; margin: 0 0 2px 0;}
.h-subtitle {font-size: 14px; color: #6b7280; margin: 0 0 12px 0;}
.h-section {font-size: 18px; font-weight: 900; margin-top: 8px;}

/* Buttons alignment */
div.stButton>button {
  border-radius: 14px !important;
  padding-top: 0.85rem !important;
  padding-bottom: 0.85rem !important;
  font-weight: 900 !important;
}

/* Verdict Badges */
.badge {padding: 6px 14px; border-radius: 999px; font-weight: 900; display: inline-block; font-size: 13px; letter-spacing: 0.4px;}
.supported {background:#e7f7ee; color:#0b6b2f; border:1px solid #a6e0bd;}
.not_supported {background:#fff0f0; color:#b00020; border:1px solid #f5b5bd;}
.unverifiable {background:#fff7e6; color:#8a5a00; border:1px solid #ffd08a;}
.hoax {background:#fff0f0; color:#b00020; border:1px solid #f5b5bd;}

/* Metric Cards */
.metric-card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
  background: white;
  box-shadow: 0 8px 30px rgba(0,0,0,0.04);
}
.metric-title {font-size: 12px; color: #6b7280; font-weight: 900; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;}
.metric-value {font-size: 26px; font-weight: 900; margin: 0;}
.metric-caption {font-size: 12px; color: #6b7280; margin-top: 6px;}

/* Evidence cards */
.ev-card {
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 18px;
  padding: 14px;
  margin-bottom: 12px;
  background: #ffffff;
  box-shadow: 0 8px 30px rgba(0,0,0,0.04);
}
.ev-title {font-size: 15px; font-weight: 900; margin-bottom: 6px;}
.ev-meta {font-size: 12px; color: #6b7280; margin-bottom: 8px;}
.ev-snippet {font-size: 13px; color: #111827; line-height: 1.45;}
.ev-chip {
  display: inline-block;
  padding: 4px 10px;
  margin-right: 6px;
  border-radius: 999px;
  background: #f3f4f6;
  border: 1px solid rgba(0,0,0,0.06);
  font-size: 12px;
  font-weight: 900;
  color: #111827;
}
a {text-decoration: none;}

/* Boxes */
.tipbox {
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 16px;
  padding: 12px 14px;
  background: #f9fafb;
  color: #111827;
}
.tipbox strong {font-weight: 900;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ======================================================
# HEADER BAR (ITMO)
# ======================================================
HEADER_HTML = """
<div class="headerbar">
  <div class="itmo-logo">ITMO</div>
  <div>
    <div class="header-title">Fake Alert Verification System</div>
    <div class="header-subtitle">
      Evidence-driven decision support for crisis misinformation — MSc Research Internship Submission
    </div>
  </div>
</div>
"""
st.markdown(HEADER_HTML, unsafe_allow_html=True)


# ======================================================
# UI HELPERS
# ======================================================
def verdict_badge(verdict: str) -> str:
    v = (verdict or "").strip().upper()
    if v == "SUPPORTED":
        return '<span class="badge supported">SUPPORTED</span>'
    if v == "UNVERIFIABLE":
        return '<span class="badge unverifiable">UNVERIFIABLE</span>'
    if v == "LIKELY HOAX":
        return '<span class="badge hoax">LIKELY HOAX</span>'
    return '<span class="badge not_supported">NOT SUPPORTED</span>'


def metric_card(title: str, value: str, caption: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def short_explanation(res: Dict[str, Any]) -> str:
    v = res.get("final_decision", "NOT SUPPORTED")
    cred = res.get("credibility_score", 0)
    risk = res.get("risk_score", 0)

    if v == "SUPPORTED":
        return f"Supported by recent corroboration from trusted sources. Credibility: {cred}/100; Hoax-risk: {risk}/100."
    if v == "LIKELY HOAX":
        return f"No valid corroboration found and strong hoax-risk signals detected. Credibility: {cred}/100; Hoax-risk: {risk}/100."
    if v == "UNVERIFIABLE":
        return f"Partial/weak corroboration retrieved but insufficient to confirm. Credibility: {cred}/100."
    return f"No trusted and relevant evidence supports this claim under the current constraints. Credibility: {cred}/100."


def safe_domain(e: Dict[str, Any]) -> str:
    return e.get("domain") or "unknown-domain"


def safe_url(e: Dict[str, Any]) -> str:
    return e.get("link") or ""


def evidence_cards(evidence: List[Dict[str, Any]], location_required: Optional[str] = None):
    if not evidence:
        st.warning("No evidence sources retrieved.")
        return

    def ev_key(x: Dict[str, Any]):
        trust = float(x.get("score", 0))
        recw = float(x.get("recency_weight", 0))
        loc = 1.0 if x.get("location_match", False) else 0.65
        return trust * recw * loc

    ev_sorted = sorted(evidence, key=ev_key, reverse=True)

    for i, e in enumerate(ev_sorted[:12], start=1):
        title = (e.get("title") or "Untitled evidence").strip()
        url = safe_url(e)
        snippet = (e.get("snippet") or "").strip()
        domain = safe_domain(e)
        trust = e.get("score", 0)
        recw = e.get("recency_weight", 0)
        locm = e.get("location_match", False)
        age = e.get("age_days", None)
        src = e.get("source", "source")

        chips = [
            f"<span class='ev-chip'>Source: {src}</span>",
            f"<span class='ev-chip'>Domain: {domain}</span>",
            f"<span class='ev-chip'>Trust: {trust}/10</span>",
            f"<span class='ev-chip'>RecencyW: {recw}</span>",
        ]
        if age is not None:
            chips.append(f"<span class='ev-chip'>Age: {age}d</span>")
        if location_required:
            chips.append(f"<span class='ev-chip'>Location: {'MATCH' if locm else 'NO MATCH'}</span>")

        chips_html = " ".join(chips)
        link_html = f"<a href='{url}' target='_blank'>{title}</a>" if url else title

        st.markdown(
            f"""
            <div class="ev-card">
              <div class="ev-title">[{i}] {link_html}</div>
              <div class="ev-meta">{chips_html}</div>
              <div class="ev-snippet">{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def make_report_markdown(res: Dict[str, Any]) -> str:
    ts = res.get("timestamp", "")
    lines = [
        "# Fake Alert Verification Report",
        "",
        "## Metadata",
        f"- **Timestamp (UTC):** {ts}",
        f"- **Crisis type:** {res.get('crisis_type','')}",
        f"- **Location extracted:** {res.get('location_extracted') or 'N/A'}",
        "",
        "## Input message",
        "```",
        res.get("input_text",""),
        "```",
        "",
        "## Decision",
        f"- **Verdict:** {res.get('final_decision','')}",
        f"- **Credibility score:** {res.get('credibility_score',0)}/100",
        f"- **Hoax-risk score:** {res.get('risk_score',0)}/100",
        f"- **Evidence query:** `{res.get('query_used','')}`",
        "",
        "## Evidence reasoning",
    ]

    vr = res.get("verdict_reasons", [])
    if vr:
        lines.extend([f"- {r}" for r in vr])
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Risk reasoning")
    rr = res.get("risk_reasons", [])
    if rr:
        lines.extend([f"- {r}" for r in rr])
    else:
        lines.append("- (none)")

    return "\n".join(lines)


def generate_pdf_report(res: dict) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 18 * mm
    y = height - 20 * mm
    line_h = 6 * mm

    def write_line(text: str, bold=False, size=11):
        nonlocal y
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, text)
        y -= line_h

    def write_paragraph(text: str, size=10):
        nonlocal y
        c.setFont("Helvetica", size)
        lines = simpleSplit(text, "Helvetica", size, width - 2 * x)
        for ln in lines:
            if y < 20 * mm:
                c.showPage()
                y = height - 20 * mm
                c.setFont("Helvetica", size)
            c.drawString(x, y, ln)
            y -= (5 * mm)

    write_line("ITMO University — MSc Research Internship", bold=True, size=13)
    write_line("Fake Alert Verification Report", bold=True, size=15)
    write_line(f"Generated: {datetime.datetime.utcnow().isoformat()}Z", size=9)
    write_line("-" * 90, size=9)

    write_line("1. Input Message", bold=True)
    write_paragraph(res.get("input_text", ""))

    write_line("")
    write_line("2. Decision Summary", bold=True)
    write_line(f"Verdict: {res.get('final_decision', 'N/A')}", bold=True)
    write_line(f"Credibility Score: {res.get('credibility_score', 'N/A')}/100")
    write_line(f"Hoax Risk Score: {res.get('risk_score', 'N/A')}/100")
    write_line(f"Crisis type: {res.get('crisis_type', 'N/A')}")
    write_line(f"Extracted location: {res.get('location_extracted') or 'N/A'}")

    write_line("")
    write_line("3. Evidence Reasoning", bold=True)
    for r in res.get("verdict_reasons", [])[:12]:
        write_paragraph(f"- {r}", size=10)

    write_line("")
    write_line("4. Risk Indicators", bold=True)
    rr = res.get("risk_reasons", [])
    if rr:
        for r in rr[:12]:
            write_paragraph(f"- {r}", size=10)
    else:
        write_paragraph("- None detected", size=10)

    write_line("")
    write_line("5. Evidence Sources (Top 6)", bold=True)
    ev = res.get("evidence", [])[:6]
    if not ev:
        write_paragraph("No evidence sources retrieved.", size=10)
    else:
        for i, e in enumerate(ev, 1):
            title = (e.get("title", "") or "")[:160]
            link = e.get("link", "")
            domain = e.get("domain", "")
            score = e.get("score", 0)
            rec = e.get("recency_weight", 0)
            write_paragraph(f"{i}) {title}", size=10)
            write_paragraph(f"   Domain: {domain} | Trust={score}/10 | RecencyW={rec}", size=9)
            write_paragraph(f"   URL: {link}", size=8)
            write_line("")

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ======================================================
# MAIN TITLE
# ======================================================
st.markdown('<div class="h-title">Fake Alert Verifier</div>', unsafe_allow_html=True)
st.markdown('<div class="h-subtitle">Evidence-based verification with time-window + location enforcement</div>', unsafe_allow_html=True)


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("⚙️ Verification Settings")
crisis_type = st.sidebar.selectbox("Crisis type", ["general", "climate", "health"], index=0)
show_evidence = st.sidebar.checkbox("Show evidence cards", True)
show_history = st.sidebar.checkbox("Show clickable history", True)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Quick test messages")
if st.sidebar.button("Supported (Weather)", use_container_width=True):
    st.session_state.prefill_text = (
        "Met Office has issued an Amber warning for rain for much of south Wales on Monday, "
        "with persistent heavy rain and flooding risk."
    )
if st.sidebar.button("Likely Hoax-style", use_container_width=True):
    st.session_state.prefill_text = (
        "BREAKING! secret flood will destroy the capital tonight. authorities are hiding this. forward now!"
    )
if st.sidebar.button("Health (Cameroon)", use_container_width=True):
    st.session_state.prefill_text = (
        "COVID outbreak confirmed in Cameroon today. Everyone should wear masks in public settings."
    )


# ======================================================
# INPUT SECTION
# ======================================================
left, right = st.columns([2.2, 1], gap="large")

with left:
    st.markdown('<div class="h-section">Input Message</div>', unsafe_allow_html=True)
    text = st.text_area(
        "Paste a crisis alert / warning message",
        height=170,
        value=st.session_state.prefill_text,
        placeholder="Paste any crisis alert / warning message here..."
    )

    cols = st.columns([1, 1, 1], gap="small")
    with cols[0]:
        run = st.button("🔍 Verify", type="primary", use_container_width=True)
    with cols[1]:
        clear = st.button("Clear", use_container_width=True)
    with cols[2]:
        use_selected = st.button("Load selected", use_container_width=True)

    if clear:
        st.session_state.prefill_text = ""
        st.session_state.selected_result = None
        st.rerun()

    if use_selected and st.session_state.selected_result:
        st.session_state.prefill_text = st.session_state.selected_result.get("input_text", "")
        st.rerun()

with right:
    st.markdown('<div class="h-section">System Pipeline Summary</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tipbox">
        <strong>Verification pipeline:</strong><br/><br/>
        <pre style="margin:0;font-size:12px;line-height:1.35;">
[User Message]
      |
      v
(1) NLP Pre-processing
    - keywords
    - time window
    - location extraction
      |
      v
(2) Evidence Retrieval
    - WHO / ReliefWeb feeds (health)
    - Serper Google search (trusted domains)
      |
      v
(3) Evidence Filtering
    - trust score
    - recency weighting
    - location enforcement
      |
      v
(4) Hoax Risk Detection
    - urgency / conspiracy / virality patterns
      |
      v
(5) Final Decision
    - SUPPORTED / UNVERIFIABLE / NOT SUPPORTED / LIKELY HOAX
        </pre>
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# HISTORY
# ======================================================
if show_history:
    st.markdown("---")
    st.markdown('<div class="h-section">History (click to reopen)</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.caption("No previous verifications yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history), start=1):
            verdict = item.get("final_decision", "N/A")
            cred = item.get("credibility_score", 0)
            ts = item.get("timestamp", "unknown")
            preview = (item.get("input_text", "")[:55] + "...") if item.get("input_text") else "(no text)"

            cols = st.columns([1.2, 1, 1.2, 4, 1.2], gap="small")
            with cols[0]:
                st.markdown(verdict_badge(verdict), unsafe_allow_html=True)
            with cols[1]:
                st.caption(f"{cred}/100")
            with cols[2]:
                st.caption(ts.split("T")[0] if "T" in ts else ts)
            with cols[3]:
                st.caption(preview)
            with cols[4]:
                if st.button("Open", key=f"open_hist_{idx}"):
                    st.session_state.selected_result = item
                    st.session_state.prefill_text = item.get("input_text", "")
                    st.rerun()


# ======================================================
# LIMITATIONS + ETHICS & SAFETY
# ======================================================
with st.expander("⚠️ Limitations, Ethics & Safety", expanded=False):
    st.markdown("""
### Limitations
This prototype is designed for MSc-level research demonstration. It has known constraints:
- **Evidence availability bias:** if institutional reporting is limited, retrieval may fail even for true events.
- **Search dependency:** evidence quality depends on the indexing/search layer (Serper/GDELT + domain filtering).
- **NLP extraction limitations:** location and date extraction are heuristic-based, not full NER/temporal parsing.
- **Ambiguity handling:** vague messages without location/time context may produce conservative outcomes.
- **Not a ground-truth oracle:** the system provides *decision support*, not definitive truth.

---

### Ethics & Safety
The system is designed to reduce harm in crisis communication:
- Always prioritizes **official channels** and **trusted sources**.
- Rejects hoax-like messages when urgency + conspiracy patterns appear **without corroboration**.
- Avoids encouraging panic, retaliation, or mob action.
- Designed for **research & education** — not to replace emergency authorities.

---

### Responsible use notice
For high-stakes incidents, users should consult:
- official government emergency announcements,
- WHO / UNICEF / ReliefWeb updates,
- verified meteorological agency warnings.
""")


# ======================================================
# RUN VERIFICATION
# ======================================================
if "run" in locals() and run:
    if not text or len(text.strip()) < 10:
        st.warning("Please enter a longer message (at least 10 characters).")
    else:
        with st.spinner("Verifying… retrieving evidence, applying constraints, computing verdict…"):
            res = verify(text, crisis_type=crisis_type)
            res["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

        st.session_state.history.append(res)
        st.session_state.selected_result = res
        st.success("Verification completed.")
        st.rerun()


# ======================================================
# RESULT VIEW
# ======================================================
if st.session_state.selected_result:
    st.markdown("---")
    st.markdown('<div class="h-section">Verification Result</div>', unsafe_allow_html=True)
    res_sel = st.session_state.selected_result

    st.markdown("#### Decision")
    st.markdown(verdict_badge(res_sel.get("final_decision", "")), unsafe_allow_html=True)
    st.caption(short_explanation(res_sel))

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        metric_card("Credibility score", f"{res_sel.get('credibility_score', 0)}/100",
                    "Higher value = stronger evidence support.")
    with c2:
        metric_card("Hoax-risk score", f"{res_sel.get('risk_score', 0)}/100",
                    "Higher value = more suspicious.")
    with c3:
        metric_card("Context", str(res_sel.get("crisis_type", "general")).upper(),
                    f"Location: {res_sel.get('location_extracted') or 'N/A'}")

    st.markdown("#### Input message")
    st.code(res_sel.get("input_text", ""), language="text")

    st.markdown("#### System reasoning")
    colA, colB = st.columns(2, gap="large")
    with colA:
        st.markdown("**Evidence reasoning**")
        for r in res_sel.get("verdict_reasons", []):
            st.write("•", r)
    with colB:
        st.markdown("**Risk reasoning**")
        rr = res_sel.get("risk_reasons", [])
        if rr:
            for r in rr:
                st.write("•", r)
        else:
            st.caption("(No significant hoax-risk signals detected.)")

    if show_evidence:
        st.markdown("#### Evidence sources")
        evidence_cards(res_sel.get("evidence", []), location_required=res_sel.get("location_extracted"))

    st.markdown("---")
    st.markdown('<div class="h-section">Export</div>', unsafe_allow_html=True)

    report_md = make_report_markdown(res_sel)
    pdf_bytes = generate_pdf_report(res_sel)

    colE1, colE2, colE3 = st.columns([1.2, 1.2, 2.5], gap="small")
    with colE1:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(res_sel, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="verification_result.json",
            mime="application/json",
            use_container_width=True
        )
    with colE2:
        st.download_button(
            "⬇️ Download Markdown",
            data=report_md.encode("utf-8"),
            file_name="itmo_fakealert_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    with colE3:
        st.download_button(
            "📄 Download PDF (A4 Report)",
            data=pdf_bytes,
            file_name="itmo_fakealert_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with st.expander("🖨️ A4 Report View (Printable)", expanded=False):
        REPORT_HTML = f"""
        <div style="
            width:210mm;
            min-height:297mm;
            padding:18mm 18mm;
            margin:auto;
            background:white;
            color:#111827;
            font-family:Arial, Helvetica, sans-serif;
        ">
        <div style="display:flex; align-items:center; gap:12px;">
          <div style="width:44px;height:44px;border-radius:10px;background:#111827;color:white;
                      display:flex;align-items:center;justify-content:center;font-weight:900;">
            ITMO
          </div>
          <div>
            <div style="font-size:18px;font-weight:900;margin:0;">Fake Alert Verification Report</div>
            <div style="font-size:12px;color:#6b7280;margin-top:2px;">
              MSc Research Internship — Evidence-driven crisis alert verification
            </div>
          </div>
        </div>
        <hr style="margin:12px 0;border:none;border-top:1px solid #e5e7eb;" />
        <h3>Input Message</h3>
        <div style="border:1px solid #e5e7eb;padding:10px;border-radius:8px;white-space:pre-wrap;">
            {res_sel.get('input_text','')}
        </div>
        <h3 style="margin-top:14px;">Decision Summary</h3>
        <ul>
            <li><b>Verdict:</b> {res_sel.get('final_decision','N/A')}</li>
            <li><b>Credibility score:</b> {res_sel.get('credibility_score',0)}/100</li>
            <li><b>Hoax-risk score:</b> {res_sel.get('risk_score',0)}/100</li>
            <li><b>Crisis type:</b> {res_sel.get('crisis_type','general')}</li>
            <li><b>Location extracted:</b> {res_sel.get('location_extracted') or 'N/A'}</li>
        </ul>
        <h3>Evidence Reasoning</h3>
        <ul>
            {''.join(f"<li>{r}</li>" for r in res_sel.get('verdict_reasons', []))}
        </ul>
        <h3>Risk Indicators</h3>
        <ul>
            {''.join(f"<li>{r}</li>" for r in res_sel.get('risk_reasons', []))}
        </ul>
        </div>
        """
        st.components.v1.html(REPORT_HTML, height=1150, scrolling=True)

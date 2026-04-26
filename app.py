"""
app.py — AI That Learns You
Full Streamlit dashboard with:
  - Productivity Predictor (ML)
  - Session History
  - PDF Study Analyser
  - MCQ Quiz Generator
  - Study Technique Checker
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json, re, os
from datetime import datetime

# Optional heavy deps — gracefully degrade if not installed
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI That Learns You",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono&family=Mulish:wght@300;400;600&display=swap');
html,[class*="css"]{font-family:'Mulish',sans-serif}
h1,h2,h3{font-family:'Syne',sans-serif!important}
.section-tag{font-size:.65rem;letter-spacing:.18em;text-transform:uppercase;color:#6b6b9a;margin-bottom:.25rem}
.mcard{background:#111224;border:1px solid #1c1d35;border-radius:10px;padding:1rem 1.2rem;text-align:center}
.mcard .ml{font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;color:#6b6b9a;margin-bottom:.3rem}
.mcard .mv{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:500}
.tip{background:#0c0d1a;border-left:3px solid #5b5ef4;border-radius:0 6px 6px 0;padding:.6rem .9rem;margin:.3rem 0;font-size:.85rem;color:#a0a0cc}
.mcq-card{background:#111224;border:1px solid #1c1d35;border-radius:10px;padding:1rem;margin:.5rem 0}
.opt-row{display:flex;align-items:center;gap:.6rem;padding:.4rem .7rem;border:1px solid #1c1d35;border-radius:5px;cursor:pointer;margin:.2rem 0;font-size:.83rem}
</style>
""", unsafe_allow_html=True)

# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "models/model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_data():
    return pd.read_csv("data/study_data.csv")

model   = load_model()
df_base = load_data()
SUBJECTS = ["DSA", "OOP", "Maths", "Physics", "History"]

# ── Session state init ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# ── Claude helper ──────────────────────────────────────────────────────────────
def ask_claude(prompt: str, system: str = "") -> str:
    if not CLAUDE_AVAILABLE:
        return "⚠️ `anthropic` package not installed. Run `pip install anthropic` and set ANTHROPIC_API_KEY."
    try:
        client = anthropic.Anthropic()
        kwargs = dict(
            model   = "claude-sonnet-4-20250514",
            max_tokens = 1024,
            messages   = [{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        msg = client.messages.create(**kwargs)
        return msg.content[0].text
    except Exception as e:
        return f"Claude API error: {e}"

# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    if not PDF_SUPPORT:
        # Fallback: try reading as plain text
        try:
            return uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    import pdfplumber, io
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# ── Score helpers ──────────────────────────────────────────────────────────────
def score_color(v):
    return "#10b981" if v >= 8 else "#f59e0b" if v >= 6 else "#f43f5e"

def build_input(h, f, d, sl, subj):
    row = {"hours_studied": h, "focus_level": f, "distractions": d, "sleep_hours": sl}
    for s in SUBJECTS:
        row[f"subject_{s}"] = 1 if subj == s else 0
    return pd.DataFrame([row])

def get_tips(h, f, d, sl, score):
    t = []
    if score >= 8: t.append("🔥 Outstanding! You are in peak learning zone.")
    elif score >= 6: t.append("✅ Solid session. A few tweaks will push you higher.")
    else: t.append("⚠️ Below average. Focus on the weak factors below.")
    if sl < 6: t.append(f"😴 Under-sleeping by {6-sl:.0f} hrs. Memory needs 7–8 hrs sleep.")
    if d > 5:  t.append("📵 High distractions. Try Pomodoro: 25-min blocks, phone away.")
    if f < 5:  t.append("🎯 Low focus. Study in single-tab, quiet environment.")
    if h < 3:  t.append("⏱ Short session. One extra focused hour boosts recall significantly.")
    if h > 7 and f < 6: t.append("🧠 Long hours + low focus = diminishing returns. Take breaks.")
    return t

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AI That Learns You")
    st.caption("Study intelligence powered by ML + AI")
    st.divider()

    page = st.radio(
        "Navigate",
        ["⚡ Predict", "📊 History", "📄 PDF Analyser", "🧩 MCQ Quiz", "🔬 Technique Checker"],
        label_visibility="collapsed",
    )

    if page == "⚡ Predict":
        st.divider()
        st.markdown('<div class="section-tag">Session Inputs</div>', unsafe_allow_html=True)
        hours       = st.slider("⏱ Hours Studied",  0, 10, 4)
        focus       = st.slider("🎯 Focus Level",    1, 10, 7)
        distractions= st.slider("📵 Distractions",   0, 10, 2)
        sleep       = st.slider("😴 Sleep Hours",     0, 10, 7)
        subject     = st.selectbox("📚 Subject", SUBJECTS)
        predict_btn = st.button("⚡ Predict Productivity", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if page == "⚡ Predict":
    st.title("Productivity Predictor")
    st.caption("Random Forest ML · configure inputs in sidebar")

    if predict_btn:
        if model is None:
            st.error("Model not found. Run `python src/train.py` first.")
        else:
            df_in = build_input(hours, focus, distractions, sleep, subject)
            raw   = model.predict(df_in)[0]
            score = round(min(max(raw, 0), 10), 2)
            col   = score_color(score)
            efficiency = round(min(score / (hours + 0.01) * 2.5, 10), 1)
            grade = "A+" if score>=9 else "A" if score>=8 else "B" if score>=7 else "C" if score>=6 else "D"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="mcard"><div class="ml">Score</div><div class="mv" style="color:{col}">{score}/10</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="mcard"><div class="ml">Efficiency</div><div class="mv">{efficiency}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="mcard"><div class="ml">Grade</div><div class="mv" style="color:{col}">{grade}</div></div>', unsafe_allow_html=True)

            st.progress(score / 10)

            st.markdown("### 💡 Recommendations")
            for tip in get_tips(hours, focus, distractions, sleep, score):
                st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)

            # Subject chart
            new_row = {"hours_studied": hours, "focus_level": focus,
                       "distractions": distractions, "sleep_hours": sleep,
                       "subject": subject, "productivity": score}
            df_aug = pd.concat([df_base, pd.DataFrame([new_row])], ignore_index=True)
            sp = df_aug.groupby("subject")["productivity"].mean().sort_values()

            fig, ax = plt.subplots(figsize=(7, 3))
            fig.patch.set_facecolor("#0c0d1a"); ax.set_facecolor("#0c0d1a")
            ax.barh(sp.index, sp.values, color=[score_color(v) for v in sp.values], edgecolor="none", height=0.55)
            ax.set_xlim(0, 10); ax.tick_params(colors="#a0a0cc")
            for s in ax.spines.values(): s.set_visible(False)
            plt.tight_layout()
            st.markdown("### 📊 Subject Performance")
            st.pyplot(fig); plt.close()

            st.info(f"🔴 **Needs work:** {sp.idxmin()}   |   🟢 **Strongest:** {sp.idxmax()}")

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "subject": subject, "hours": hours, "focus": focus,
                "distractions": distractions, "sleep": sleep, "score": score,
            })
    else:
        st.info("👈 Set session inputs in the sidebar and click **Predict**.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 History":
    st.title("Session History")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(
            df_hist.style.background_gradient(subset=["score"], cmap="RdYlGn", vmin=0, vmax=10),
            use_container_width=True,
        )
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#0c0d1a"); ax.set_facecolor("#0c0d1a")
        ax.plot(df_hist["score"], marker="o", color="#5b5ef4", linewidth=2)
        ax.fill_between(range(len(df_hist)), df_hist["score"], alpha=0.12, color="#5b5ef4")
        ax.set_ylim(0, 10); ax.set_xticks(range(len(df_hist)))
        ax.set_xticklabels(df_hist["time"], color="#a0a0cc", fontsize=8)
        ax.tick_params(colors="#a0a0cc")
        for s in ax.spines.values(): s.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("No sessions recorded yet. Run a prediction first.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PDF ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📄 PDF Analyser":
    st.title("PDF Study Analyser")
    st.caption("Upload lecture notes or textbook pages — AI extracts key concepts and weak areas")

    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded:
        with st.spinner("Extracting text…"):
            st.session_state.pdf_text = extract_pdf_text(uploaded)
        st.success(f"Loaded {len(st.session_state.pdf_text):,} characters from {uploaded.name}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Analyse Document", use_container_width=True, type="primary"):
                snippet = st.session_state.pdf_text[:3500]
                prompt = (
                    f"Analyse this study document and provide:\n"
                    f"1. **Summary** (3–4 sentences)\n"
                    f"2. **Key Concepts** (bullet list of 5–8 topics)\n"
                    f"3. **Weak Areas to Focus On**\n"
                    f"4. **Suggested Study Order**\n\nDocument:\n{snippet}"
                )
                with st.spinner("Claude is analysing…"):
                    result = ask_claude(prompt, "You are a concise study assistant.")
                st.markdown(result)
        with col2:
            if st.button("🧩 Generate Quiz from this PDF", use_container_width=True):
                st.session_state.quiz_source = st.session_state.pdf_text[:2500]
                st.info("Go to **MCQ Quiz** tab — quiz source is preloaded from this PDF.")
    else:
        st.info("Upload a PDF or TXT file to begin analysis.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MCQ QUIZ
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧩 MCQ Quiz":
    st.title("MCQ Quiz Generator")
    st.caption("AI generates fresh multiple-choice questions with explanations")

    col1, col2 = st.columns([2, 1])
    with col1:
        subj_opts = SUBJECTS + ["Custom topic…"]
        q_subject = st.selectbox("Topic", subj_opts)
        if q_subject == "Custom topic…":
            default_custom = getattr(st.session_state, "quiz_source", "")
            q_custom = st.text_area("Custom topic or paste PDF content", value=default_custom, height=100)
        else:
            q_custom = ""
    with col2:
        q_count = st.selectbox("No. of Questions", [3, 5, 8])

    if st.button("⚡ Generate Questions", type="primary"):
        topic = q_custom.strip() if q_subject == "Custom topic…" and q_custom.strip() else q_subject
        prompt = (
            f"Generate exactly {q_count} multiple-choice questions about: {topic[:1500]}\n\n"
            "Return ONLY a JSON array, no markdown, no backticks:\n"
            '[{"q":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","explanation":"..."}]'
        )
        with st.spinner("Generating questions…"):
            raw = ask_claude(prompt)
        try:
            clean = re.sub(r"```json|```", "", raw).strip()
            st.session_state.quiz_data = json.loads(clean)
            for q in st.session_state.quiz_data:
                q["selected"] = None
                q["revealed"] = False
        except Exception:
            st.error("Could not parse questions — try again.")
            st.session_state.quiz_data = []

    if st.session_state.quiz_data:
        correct = 0
        submitted = all(q["revealed"] for q in st.session_state.quiz_data)

        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            answer_idx = ord(q["answer"]) - 65
            opts = q["options"]
            letters = ["A", "B", "C", "D"]

            for j, opt in enumerate(opts):
                label = opt.lstrip("ABCD. ")
                if not q["revealed"]:
                    if st.button(f"{letters[j]}. {label}", key=f"opt_{i}_{j}"):
                        st.session_state.quiz_data[i]["selected"] = j
                        st.rerun()
                else:
                    if j == answer_idx:
                        st.success(f"✅ {letters[j]}. {label}")
                    elif j == q.get("selected"):
                        st.error(f"❌ {letters[j]}. {label}")
                    else:
                        st.write(f"   {letters[j]}. {label}")

            if q["selected"] is not None and not q["revealed"]:
                st.caption(f"Selected: {letters[q['selected']]}")

            if q["revealed"]:
                if q.get("selected") == answer_idx:
                    correct += 1
                if q.get("explanation"):
                    st.info(f"💡 {q['explanation']}")
            st.divider()

        col1, col2 = st.columns([1, 1])
        with col1:
            if not submitted:
                if st.button("✓ Submit All Answers", type="primary", use_container_width=True):
                    for q in st.session_state.quiz_data:
                        q["revealed"] = True
                    st.rerun()
        if submitted:
            st.success(f"🏆 Final Score: **{correct}/{len(st.session_state.quiz_data)}**")
            if st.button("🔄 New Quiz"):
                st.session_state.quiz_data = []
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TECHNIQUE CHECKER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Technique Checker":
    st.title("Study Technique Checker")
    st.caption("Science-backed analysis of how you study, with personalised improvements")

    TECHNIQUES = {
        "🍅 Pomodoro":            {"focus":5,"retention":4,"efficiency":5,"science":4},
        "🧠 Active Recall":       {"focus":4,"retention":5,"efficiency":5,"science":5},
        "📅 Spaced Repetition":   {"focus":3,"retention":5,"efficiency":5,"science":5},
        "🗺️ Mind Mapping":        {"focus":4,"retention":4,"efficiency":3,"science":3},
        "✏️ Feynman Technique":   {"focus":5,"retention":5,"efficiency":4,"science":5},
        "📖 Passive Re-reading":  {"focus":2,"retention":2,"efficiency":2,"science":1},
    }

    selected = st.multiselect("Select your technique(s)", list(TECHNIQUES.keys()))
    routine  = st.text_area("Describe your study routine in detail", height=100,
                             placeholder="e.g. I study 3 hrs straight, re-read notes, highlight…")
    subj     = st.selectbox("Subject context", SUBJECTS, key="tech_subj")

    if st.button("🔬 Analyse My Technique", type="primary"):
        if not selected and not routine.strip():
            st.warning("Please select at least one technique or describe your routine.")
        else:
            # Ratings
            if selected:
                ratings = {k: 0 for k in ["focus","retention","efficiency","science"]}
                for t in selected:
                    for k in ratings:
                        ratings[k] += TECHNIQUES[t][k]
                n = len(selected)
                avg = {k: round(v/n) for k, v in ratings.items()}
            else:
                avg = {"focus":3,"retention":3,"efficiency":3,"science":3}

            st.markdown("### 📊 Technique Ratings")
            cols = st.columns(4)
            labels = ["Focus Quality","Memory Retention","Time Efficiency","Science Backing"]
            for i, (key, lbl) in enumerate(zip(["focus","retention","efficiency","science"], labels)):
                with cols[i]:
                    stars = "⭐" * avg[key] + "☆" * (5 - avg[key])
                    st.metric(lbl, stars)

            # Claude analysis
            tech_names = [t.split(" ",1)[1] for t in selected] if selected else ["Custom technique"]
            prompt = (
                f"A student studying {subj} uses: {', '.join(tech_names)}.\n"
                f"Routine: \"{routine.strip() or 'Not described'}\"\n\n"
                "Provide:\n1. **Overall Assessment** (2-3 sentences)\n"
                "2. **What's Working Well** (2-3 bullets)\n"
                "3. **What to Improve** (3-4 specific actionable bullets)\n"
                "4. **One Concrete Change for This Week**"
            )
            with st.spinner("Claude is evaluating…"):
                feedback = ask_claude(prompt, "You are a cognitive science expert specialising in learning. Be specific and practical.")
            st.markdown("### 🤖 AI Feedback")
            st.markdown(feedback)

    else:
        st.info("Select technique(s), describe your routine and click **Analyse**.")

"""
app.py — AI That Learns You
Run: streamlit run app.py

Fix applied: loads models/feature_columns.pkl (saved by train.py) so that
build_input() produces a DataFrame with the EXACT same column names and order
that the model was trained on — preventing the sklearn feature-name ValueError.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, re, os, sys, time, subprocess
from datetime import datetime

# ── Secrets (Streamlit Cloud) ──────────────────────────────────────────────────
if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

# ── src/ on import path ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from pdf_analyzer      import extract_text
    PDF_MODULE = True
except ImportError:
    PDF_MODULE = False

try:
    from quiz_generator    import generate_mcq
    QUIZ_MODULE = True
except ImportError:
    QUIZ_MODULE = False

try:
    from technique_checker import get_ratings, TECHNIQUE_RATINGS
    TECH_MODULE = True
except ImportError:
    TECH_MODULE = False

try:
    from rl_planner        import build_plan, summarise_plan, get_strategy_notes
    RL_MODULE = True
except ImportError:
    RL_MODULE = False

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
.tip{
    background:#0c0d1a;
    border-left:3px solid #5b5ef4;
    border-radius:0 6px 6px 0;
    padding:.6rem .9rem;
    margin:.3rem 0;
    font-size:.85rem;
    color:#a0a0cc;
}
.mcard{
    background:#111224;
    border:1px solid #1c1d35;
    border-radius:10px;
    padding:1rem 1.2rem;
    text-align:center;
}
.mcard .ml{
    font-size:.65rem;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:#6b6b9a;
    margin-bottom:.3rem;
}
.mcard .mv{
    font-family:'JetBrains Mono',monospace;
    font-size:1.9rem;
    font-weight:500;
}
</style>
""", unsafe_allow_html=True)

# ── Auto-train if model missing (Streamlit Cloud first run) ────────────────────
def ensure_model():
    if not os.path.exists("models/model.pkl"):
        os.makedirs("models", exist_ok=True)
        with st.spinner("⚙️ First run — training model (takes ~10 sec)…"):
            result = subprocess.run(
                [sys.executable, "src/train.py"],
                capture_output=True, text=True
            )
        if result.returncode != 0:
            st.error(f"Training failed:\n{result.stderr}")
            st.stop()
        st.success("✅ Model trained successfully!")
        st.rerun()

ensure_model()

# ── Load model + feature columns ──────────────────────────────────────────────
@st.cache_resource
def load_model_and_features():
    """
    Load both the model AND the saved feature column list.
    The column list guarantees build_input() produces identical features.
    """
    model = joblib.load("models/model.pkl")

    feat_path = "models/feature_columns.pkl"
    if os.path.exists(feat_path):
        feature_columns = joblib.load(feat_path)
    else:
        # Fallback: reconstruct from metrics.json if it exists
        metrics_path = "models/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                feature_columns = json.load(f).get("features", [])
        else:
            feature_columns = []

    return model, feature_columns

@st.cache_data
def load_data():
    return pd.read_csv("data/study_data.csv")

model, FEATURE_COLUMNS = load_model_and_features()
df_base = load_data()
SUBJECTS = ["DSA", "OOP", "Maths", "Physics", "History"]

# Session state
for k, v in [("history", []), ("quiz_data", []), ("pdf_text", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def sc(v):
    return "#10b981" if v >= 8 else "#f59e0b" if v >= 6 else "#f43f5e"

def grade(v):
    return "A+" if v >= 9 else "A" if v >= 8 else "B" if v >= 7 else "C" if v >= 6 else "D"

def build_input(h, f, d, sl, subj):
    """
    Build a prediction DataFrame that EXACTLY matches the columns the model
    was trained on. Uses FEATURE_COLUMNS loaded from models/feature_columns.pkl.
    Falls back to manual construction if columns list is empty.
    """
    if FEATURE_COLUMNS:
        # Build a row of zeros, then fill in the known values
        row = {col: 0 for col in FEATURE_COLUMNS}
        row["hours_studied"]  = h
        row["focus_level"]    = f
        row["distractions"]   = d
        row["sleep_hours"]    = sl
        col_name = f"subject_{subj}"
        if col_name in row:
            row[col_name] = 1
        # Return with columns in the exact same order as training
        return pd.DataFrame([row])[FEATURE_COLUMNS]
    else:
        # Fallback: manual construction (works if subjects match CSV)
        row = {"hours_studied": h, "focus_level": f,
               "distractions": d, "sleep_hours": sl}
        for s in SUBJECTS:
            row[f"subject_{s}"] = 1 if subj == s else 0
        return pd.DataFrame([row])

def get_tips(h, f, d, sl, score):
    t = []
    if score >= 8:     t.append("🔥 Outstanding session! You are in peak learning zone.")
    elif score >= 6:   t.append("✅ Solid session. Small tweaks will push you higher.")
    else:              t.append("⚠️ Below average. Address the weak factors below.")
    if sl < 6:         t.append(f"😴 Under-sleeping by {6-sl:.0f} hrs. Memory needs 7–8 hrs sleep.")
    if d > 5:          t.append("📵 High distractions. Try Pomodoro: 25-min blocks, phone away.")
    if f < 5:          t.append("🎯 Low focus. Study in a quiet environment, one tab open.")
    if h < 3:          t.append("⏱ Short session. One extra focused hour boosts recall significantly.")
    if h > 7 and f < 6: t.append("🧠 Long hours + low focus = diminishing returns. Take a break.")
    return t

def dark_chart(data, title):
    sp = pd.Series(data).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#0c0d1a")
    ax.set_facecolor("#0c0d1a")
    ax.barh(sp.index, sp.values, color=[sc(v) for v in sp.values],
            edgecolor="none", height=0.55)
    ax.set_xlim(0, 10)
    ax.set_title(title, color="#e0e0ff", fontsize=9)
    ax.tick_params(colors="#a0a0cc")
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    return fig

def ask_claude(prompt, system=""):
    if not CLAUDE_AVAILABLE:
        return "⚠️ Install `anthropic` and set ANTHROPIC_API_KEY to use AI features."
    try:
        client = anthropic.Anthropic()
        kw = dict(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kw["system"] = system
        return client.messages.create(**kw).content[0].text
    except Exception as e:
        return f"Claude API error: {e}"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AI That Learns You")
    st.caption("Study intelligence · ML + AI")
    st.divider()

    page = st.radio(
        "Navigate",
        ["⚡ Predict", "📊 History", "📄 PDF Analyser",
         "🧩 MCQ Quiz", "🔬 Technique Checker", "🤖 RL Planner"],
        label_visibility="collapsed",
    )

    if page == "⚡ Predict":
        st.divider()
        st.markdown("**Session Inputs**")
        hours        = st.slider("⏱ Hours Studied", 0, 10, 4)
        focus        = st.slider("🎯 Focus Level",   1, 10, 7)
        distractions = st.slider("📵 Distractions",  0, 10, 2)
        sleep        = st.slider("😴 Sleep Hours",    0, 10, 7)
        subject      = st.selectbox("📚 Subject", SUBJECTS)
        predict_btn  = st.button("⚡ Predict Productivity",
                                  use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════════════════════
# ⚡  PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "⚡ Predict":
    st.title("Productivity Predictor")
    st.caption("Random Forest ML · configure session in sidebar")

    if predict_btn:
        df_in = build_input(hours, focus, distractions, sleep, subject)
        try:
            score = round(float(np.clip(model.predict(df_in)[0], 0, 10)), 2)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Try deleting `models/model.pkl` and running `python src/train.py` again.")
            st.stop()

        eff = round(min(score / (hours + 0.01) * 2.5, 10), 1)
        col_hex = sc(score)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="mcard"><div class="ml">Score</div>'
                f'<div class="mv" style="color:{col_hex}">{score}/10</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="mcard"><div class="ml">Efficiency</div>'
                f'<div class="mv">{eff}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="mcard"><div class="ml">Grade</div>'
                f'<div class="mv" style="color:{col_hex}">{grade(score)}</div></div>',
                unsafe_allow_html=True,
            )

        st.progress(score / 10)

        st.markdown("### 💡 Recommendations")
        for tip in get_tips(hours, focus, distractions, sleep, score):
            st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)

        # Subject performance chart
        new_row = {
            "hours_studied": hours, "focus_level": focus,
            "distractions": distractions, "sleep_hours": sleep,
            "subject": subject, "productivity": score,
        }
        df_aug = pd.concat([df_base, pd.DataFrame([new_row])], ignore_index=True)
        sp = df_aug.groupby("subject")["productivity"].mean()

        st.markdown("### 📊 Subject Performance")
        st.pyplot(dark_chart(sp.to_dict(), "Average Productivity by Subject"))
        plt.close()
        st.info(f"🔴 **Needs work:** {sp.idxmin()}   |   🟢 **Strongest:** {sp.idxmax()}")

        # Save to session history
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "subject": subject, "hours": hours, "focus": focus,
            "distractions": distractions, "sleep": sleep, "score": score,
        })
    else:
        st.info("👈 Set your session inputs in the sidebar and click **Predict**.")

# ═══════════════════════════════════════════════════════════════════════════════
# 📊  HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 History":
    st.title("Session History")
    if st.session_state.history:
        df_h = pd.DataFrame(st.session_state.history)
        st.dataframe(
            df_h.style.background_gradient(subset=["score"], cmap="RdYlGn", vmin=0, vmax=10),
            use_container_width=True,
        )
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#0c0d1a"); ax.set_facecolor("#0c0d1a")
        ax.plot(df_h["score"], marker="o", color="#5b5ef4", linewidth=2)
        ax.fill_between(range(len(df_h)), df_h["score"], alpha=0.12, color="#5b5ef4")
        ax.set_ylim(0, 10)
        ax.set_xticks(range(len(df_h)))
        ax.set_xticklabels(df_h["time"], color="#a0a0cc", fontsize=8)
        ax.tick_params(colors="#a0a0cc")
        for s in ax.spines.values(): s.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("No sessions recorded yet. Run a prediction first.")

# ═══════════════════════════════════════════════════════════════════════════════
# 📄  PDF ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 PDF Analyser":
    st.title("PDF Study Analyser")
    st.caption("Upload notes or a textbook — AI extracts key concepts and weak areas")

    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded:
        with st.spinner("Extracting text…"):
            if PDF_MODULE:
                st.session_state.pdf_text = extract_text(uploaded)
            else:
                raw = uploaded.read()
                st.session_state.pdf_text = (
                    raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
                )
        n = len(st.session_state.pdf_text)
        if n:
            st.success(f"✅ {n:,} characters extracted from `{uploaded.name}`")
        else:
            st.warning("⚠️ Could not extract text. Try a .txt file or a text-based PDF.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Analyse Document", use_container_width=True, type="primary",
                     disabled=not bool(st.session_state.pdf_text)):
            with st.spinner("Claude is analysing…"):
                result = ask_claude(
                    f"Analyse this study document and provide:\n"
                    f"1. **Summary** (3–4 sentences)\n"
                    f"2. **Key Concepts** (5–8 bullet points)\n"
                    f"3. **Weak Areas to Focus On**\n"
                    f"4. **Suggested Study Order**\n\n"
                    f"Document:\n{st.session_state.pdf_text[:3500]}",
                    "You are a concise study assistant.",
                )
            st.markdown(result)
    with c2:
        if st.button("🧩 Generate Quiz from PDF", use_container_width=True,
                     disabled=not bool(st.session_state.pdf_text)):
            st.session_state.quiz_source = st.session_state.pdf_text[:2500]
            st.success("✅ PDF loaded as quiz source → switch to **MCQ Quiz** tab")

    if not st.session_state.pdf_text:
        st.info("Upload a PDF or TXT file to begin.")

# ═══════════════════════════════════════════════════════════════════════════════
# 🧩  MCQ QUIZ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧩 MCQ Quiz":
    st.title("MCQ Quiz Generator")
    st.caption("AI generates fresh multiple-choice questions with explanations")

    c1, c2 = st.columns([2, 1])
    with c1:
        qsubj = st.selectbox("Topic", SUBJECTS + ["Custom…"])
        qcustom = ""
        if qsubj == "Custom…":
            qcustom = st.text_area(
                "Custom topic or paste PDF content",
                value=getattr(st.session_state, "quiz_source", ""),
                height=90,
            )
    with c2:
        qcount = st.selectbox("No. of Questions", [3, 5, 8])

    if st.button("⚡ Generate Questions", type="primary"):
        topic = qcustom.strip() if qsubj == "Custom…" and qcustom.strip() else qsubj
        with st.spinner("Generating questions…"):
            if QUIZ_MODULE:
                try:
                    st.session_state.quiz_data = generate_mcq(topic[:1500], qcount)
                    for q in st.session_state.quiz_data:
                        q["selected"] = None
                        q["revealed"] = False
                except Exception as e:
                    st.error(str(e))
                    st.session_state.quiz_data = []
            else:
                raw = ask_claude(
                    f"Generate exactly {qcount} MCQs about: {topic[:1500]}\n"
                    "Return ONLY a JSON array, no markdown:\n"
                    '[{"q":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","explanation":"..."}]'
                )
                try:
                    st.session_state.quiz_data = json.loads(
                        re.sub(r"```json|```", "", raw).strip()
                    )
                    for q in st.session_state.quiz_data:
                        q["selected"] = None
                        q["revealed"] = False
                except Exception:
                    st.error("Could not parse questions — try again.")
                    st.session_state.quiz_data = []

    if st.session_state.quiz_data:
        done = all(q.get("revealed") for q in st.session_state.quiz_data)
        correct = 0

        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            answer_idx = ord(q["answer"].upper()) - 65
            letters = ["A", "B", "C", "D"]

            for j, opt in enumerate(q["options"]):
                label = opt.lstrip("ABCD. ")
                if not q.get("revealed"):
                    if st.button(f"{letters[j]}. {label}", key=f"opt_{i}_{j}"):
                        st.session_state.quiz_data[i]["selected"] = j
                        st.rerun()
                else:
                    if j == answer_idx:        st.success(f"✅ {letters[j]}. {label}")
                    elif j == q.get("selected"): st.error(f"❌ {letters[j]}. {label}")
                    else:                        st.write(f"   {letters[j]}. {label}")

            if q.get("selected") is not None and not q.get("revealed"):
                st.caption(f"Selected: {letters[q['selected']]}")
            if q.get("revealed"):
                if q.get("selected") == answer_idx:
                    correct += 1
                if q.get("explanation"):
                    st.info(f"💡 {q['explanation']}")
            st.divider()

        if not done:
            if st.button("✓ Submit All Answers", type="primary"):
                for q in st.session_state.quiz_data:
                    q["revealed"] = True
                st.rerun()
        else:
            st.success(f"🏆 Final Score: **{correct}/{len(st.session_state.quiz_data)}**")
            if st.button("🔄 New Quiz"):
                st.session_state.quiz_data = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 🔬  TECHNIQUE CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Technique Checker":
    st.title("Study Technique Checker")
    st.caption("Science-backed ratings + AI personalised feedback")

    TMAP = {
        "🍅 Pomodoro":           "Pomodoro",
        "🧠 Active Recall":      "Active Recall",
        "📅 Spaced Repetition":  "Spaced Repetition",
        "🗺️ Mind Mapping":       "Mind Mapping",
        "✏️ Feynman Technique":  "Feynman Technique",
        "📖 Passive Re-reading": "Passive Re-reading",
    }
    FALLBACK = {
        "Pomodoro":           {"focus":5,"retention":4,"efficiency":5,"science":4},
        "Active Recall":      {"focus":4,"retention":5,"efficiency":5,"science":5},
        "Spaced Repetition":  {"focus":3,"retention":5,"efficiency":5,"science":5},
        "Mind Mapping":       {"focus":4,"retention":4,"efficiency":3,"science":3},
        "Feynman Technique":  {"focus":5,"retention":5,"efficiency":4,"science":5},
        "Passive Re-reading": {"focus":2,"retention":2,"efficiency":2,"science":1},
    }

    sel_d   = st.multiselect("Select your technique(s)", list(TMAP.keys()))
    sel     = [TMAP[k] for k in sel_d]
    routine = st.text_area("Describe your study routine in detail", height=100,
                            placeholder="e.g. I study 3 hrs straight, re-read notes, highlight…")
    subj    = st.selectbox("Subject context", SUBJECTS, key="tech_subj")

    if st.button("🔬 Analyse My Technique", type="primary"):
        if not sel and not routine.strip():
            st.warning("Please select at least one technique or describe your routine.")
        else:
            if TECH_MODULE:
                r = get_ratings(sel)
            elif sel:
                tot = {"focus":0,"retention":0,"efficiency":0,"science":0}
                for t in sel:
                    rv = FALLBACK.get(t, {"focus":3,"retention":3,"efficiency":3,"science":3})
                    for k in tot: tot[k] += rv[k]
                r = {k: round(v/len(sel), 1) for k, v in tot.items()}
            else:
                r = {"focus":3,"retention":3,"efficiency":3,"science":3}

            st.markdown("### 📊 Technique Ratings")
            cols = st.columns(4)
            for col, key, lbl in zip(cols,
                ["focus","retention","efficiency","science"],
                ["Focus","Retention","Efficiency","Science"]):
                with col:
                    st.metric(lbl, "⭐" * round(r[key]) + "☆" * (5 - round(r[key])))

            tech_names = sel if sel else ["Custom technique"]
            prompt = (
                f"Student studying {subj} uses: {', '.join(tech_names)}.\n"
                f"Routine: \"{routine.strip() or 'Not described'}\"\n\n"
                "Provide:\n1. **Overall Assessment** (2-3 sentences)\n"
                "2. **What's Working Well** (2-3 bullets)\n"
                "3. **What to Improve** (3-4 actionable bullets)\n"
                "4. **One Concrete Change This Week**"
            )
            with st.spinner("Claude is evaluating…"):
                feedback = ask_claude(prompt,
                    "You are a cognitive science expert. Be specific and practical.")
            st.markdown("### 🤖 AI Feedback")
            st.markdown(feedback)
    else:
        st.info("Select technique(s), describe your routine and click **Analyse**.")

# ═══════════════════════════════════════════════════════════════════════════════
# 🤖  RL PLANNER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 RL Planner":
    st.title("RL Study Planner")
    st.caption("ε-greedy Reinforcement Learning agent builds your optimal weekly schedule")

    c1, c2, c3 = st.columns(3)
    with c1: rh = st.slider("⏱ Available Hours / Day", 1, 10, 3)
    with c2: rd = st.slider("📅 Days Until Exam",       1, 30, 7)
    with c3: rp = st.selectbox("🎯 Priority Subject", ["Auto"] + SUBJECTS)

    if st.button("🤖 Generate Plan", type="primary"):
        with st.spinner("RL agent computing optimal plan…"):
            time.sleep(0.6)

        if RL_MODULE:
            plan = build_plan(rh, rd, rp)
            sm   = summarise_plan(plan)
        else:
            import random
            IC = {"DSA":"💻","OOP":"🧱","Maths":"📐","Physics":"⚛️","History":"📜"}
            Q  = {"DSA":0.70,"OOP":0.80,"Maths":0.55,"Physics":0.60,"History":0.50}
            if rp != "Auto" and rp in Q:
                Q[rp] = min(1.0, Q[rp] * 1.5)
            td  = min(rd, 7)
            spd = max(1, round(rh / 1.5))
            hps = round(rh / spd, 1)
            plan = []
            for d in range(td):
                dq = dict(Q); slots = []
                for _ in range(spd):
                    s = max(dq, key=dq.__getitem__) if random.random() > 0.1 else random.choice(SUBJECTS)
                    slots.append({"subject": s, "hours": hps,
                                  "intensity": min(5, max(1, round(hps * 1.8))),
                                  "icon": IC.get(s, "📚")})
                    dq[s] *= 0.72
                pred = round(min(10, sum(x["intensity"] for x in slots) / len(slots) * 2 + 4), 1)
                plan.append({"day": d+1,
                              "label": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d % 7],
                              "slots": slots, "total_hours": rh, "predicted_score": pred})
            sh = {}
            for day in plan:
                for s in day["slots"]:
                    sh[s["subject"]] = sh.get(s["subject"], 0) + s["hours"]
            sm = {"total_hours": round(rh * len(plan), 1),
                  "days_planned": len(plan),
                  "top_subject":  max(sh, key=sh.__getitem__) if sh else "DSA",
                  "subject_hours": sh}

        # Stats row
        c1, c2, c3, c4 = st.columns(4)
        for col, lb, val in zip(
            [c1, c2, c3, c4],
            ["Total Hours", "Days Planned", "Top Subject", "Avg / Day"],
            [f"{sm['total_hours']}h", sm['days_planned'], sm['top_subject'], f"{rh}h"],
        ):
            with col:
                st.markdown(
                    f'<div class="mcard"><div class="ml">{lb}</div>'
                    f'<div class="mv" style="font-size:1.3rem">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### 📅 Day-by-Day Plan")
        for day in plan:
            with st.expander(
                f"Day {day['day']} · {day['label']}  —  "
                f"Predicted score: {day['predicted_score']}/10", expanded=True
            ):
                cols = st.columns(len(day["slots"]))
                for col, slot in zip(cols, day["slots"]):
                    with col:
                        st.markdown(
                            f"**{slot['icon']} {slot['subject']}**  \n"
                            f"{slot['hours']}h · intensity {slot['intensity']}/5"
                        )
                        st.progress(slot["intensity"] / 5)

        st.markdown("### 🧠 Agent Strategy Notes")
        with st.spinner("Getting strategy from Claude…"):
            if RL_MODULE:
                notes = get_strategy_notes(plan, rh, rd, rp)
            else:
                sh_s = ", ".join(f"{s} {h:.1f}h" for s, h in sm["subject_hours"].items())
                notes = ask_claude(
                    f"Student has {rd} days to exam, studies {rh}h/day, priority {rp}. "
                    f"Plan: {sh_s}. Write exactly 3 bullet strategy tips. Max 120 words.",
                    "Concise study planner AI. Bullet points only.",
                )
        st.markdown(notes)
    else:
        st.info("Set your availability and click **Generate Plan** to get your RL-optimised schedule.")

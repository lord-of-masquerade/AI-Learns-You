"""
app.py — AI That Learns You
Styled to match demo.html as closely as Streamlit allows.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, re, os, sys, time, subprocess, random
from datetime import datetime

# ── Streamlit Cloud secrets ────────────────────────────────────────────────────
if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from pdf_analyzer      import extract_text;    PDF_MODULE  = True
except ImportError:
    PDF_MODULE = False
try:
    from quiz_generator    import generate_mcq;    QUIZ_MODULE = True
except ImportError:
    QUIZ_MODULE = False
try:
    from technique_checker import get_ratings;     TECH_MODULE = True
except ImportError:
    TECH_MODULE = False
try:
    from rl_planner import build_plan, summarise_plan, get_strategy_notes; RL_MODULE = True
except ImportError:
    RL_MODULE = False
try:
    import anthropic; CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI That Learns You",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — makes Streamlit look like the demo.html design
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&family=Mulish:wght@300;400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    background-color: #070810 !important;
    font-family: 'Mulish', sans-serif !important;
    color: #eeeeff !important;
}
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; letter-spacing:-.01em; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0d1a !important;
    border-right: 1px solid #1c1d35 !important;
}
[data-testid="stSidebar"] * { color: #eeeeff !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #6b6b9a !important; font-size:.75rem; letter-spacing:.15em; text-transform:uppercase; }
[data-testid="stSidebar"] h2 { font-family:'Syne',sans-serif !important; font-size:1.05rem !important; color:#eeeeff !important; }

/* Sidebar nav radio — make it look like nav pills */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    gap: .6rem !important;
    padding: .5rem .8rem !important;
    border-radius: 7px !important;
    cursor: pointer !important;
    color: #a0a0cc !important;
    font-size: .84rem !important;
    transition: all .15s !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #111224 !important;
    color: #eeeeff !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div + div {
    background: #111224 !important;
    border-color: #1c1d35 !important;
    color: #eeeeff !important;
}
/* Hide the actual radio dot */
[data-testid="stSidebar"] [data-baseweb="radio"] [class*="indicator"] { display:none !important; }

/* ── Sliders — purple thumb, thin track ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #5b5ef4 !important;
    box-shadow: 0 0 0 3px rgba(91,94,244,.25) !important;
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
    border: none !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [class*="Track"] {
    height: 3px !important;
    border-radius: 2px !important;
}
/* Filled portion purple */
[data-testid="stSlider"] [data-baseweb="slider"] [class*="Track"][style*="width"] {
    background: #5b5ef4 !important;
}
[data-testid="stSlider"] div[class*="tickBar"] { display: none !important; }
[data-testid="stSlider"] p { color: #a0a0cc !important; font-size:.7rem !important; letter-spacing:.1em; text-transform:uppercase; }
/* Slider value number — larger and white */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .92rem !important;
    color: #eeeeff !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ── Primary button — purple gradient ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #5b5ef4, #4338ca) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: .85rem !important;
    letter-spacing: .04em !important;
    border-radius: 8px !important;
    padding: .6rem 1.4rem !important;
    transition: all .2s !important;
    box-shadow: none !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(91,94,244,.4) !important;
}
/* Secondary / regular button */
.stButton > button {
    background: #111224 !important;
    border: 1px solid #1c1d35 !important;
    color: #a0a0cc !important;
    border-radius: 7px !important;
    font-family: 'Mulish', sans-serif !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    border-color: #5b5ef4 !important;
    color: #eeeeff !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #111224 !important;
    border: 1px solid #1c1d35 !important;
    border-radius: 7px !important;
    color: #eeeeff !important;
}
[data-testid="stSelectbox"] svg { fill: #6b6b9a !important; }
[data-baseweb="popover"] [data-baseweb="menu"] {
    background: #111224 !important;
    border: 1px solid #1c1d35 !important;
}
[data-baseweb="popover"] li { color: #a0a0cc !important; }
[data-baseweb="popover"] li:hover { background: #1c1d35 !important; color:#eeeeff !important; }

/* ── Multiselect ── */
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background: #111224 !important;
    border: 1px solid #1c1d35 !important;
    border-radius: 7px !important;
}
[data-baseweb="tag"] { background: rgba(91,94,244,.15) !important; border: 1px solid rgba(91,94,244,.3) !important; }
[data-baseweb="tag"] span { color: #a0a0cc !important; }

/* ── Text area ── */
[data-testid="stTextArea"] textarea {
    background: #111224 !important;
    border: 1px solid #1c1d35 !important;
    border-radius: 7px !important;
    color: #eeeeff !important;
    font-family: 'Mulish', sans-serif !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #5b5ef4 !important;
    box-shadow: 0 0 0 2px rgba(91,94,244,.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background: #111224 !important;
    border: 2px dashed #1c1d35 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] section:hover { border-color: #5b5ef4 !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: #111224 !important;
    border: 1px solid #1c1d35 !important;
    border-radius: 10px !important;
    padding: .8rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: #6b6b9a !important; font-size:.68rem !important; letter-spacing:.12em !important; text-transform:uppercase !important; }
[data-testid="stMetricValue"] { color: #eeeeff !important; font-family:'JetBrains Mono',monospace !important; }

/* ── Progress bar — purple ── */
[data-testid="stProgress"] > div > div {
    background: rgba(91,94,244,.18) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg,#5b5ef4,#00d9c0) !important;
    border-radius: 4px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1c1d35 !important; border-radius: 10px !important; overflow:hidden !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111224 !important;
    border: 1px solid #1c1d35 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #eeeeff !important; font-family:'Syne',sans-serif !important; }

/* ── Info / Success / Warning / Error boxes ── */
[data-testid="stAlert"] { border-radius: 8px !important; }
.stAlert[data-baseweb="notification"][kind="info"]    { background: rgba(91,94,244,.08) !important; border-left: 3px solid #5b5ef4 !important; }
.stAlert[data-baseweb="notification"][kind="success"] { background: rgba(16,185,129,.08) !important; border-left: 3px solid #10b981 !important; }
.stAlert[data-baseweb="notification"][kind="warning"] { background: rgba(245,158,11,.08) !important; border-left: 3px solid #f59e0b !important; }
.stAlert[data-baseweb="notification"][kind="error"]   { background: rgba(244,63,94,.08) !important;  border-left: 3px solid #f43f5e !important; }

/* ── Divider ── */
hr { border-color: #1c1d35 !important; }

/* ── Custom component classes ── */
.page-head { border-bottom: 1px solid #1c1d35; padding-bottom: .9rem; margin-bottom: 1.4rem; }
.page-head h1 { font-size: 1.7rem !important; margin-bottom:.25rem; }
.page-head p  { font-size: .78rem; color: #6b6b9a; margin:0; }

.panel {
    background: #0f1020;
    border: 1px solid #1c1d35;
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
}
.panel-label {
    font-size: .63rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #6b6b9a;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: .4rem;
}

.mcard {
    background: #111224;
    border: 1px solid #1c1d35;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.mcard .ml {
    font-size:.63rem; letter-spacing:.12em; text-transform:uppercase;
    color:#6b6b9a; margin-bottom:.35rem;
}
.mcard .mv {
    font-family:'JetBrains Mono',monospace;
    font-size:2rem; font-weight:500;
}

.tip {
    background: #0c0d1a;
    border-left: 3px solid #5b5ef4;
    border-radius: 0 7px 7px 0;
    padding: .65rem .9rem;
    margin: .4rem 0;
    font-size: .84rem;
    color: #a0a0cc;
}

.ph-box {
    background: #0f1020;
    border: 1px dashed #1c1d35;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    color: #6b6b9a;
    font-size: .82rem;
    line-height: 1.7;
}
.ph-box .ph-icon { font-size: 2.4rem; margin-bottom: .6rem; }

.subj-pill-row { display:flex; gap:.45rem; flex-wrap:wrap; margin-top:.35rem; }
.subj-pill {
    background: #111224;
    border: 1px solid #1c1d35;
    border-radius: 6px;
    padding: .35rem .8rem;
    font-size: .78rem;
    color: #a0a0cc;
    cursor: pointer;
    transition: all .15s;
    font-family: 'Mulish', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── Auto-train on Streamlit Cloud first run ────────────────────────────────────
def ensure_model():
    if not os.path.exists("models/model.pkl"):
        os.makedirs("models", exist_ok=True)
        with st.spinner("⚙️ First run — training model (~10 sec)…"):
            r = subprocess.run([sys.executable, "src/train.py"],
                               capture_output=True, text=True)
        if r.returncode != 0:
            st.error(f"Training failed:\n{r.stderr}"); st.stop()
        st.rerun()

ensure_model()

# ── Load ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_features():
    mdl  = joblib.load("models/model.pkl")
    feat = (joblib.load("models/feature_columns.pkl")
            if os.path.exists("models/feature_columns.pkl") else [])
    if not feat and os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            feat = json.load(f).get("features", [])
    return mdl, feat

@st.cache_data
def load_data():
    return pd.read_csv("data/study_data.csv")

model, FEATURE_COLUMNS = load_model_and_features()
df_base  = load_data()
SUBJECTS = ["DSA","OOP","Maths","Physics","History"]

for k, v in [("history",[]),("quiz_data",[]),("pdf_text",""),("active_subj","DSA")]:
    if k not in st.session_state: st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def sc(v): return "#10b981" if v>=8 else "#f59e0b" if v>=6 else "#f43f5e"
def grade(v): return "A+" if v>=9 else "A" if v>=8 else "B" if v>=7 else "C" if v>=6 else "D"

def build_input(h,f,d,sl,subj):
    if FEATURE_COLUMNS:
        row={c:0 for c in FEATURE_COLUMNS}
        row["hours_studied"]=h; row["focus_level"]=f
        row["distractions"]=d; row["sleep_hours"]=sl
        k=f"subject_{subj}"
        if k in row: row[k]=1
        return pd.DataFrame([row])[FEATURE_COLUMNS]
    row={"hours_studied":h,"focus_level":f,"distractions":d,"sleep_hours":sl}
    for s in SUBJECTS: row[f"subject_{s}"]=1 if subj==s else 0
    return pd.DataFrame([row])

def get_tips(h,f,d,sl,score):
    t=[]
    if score>=8: t.append("🔥 Outstanding session! You're in peak learning zone.")
    elif score>=6: t.append("✅ Solid session. Small tweaks will push you higher.")
    else: t.append("⚠️ Below average. Address the weak factors below.")
    if sl<6: t.append(f"😴 Under-sleeping by {6-sl:.0f} hrs. Memory needs 7–8 hrs.")
    if d>5: t.append("📵 High distractions. Try Pomodoro: 25-min blocks, phone away.")
    if f<5: t.append("🎯 Low focus. Study in a quiet environment, one tab open.")
    if h<3: t.append("⏱ Short session. One extra focused hour boosts recall significantly.")
    if h>7 and f<6: t.append("🧠 Long hours + low focus = diminishing returns. Take a break.")
    return t

def dark_chart(data,title):
    sp=pd.Series(data).sort_values()
    fig,ax=plt.subplots(figsize=(6,3))
    fig.patch.set_facecolor("#0c0d1a"); ax.set_facecolor("#0c0d1a")
    ax.barh(sp.index,sp.values,color=[sc(v) for v in sp.values],edgecolor="none",height=0.55)
    ax.set_xlim(0,10); ax.set_title(title,color="#e0e0ff",fontsize=9)
    ax.tick_params(colors="#a0a0cc")
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout(); return fig

def ask_claude(prompt,system=""):
    if not CLAUDE_AVAILABLE:
        return "⚠️ Set ANTHROPIC_API_KEY and install `anthropic` to use AI features."
    try:
        client=anthropic.Anthropic()
        kw=dict(model="claude-sonnet-4-20250514",max_tokens=1024,
                messages=[{"role":"user","content":prompt}])
        if system: kw["system"]=system
        return client.messages.create(**kw).content[0].text
    except Exception as e:
        return f"Claude API error: {e}"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:.2rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;color:#eeeeff;line-height:1.3">
        <span style="color:#5b5ef4">AI That</span> Learns You
      </div>
      <div style="font-size:.62rem;letter-spacing:.2em;color:#6b6b9a;text-transform:uppercase;margin-top:.2rem">
        Study Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#6b6b9a;margin-bottom:.5rem">Navigate</div>', unsafe_allow_html=True)

    page = st.radio("nav", [
        "⚡  Predict","📊  History","📄  PDF Analyser",
        "🧩  MCQ Quiz","🔬  Technique Checker","🤖  RL Planner"
    ], label_visibility="collapsed")

    if "Predict" in page:
        st.divider()
        st.markdown('<div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#6b6b9a;margin-bottom:.6rem">Session Inputs</div>', unsafe_allow_html=True)
        hours        = st.slider("⏱ Hours Studied",  0, 10, 4)
        focus        = st.slider("🎯 Focus Level",    1, 10, 7)
        distractions = st.slider("📵 Distractions",   0, 10, 2)
        sleep        = st.slider("😴 Sleep Hours",     0, 10, 7)
        subject      = st.selectbox("📚 Subject", SUBJECTS)
        st.markdown("")
        predict_btn  = st.button("⚡  Predict Productivity",
                                  use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# ⚡  PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "Predict" in page:
    st.markdown("""
    <div class="page-head">
      <h1>Study Intelligence Dashboard</h1>
      <p>Random Forest ML · Adjust your session below and click Predict</p>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        df_in = build_input(hours,focus,distractions,sleep,subject)
        try:
            score = round(float(np.clip(model.predict(df_in)[0],0,10)),2)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Delete models/model.pkl and run `python src/train.py` again.")
            st.stop()

        eff     = round(min(score/(hours+0.01)*2.5,10),1)
        col_hex = sc(score)

        # ── Metric cards ──
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="mcard"><div class="ml">Predicted Score</div>'
                        f'<div class="mv" style="color:{col_hex}">{score}<span style="font-size:1rem;color:#6b6b9a">/10</span></div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mcard"><div class="ml">Efficiency Index</div>'
                        f'<div class="mv">{eff}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="mcard"><div class="ml">Grade</div>'
                        f'<div class="mv" style="color:{col_hex}">{grade(score)}</div></div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Productivity — {int(score*10)}%**")
        st.progress(score/10)

        # ── Tips ──
        st.markdown('<div style="font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#00d9c0;margin:1.2rem 0 .5rem">💡 Recommendations</div>', unsafe_allow_html=True)
        for tip in get_tips(hours,focus,distractions,sleep,score):
            st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)

        # ── Subject chart ──
        df_aug = pd.concat([df_base, pd.DataFrame([{
            "hours_studied":hours,"focus_level":focus,
            "distractions":distractions,"sleep_hours":sleep,
            "subject":subject,"productivity":score}])], ignore_index=True)
        sp = df_aug.groupby("subject")["productivity"].mean()
        st.markdown('<div style="font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#6b6b9a;margin:1.4rem 0 .5rem">📊 Subject Performance</div>', unsafe_allow_html=True)
        st.pyplot(dark_chart(sp.to_dict(),"Average Productivity by Subject")); plt.close()
        st.info(f"🔴 **Needs work:** {sp.idxmin()}   |   🟢 **Strongest:** {sp.idxmax()}")

        st.session_state.history.append({"time":datetime.now().strftime("%H:%M"),
            "subject":subject,"hours":hours,"focus":focus,
            "distractions":distractions,"sleep":sleep,"score":score})
    else:
        st.markdown("""
        <div class="ph-box">
          <div class="ph-icon">🎯</div>
          <strong>Adjust the session parameters on the left</strong><br>
          and click <strong>Predict Productivity</strong> to see your score,
          subject chart, and personalised recommendations.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 📊  HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "History" in page:
    st.markdown('<div class="page-head"><h1>Session History</h1><p>Your study sessions this visit</p></div>', unsafe_allow_html=True)
    if st.session_state.history:
        df_h=pd.DataFrame(st.session_state.history)
        st.dataframe(df_h.style.background_gradient(subset=["score"],cmap="RdYlGn",vmin=0,vmax=10),
                     use_container_width=True)
        fig,ax=plt.subplots(figsize=(8,3))
        fig.patch.set_facecolor("#0c0d1a"); ax.set_facecolor("#0c0d1a")
        ax.plot(df_h["score"],marker="o",color="#5b5ef4",linewidth=2)
        ax.fill_between(range(len(df_h)),df_h["score"],alpha=0.12,color="#5b5ef4")
        ax.set_ylim(0,10); ax.set_xticks(range(len(df_h)))
        ax.set_xticklabels(df_h["time"],color="#a0a0cc",fontsize=8)
        ax.tick_params(colors="#a0a0cc")
        for s in ax.spines.values(): s.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.markdown('<div class="ph-box"><div class="ph-icon">📊</div>No sessions yet. Run a prediction first.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 📄  PDF ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
elif "PDF" in page:
    st.markdown('<div class="page-head"><h1>PDF Study Analyser</h1><p>Upload notes or a textbook — AI extracts key concepts, weak areas, and study order</p></div>', unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload PDF or TXT",type=["pdf","txt"])
    if uploaded:
        with st.spinner("Extracting text…"):
            if PDF_MODULE: st.session_state.pdf_text=extract_text(uploaded)
            else:
                raw=uploaded.read()
                st.session_state.pdf_text=raw.decode("utf-8",errors="ignore") if isinstance(raw,bytes) else raw
        n=len(st.session_state.pdf_text)
        if n: st.success(f"✅ {n:,} characters from `{uploaded.name}`")
        else: st.warning("Could not extract text. Try a plain .txt file.")
    c1,c2=st.columns(2)
    with c1:
        if st.button("🔍 Analyse Document",use_container_width=True,type="primary",
                     disabled=not bool(st.session_state.pdf_text)):
            with st.spinner("Claude is analysing…"):
                r=ask_claude(f"Analyse this study document:\n1.**Summary**\n2.**Key Concepts**\n"
                             f"3.**Weak Areas**\n4.**Study Order**\n\nDoc:\n{st.session_state.pdf_text[:3500]}",
                             "You are a concise study assistant.")
            st.markdown(r)
    with c2:
        if st.button("🧩 Generate Quiz from PDF",use_container_width=True,
                     disabled=not bool(st.session_state.pdf_text)):
            st.session_state.quiz_source=st.session_state.pdf_text[:2500]
            st.success("✅ PDF loaded → switch to **MCQ Quiz** tab")
    if not st.session_state.pdf_text:
        st.markdown('<div class="ph-box"><div class="ph-icon">📄</div>Upload a PDF or TXT file to begin analysis.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🧩  MCQ QUIZ
# ══════════════════════════════════════════════════════════════════════════════
elif "MCQ" in page:
    st.markdown('<div class="page-head"><h1>MCQ Quiz Generator</h1><p>AI generates fresh multiple-choice questions with explanations</p></div>', unsafe_allow_html=True)
    c1,c2=st.columns([2,1])
    with c1:
        qsubj=st.selectbox("Topic",SUBJECTS+["Custom…"])
        qcustom=""
        if qsubj=="Custom…":
            qcustom=st.text_area("Custom topic or paste PDF content",
                                  value=getattr(st.session_state,"quiz_source",""),height=90)
    with c2:
        qcount=st.selectbox("Questions",[3,5,8])
    if st.button("⚡  Generate Questions",type="primary"):
        topic=qcustom.strip() if qsubj=="Custom…" and qcustom.strip() else qsubj
        with st.spinner("Generating…"):
            if QUIZ_MODULE:
                try:
                    st.session_state.quiz_data=generate_mcq(topic[:1500],qcount)
                    for q in st.session_state.quiz_data: q["selected"]=None; q["revealed"]=False
                except Exception as e:
                    st.error(str(e)); st.session_state.quiz_data=[]
            else:
                raw=ask_claude(f"Generate {qcount} MCQs about {topic[:1500]}.\nReturn ONLY JSON array:\n"
                               '[{"q":"...","options":["A....","B....","C....","D...."],"answer":"A","explanation":"..."}]')
                try:
                    st.session_state.quiz_data=json.loads(re.sub(r"```json|```","",raw).strip())
                    for q in st.session_state.quiz_data: q["selected"]=None; q["revealed"]=False
                except: st.error("Could not parse. Try again."); st.session_state.quiz_data=[]

    if st.session_state.quiz_data:
        done=all(q.get("revealed") for q in st.session_state.quiz_data); correct=0
        for i,q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            ai=ord(q["answer"].upper())-65; lts=["A","B","C","D"]
            for j,opt in enumerate(q["options"]):
                lbl=opt.lstrip("ABCD. ")
                if not q.get("revealed"):
                    if st.button(f"{lts[j]}. {lbl}",key=f"o{i}{j}"):
                        st.session_state.quiz_data[i]["selected"]=j; st.rerun()
                else:
                    if j==ai: st.success(f"✅ {lts[j]}. {lbl}")
                    elif j==q.get("selected"): st.error(f"❌ {lts[j]}. {lbl}")
                    else: st.write(f"   {lts[j]}. {lbl}")
            if q.get("selected") is not None and not q.get("revealed"):
                st.caption(f"Selected: {lts[q['selected']]}")
            if q.get("revealed"):
                if q.get("selected")==ai: correct+=1
                if q.get("explanation"): st.info(f"💡 {q['explanation']}")
            st.divider()
        if not done:
            if st.button("✓ Submit All",type="primary"):
                for q in st.session_state.quiz_data: q["revealed"]=True; st.rerun()
        else:
            st.success(f"🏆 Final Score: **{correct}/{len(st.session_state.quiz_data)}**")
            if st.button("🔄 New Quiz"): st.session_state.quiz_data=[]; st.rerun()
    elif not st.session_state.get("quiz_data"):
        st.markdown('<div class="ph-box"><div class="ph-icon">🧩</div>Select a topic and click Generate — AI will create fresh questions with explanations.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🔬  TECHNIQUE CHECKER
# ══════════════════════════════════════════════════════════════════════════════
elif "Technique" in page:
    st.markdown('<div class="page-head"><h1>Study Technique Checker</h1><p>Science-backed ratings + AI personalised feedback</p></div>', unsafe_allow_html=True)
    TMAP={"🍅 Pomodoro":"Pomodoro","🧠 Active Recall":"Active Recall",
          "📅 Spaced Repetition":"Spaced Repetition","🗺️ Mind Mapping":"Mind Mapping",
          "✏️ Feynman Technique":"Feynman Technique","📖 Passive Re-reading":"Passive Re-reading"}
    FALLBACK={"Pomodoro":{"focus":5,"retention":4,"efficiency":5,"science":4},
              "Active Recall":{"focus":4,"retention":5,"efficiency":5,"science":5},
              "Spaced Repetition":{"focus":3,"retention":5,"efficiency":5,"science":5},
              "Mind Mapping":{"focus":4,"retention":4,"efficiency":3,"science":3},
              "Feynman Technique":{"focus":5,"retention":5,"efficiency":4,"science":5},
              "Passive Re-reading":{"focus":2,"retention":2,"efficiency":2,"science":1}}
    sel_d=st.multiselect("Select your technique(s)",list(TMAP.keys()))
    sel=[TMAP[k] for k in sel_d]
    routine=st.text_area("Describe your study routine",height=100,
                          placeholder="e.g. I study 3 hrs straight, re-read notes, highlight important lines…")
    subj=st.selectbox("Subject context",SUBJECTS,key="tech_subj")
    if st.button("🔬  Analyse My Technique",type="primary"):
        if not sel and not routine.strip():
            st.warning("Select at least one technique or describe your routine.")
        else:
            if TECH_MODULE: r=get_ratings(sel)
            elif sel:
                tot={"focus":0,"retention":0,"efficiency":0,"science":0}
                for t in sel:
                    rv=FALLBACK.get(t,{"focus":3,"retention":3,"efficiency":3,"science":3})
                    for k in tot: tot[k]+=rv[k]
                r={k:round(v/len(sel),1) for k,v in tot.items()}
            else: r={"focus":3,"retention":3,"efficiency":3,"science":3}
            st.markdown("### 📊 Technique Ratings")
            cs=st.columns(4)
            for c,k,l in zip(cs,["focus","retention","efficiency","science"],
                              ["Focus","Retention","Efficiency","Science"]):
                with c: st.metric(l,"⭐"*round(r[k])+"☆"*(5-round(r[k])))
            tn=sel if sel else ["Custom technique"]
            p=(f"Student studying {subj} uses: {', '.join(tn)}.\n"
               f"Routine: \"{routine or 'Not described'}\"\n\n"
               "Provide:\n1.**Overall Assessment**\n2.**What's Working Well**\n"
               "3.**What to Improve** (3-4 actionable bullets)\n4.**One Concrete Change This Week**")
            with st.spinner("Claude evaluating…"):
                fb=ask_claude(p,"Cognitive science expert. Be specific and practical.")
            st.markdown("### 🤖 AI Feedback"); st.markdown(fb)
    else:
        st.markdown('<div class="ph-box"><div class="ph-icon">🔬</div>Select technique(s), describe your routine and click Analyse to get science-backed ratings + AI feedback.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🤖  RL PLANNER
# ══════════════════════════════════════════════════════════════════════════════
elif "RL" in page:
    st.markdown('<div class="page-head"><h1>RL Study Planner</h1><p>ε-greedy Reinforcement Learning agent builds your optimal weekly schedule</p></div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: rh=st.slider("⏱ Hours / Day",1,10,3)
    with c2: rd=st.slider("📅 Days to Exam",1,30,7)
    with c3: rp=st.selectbox("🎯 Priority Subject",["Auto"]+SUBJECTS)
    if st.button("🤖  Generate Plan",type="primary"):
        with st.spinner("RL agent computing optimal plan…"): time.sleep(0.6)
        if RL_MODULE:
            plan=build_plan(rh,rd,rp); sm=summarise_plan(plan)
        else:
            IC={"DSA":"💻","OOP":"🧱","Maths":"📐","Physics":"⚛️","History":"📜"}
            Q={"DSA":0.70,"OOP":0.80,"Maths":0.55,"Physics":0.60,"History":0.50}
            if rp!="Auto" and rp in Q: Q[rp]=min(1.0,Q[rp]*1.5)
            td=min(rd,7); spd=max(1,round(rh/1.5)); hps=round(rh/spd,1)
            plan=[]
            for d in range(td):
                dq=dict(Q); slots=[]
                for _ in range(spd):
                    s=max(dq,key=dq.__getitem__) if random.random()>0.1 else random.choice(SUBJECTS)
                    slots.append({"subject":s,"hours":hps,"intensity":min(5,max(1,round(hps*1.8))),"icon":IC.get(s,"📚")})
                    dq[s]*=0.72
                pred=round(min(10,sum(x["intensity"] for x in slots)/len(slots)*2+4),1)
                plan.append({"day":d+1,"label":["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d%7],
                             "slots":slots,"total_hours":rh,"predicted_score":pred})
            sh={}
            for day in plan:
                for s in day["slots"]: sh[s["subject"]]=sh.get(s["subject"],0)+s["hours"]
            sm={"total_hours":round(rh*len(plan),1),"days_planned":len(plan),
                "top_subject":max(sh,key=sh.__getitem__) if sh else "DSA","subject_hours":sh}

        c1,c2,c3,c4=st.columns(4)
        for col,lb,val in zip([c1,c2,c3,c4],
            ["Total Hours","Days Planned","Top Subject","Avg/Day"],
            [f"{sm['total_hours']}h",sm['days_planned'],sm['top_subject'],f"{rh}h"]):
            with col:
                st.markdown(f'<div class="mcard"><div class="ml">{lb}</div>'
                            f'<div class="mv" style="font-size:1.3rem">{val}</div></div>',
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        for day in plan:
            with st.expander(f"Day {day['day']} · {day['label']}  —  Predicted: {day['predicted_score']}/10",expanded=True):
                cols=st.columns(len(day["slots"]))
                for col,slot in zip(cols,day["slots"]):
                    with col:
                        st.markdown(f"**{slot['icon']} {slot['subject']}**  \n{slot['hours']}h · intensity {slot['intensity']}/5")
                        st.progress(slot["intensity"]/5)

        st.markdown("### 🧠 Agent Strategy Notes")
        with st.spinner("Getting strategy from Claude…"):
            if RL_MODULE: notes=get_strategy_notes(plan,rh,rd,rp)
            else:
                sh_s=", ".join(f"{s} {h:.1f}h" for s,h in sm["subject_hours"].items())
                notes=ask_claude(f"Student: {rd}d to exam, {rh}h/day, priority {rp}. Plan: {sh_s}.\n3 bullet tips. Max 120 words.","Concise study planner.")
        st.markdown(notes)
    else:
        st.markdown('<div class="ph-box"><div class="ph-icon">🤖</div>Set your available hours and exam deadline, then click Generate Plan to get your RL-optimised schedule.</div>', unsafe_allow_html=True)
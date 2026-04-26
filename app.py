from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.intelligence import (
    DEFAULT_SUBJECTS,
    add_time_columns,
    analyze_pdf_complexity,
    bootstrap_models_if_missing,
    build_complexity_classifier,
    build_full_training_data,
    build_rl_state,
    build_rl_study_plan,
    check_study_methods,
    choose_rl_action,
    compute_spider_metrics,
    compute_consistency_score,
    detect_behavior_patterns,
    extract_pdf_text,
    forecast_with_recommendation,
    generate_quiz_from_text,
    load_models,
    load_profile,
    load_rl_memory,
    make_productivity_input,
    make_recommendation_input,
    maybe_retrain_models,
    quiz_to_pdf_bytes,
    safe_read_csv,
    save_rl_memory,
    update_profile,
    update_rl_q,
)


MODEL_DIR = Path("models")
BASE_DATA_PATH = Path("data/study_data.csv")
HISTORY_PATH = Path("data/user_history.csv")
PROFILE_PATH = Path("data/user_profile.json")
RL_MEMORY_PATH = Path("data/rl_memory.json")

PAGES = [
    "Predict",
    "History",
    "PDF Analyzer",
    "Quiz Converter",
    "Technique Checker",
    "RL Planner",
    "Focus Session",
    "Dashboard",
]

THEMES = {
    "Neon Grid": {
        "bg": "#070810",
        "panel": "#0c0d1a",
        "card": "#111224",
        "border": "#1c1d35",
        "accent": "#5b5ef4",
        "teal": "#00d9c0",
        "amber": "#f59e0b",
        "green": "#10b981",
        "text": "#eeeeff",
        "muted": "#6b6b9a",
        "soft": "#a0a0cc",
        "grad1": "rgba(91,94,244,.08)",
        "grad2": "rgba(0,217,192,.06)",
    },
    "Cyber Teal": {
        "bg": "#061117",
        "panel": "#0a1b24",
        "card": "#0f2431",
        "border": "#1e3a47",
        "accent": "#00bcd4",
        "teal": "#34f5c5",
        "amber": "#f7c948",
        "green": "#50e3a4",
        "text": "#e6fbff",
        "muted": "#7cb6c9",
        "soft": "#a6d7e4",
        "grad1": "rgba(0,188,212,.1)",
        "grad2": "rgba(80,227,164,.08)",
    },
    "Sunset Pulse": {
        "bg": "#14090c",
        "panel": "#251116",
        "card": "#2d171d",
        "border": "#4a2630",
        "accent": "#ff4d6d",
        "teal": "#ff9f43",
        "amber": "#ffd166",
        "green": "#80ed99",
        "text": "#ffeef3",
        "muted": "#d6a5b2",
        "soft": "#efc6d0",
        "grad1": "rgba(255,77,109,.12)",
        "grad2": "rgba(255,159,67,.09)",
    },
}

TECHNIQUE_LIBRARY = {
    "Pomodoro": {
        "desc": "Short timed blocks with micro-breaks.",
        "ratings": {"Focus": 5, "Retention": 4, "Efficiency": 5, "Science": 4},
    },
    "Active Recall": {
        "desc": "Recall first, then verify with notes.",
        "ratings": {"Focus": 4, "Retention": 5, "Efficiency": 5, "Science": 5},
    },
    "Spaced Repetition": {
        "desc": "Review on increasing intervals.",
        "ratings": {"Focus": 3, "Retention": 5, "Efficiency": 5, "Science": 5},
    },
    "Practice Testing": {
        "desc": "Frequent problem solving and mock testing.",
        "ratings": {"Focus": 4, "Retention": 5, "Efficiency": 5, "Science": 5},
    },
    "Feynman Technique": {
        "desc": "Teach concepts in simple language.",
        "ratings": {"Focus": 5, "Retention": 5, "Efficiency": 4, "Science": 5},
    },
    "Mind Mapping": {
        "desc": "Visual linking of concepts and hierarchies.",
        "ratings": {"Focus": 4, "Retention": 4, "Efficiency": 3, "Science": 3},
    },
    "Passive Re-reading": {
        "desc": "Re-reading notes without active retrieval.",
        "ratings": {"Focus": 2, "Retention": 2, "Efficiency": 2, "Science": 1},
    },
}


def apply_demo_theme(theme_name):
    theme = THEMES.get(theme_name, THEMES["Neon Grid"])
    css_template = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=Mulish:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');
        :root{
          --bg:__BG__;--panel:__PANEL__;--card:__CARD__;--border:__BORDER__;
          --accent:__ACCENT__;--teal:__TEAL__;--amber:__AMBER__;--green:__GREEN__;
          --text:__TEXT__;--muted:__MUTED__;--soft:__SOFT__;
        }
        .stApp{
          background:
            radial-gradient(ellipse 800px 500px at 10% 20%, __GRAD1__ 0%, transparent 70%),
            radial-gradient(ellipse 500px 300px at 90% 70%, __GRAD2__ 0%, transparent 70%),
            var(--bg);
          color:var(--text);
          font-family:'Mulish', sans-serif;
        }
        [data-testid="stSidebar"]{
          background:rgba(12,13,26,.98);
          border-right:1px solid var(--border);
        }
        [data-testid="stSidebar"] *{
          color:var(--text) !important;
        }
        h1,h2,h3{
          font-family:'Syne', sans-serif !important;
          color:var(--text) !important;
          letter-spacing:.02em;
        }
        .brand{
          background:var(--card);
          border:1px solid var(--border);
          border-radius:12px;
          padding:.8rem .9rem;
          margin-bottom:.8rem;
        }
        .brand h3{
          margin:0;
          font-size:1rem;
          font-weight:800;
        }
        .brand p{
          margin:.15rem 0 0 0;
          font-size:.66rem;
          letter-spacing:.12em;
          text-transform:uppercase;
          color:var(--muted) !important;
        }
        .hero{
          background:var(--card);
          border:1px solid var(--border);
          border-radius:14px;
          padding:1.1rem 1.2rem;
          margin-bottom:.8rem;
        }
        .hero p{
          color:var(--muted);
          margin:.2rem 0 0 0;
          font-size:.9rem;
        }
        [data-testid="stRadio"] label{
          background:transparent;
          border:1px solid transparent;
          border-radius:8px;
          padding:.45rem .6rem;
          margin-bottom:.2rem;
          transition:.15s ease;
        }
        [data-testid="stRadio"] label:hover{
          background:var(--card);
          border-color:var(--border);
        }
        .stSlider, .stSelectbox, .stNumberInput, .stTextArea, .stFileUploader, .stDataFrame, .stMetric{
          background:var(--card);
          border:1px solid var(--border);
          border-radius:10px;
          padding:.55rem .7rem;
        }
        .stButton > button{
          background:linear-gradient(135deg,var(--accent),#4338ca);
          color:white;
          border:none;
          border-radius:8px;
          font-family:'Syne', sans-serif;
          font-size:.82rem;
          letter-spacing:.03em;
        }
        .label{
          font-size:.67rem;
          color:var(--muted);
          text-transform:uppercase;
          letter-spacing:.12em;
          margin-bottom:.3rem;
          display:block;
        }
        .page-head{
          border-bottom:1px solid var(--border);
          padding-bottom:.55rem;
          margin-bottom:.7rem;
        }
        .page-head h2{
          margin:0;
          font-size:1.25rem;
        }
        .page-head p{
          margin:.2rem 0 0;
          color:var(--muted);
          font-size:.82rem;
        }
        .note{
          color:var(--soft);
          font-size:.86rem;
          line-height:1.6;
        }
        .tech-card{
          background:var(--card);
          border:1px solid var(--border);
          border-radius:10px;
          padding:.7rem .8rem;
          margin-bottom:.5rem;
        }
        .tech-card p{
          color:var(--muted);
          font-size:.78rem;
          margin:.2rem 0 0;
        }
        .stars{
          font-family:'JetBrains Mono', monospace;
          color:var(--amber);
          letter-spacing:.08em;
        }
        </style>
    """
    css = (
        css_template.replace("__BG__", theme["bg"])
        .replace("__PANEL__", theme["panel"])
        .replace("__CARD__", theme["card"])
        .replace("__BORDER__", theme["border"])
        .replace("__ACCENT__", theme["accent"])
        .replace("__TEAL__", theme["teal"])
        .replace("__AMBER__", theme["amber"])
        .replace("__GREEN__", theme["green"])
        .replace("__TEXT__", theme["text"])
        .replace("__MUTED__", theme["muted"])
        .replace("__SOFT__", theme["soft"])
        .replace("__GRAD1__", theme["grad1"])
        .replace("__GRAD2__", theme["grad2"])
    )
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_complexity_classifier():
    return build_complexity_classifier()


def persist_history_row(row):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        HISTORY_PATH,
        mode="a",
        header=not HISTORY_PATH.exists(),
        index=False,
    )


def get_past_productivity(full_df, subject):
    if len(full_df) == 0:
        return 6.0
    subject_hist = full_df[full_df["subject"] == subject]["productivity"]
    if len(subject_hist) > 0:
        return float(subject_hist.tail(5).mean())
    return float(full_df["productivity"].mean())


def profile_adjustment_factor(profile, subject):
    subject_perf = profile.get("subject_productivity", {})
    if not subject_perf or subject not in subject_perf:
        return 1.0
    subject_score = float(subject_perf[subject])
    global_avg = float(np.mean(list(subject_perf.values())))
    return 1.1 if subject_score < global_avg else 0.95


def render_focus_heatmap(full_df):
    heat_df = add_time_columns(full_df)
    pivot = heat_df.pivot_table(
        index="focus_level",
        columns="time_bucket",
        values="productivity",
        aggfunc="mean",
    )
    ordered_cols = ["Morning", "Afternoon", "Evening", "Night"]
    pivot = pivot.reindex(columns=ordered_cols).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_title("Focus vs Time Heatmap")
    ax.set_xlabel("Time Bucket")
    ax.set_ylabel("Focus Level")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist())
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    fig.colorbar(im, ax=ax, label="Avg Productivity")
    st.pyplot(fig)


def render_spider_chart(chart_data, color="#5b5ef4"):
    labels = chart_data["labels"]
    values = chart_data["values"]
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.22, color=color)
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(chart_data["title"])
    st.pyplot(fig)


def render_page_head(title, subtitle):
    st.markdown(
        f"""
        <div class="page-head">
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_pdf_state():
    st.session_state.setdefault("pdf_multiplier", 1.0)
    st.session_state.setdefault("pdf_result", {})
    st.session_state.setdefault("pdf_text", "")
    st.session_state.setdefault("pdf_quiz", [])


def ensure_technique_state():
    st.session_state.setdefault("selected_techniques", [])
    st.session_state.setdefault("technique_summary", "")


def method_name_to_library_key(method_name):
    if method_name == "Pomodoro Blocks":
        return "Pomodoro"
    return method_name


def compute_technique_ratings(selected_techniques):
    metrics = ["Focus", "Retention", "Efficiency", "Science"]
    if not selected_techniques:
        return {m: 3.0 for m in metrics}
    accum = {m: 0.0 for m in metrics}
    for name in selected_techniques:
        key = method_name_to_library_key(name)
        profile = TECHNIQUE_LIBRARY.get(key, {"ratings": {m: 3 for m in metrics}})
        for metric in metrics:
            accum[metric] += float(profile["ratings"][metric])
    count = float(len(selected_techniques))
    return {m: round(accum[m] / count, 1) for m in metrics}


def get_dynamic_chart_df(history_df, fallback_df):
    required = ["hours_studied", "focus_level", "distractions", "sleep_hours", "subject", "productivity"]
    if history_df is not None and len(history_df) > 0 and all(col in history_df.columns for col in required):
        clean = history_df.dropna(subset=required)
        if len(clean) > 0:
            return clean.copy()
    return fallback_df.copy()


def build_live_spider_metrics(hours, focus, distractions, sleep, subject, models):
    live_input = make_productivity_input(
        hours=hours,
        focus=focus,
        distractions=distractions,
        sleep=sleep,
        subject=subject,
        columns=models["productivity_columns"],
    )
    live_pred = float(models["productivity_model"].predict(live_input)[0])
    return {
        "title": f"{subject} Live Session Profile",
        "labels": ["Hours", "Focus", "Sleep", "Productivity", "Distraction Control"],
        "values": [
            round(float(hours), 2),
            round(float(focus), 2),
            round(float(sleep), 2),
            round(float(np.clip(live_pred, 0, 10)), 2),
            round(float(np.clip(10 - distractions, 0, 10)), 2),
        ],
    }


def render_predict_page(base_df, full_df, models, profile, rl_memory):
    render_page_head("Study Intelligence", "Set session parameters and run adaptive prediction.")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("<span class='label'>Session Parameters</span>", unsafe_allow_html=True)
        st.session_state.setdefault("hours_studied", 4.0)
        st.session_state.setdefault("focus_level", 6)
        st.session_state.setdefault("distractions", 3)
        st.session_state.setdefault("sleep_hours", 7.0)
        st.session_state.setdefault("subject", DEFAULT_SUBJECTS[0])

        hours = st.slider("Hours Studied", 0.0, 10.0, key="hours_studied", step=0.5)
        focus = st.slider("Focus Level", 1, 10, key="focus_level")
        distractions = st.slider("Distractions", 0, 10, key="distractions")
        sleep = st.slider("Sleep Hours", 0.0, 10.0, key="sleep_hours", step=0.5)
        subject = st.selectbox("Subject", DEFAULT_SUBJECTS, key="subject")
        predict_clicked = st.button("Predict Productivity")

    with right:
        st.markdown("<span class='label'>Prediction Output</span>", unsafe_allow_html=True)
        selected_spider_subject = subject
        st.caption(f"Spider subject (live): {selected_spider_subject}")
        live_spider = build_live_spider_metrics(
            hours=hours,
            focus=focus,
            distractions=distractions,
            sleep=sleep,
            subject=selected_spider_subject,
            models=models,
        )
        if "last_prediction" in st.session_state:
            pred_data = st.session_state.last_prediction
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted", f"{pred_data['pred']:.2f}/10")
            c2.metric("Recommended", f"{pred_data['rec']:.2f}h")
            c3.metric("Forecast", f"{pred_data['forecast']:.2f}/10")
            st.info(pred_data["rl_tip"])
        else:
            st.markdown(
                "<p class='note'>Run prediction to view personalized recommendation and RL action.</p>",
                unsafe_allow_html=True,
            )
        render_spider_chart(live_spider, color="#00d9c0")

    if not predict_clicked:
        return

    prod_input = make_productivity_input(
        hours=hours,
        focus=focus,
        distractions=distractions,
        sleep=sleep,
        subject=subject,
        columns=models["productivity_columns"],
    )
    pred_value = float(models["productivity_model"].predict(prod_input)[0])

    past_prod = get_past_productivity(full_df, subject)
    rec_input = make_recommendation_input(
        subject=subject,
        focus=focus,
        past_productivity=past_prod,
        columns=models["recommendation_columns"],
    )
    raw_recommendation = float(models["recommendation_model"].predict(rec_input)[0])
    adaptive_factor = profile_adjustment_factor(profile, subject)
    pdf_factor = float(st.session_state.get("pdf_multiplier", 1.0))
    recommended_hours = float(np.clip(raw_recommendation * adaptive_factor * pdf_factor, 1, 10))

    forecast = forecast_with_recommendation(
        productivity_model=models["productivity_model"],
        productivity_columns=models["productivity_columns"],
        subject=subject,
        focus=focus,
        distractions=distractions,
        sleep=sleep,
        recommended_hours=recommended_hours,
    )

    rl_state = build_rl_state(subject, focus, past_prod, distractions)
    rl_action_name, _ = choose_rl_action(rl_memory, rl_state, epsilon=0.08)
    reward = np.clip((pred_value / 10.0) + (forecast / 15.0) - (distractions / 25.0), 0, 1.7)
    update_rl_q(rl_memory, rl_state, rl_action_name, reward)
    save_rl_memory(RL_MEMORY_PATH, rl_memory)

    now = datetime.now()
    new_row = {
        "hours_studied": float(hours),
        "focus_level": int(focus),
        "distractions": int(distractions),
        "sleep_hours": float(sleep),
        "subject": subject,
        "productivity": round(pred_value, 2),
        "recommended_hours": round(recommended_hours, 2),
        "date": now.date(),
        "timestamp": now.isoformat(),
    }
    persist_history_row(new_row)

    refreshed_history = safe_read_csv(HISTORY_PATH)
    refreshed_df = build_full_training_data(base_df, refreshed_history)
    refreshed_chart_df = get_dynamic_chart_df(refreshed_history, refreshed_df)
    retrained = maybe_retrain_models(refreshed_df, MODEL_DIR, min_rows=20)
    if retrained:
        st.session_state.models = load_models(MODEL_DIR)

    update_profile(PROFILE_PATH, refreshed_df)

    st.session_state.last_prediction = {
        "pred": pred_value,
        "rec": recommended_hours,
        "forecast": forecast,
        "rl_tip": f"RL Action: {rl_action_name}. Recommendation includes profile and PDF complexity.",
        "spider": compute_spider_metrics(refreshed_chart_df, subject=subject),
    }
    st.rerun()


def render_history_page():
    render_page_head("History", "Recent logged sessions and outcomes.")
    history_df = safe_read_csv(HISTORY_PATH)
    if len(history_df) == 0:
        st.info("No sessions logged yet.")
        return
    show_cols = [
        col
        for col in [
            "date",
            "subject",
            "hours_studied",
            "focus_level",
            "distractions",
            "sleep_hours",
            "productivity",
            "recommended_hours",
        ]
        if col in history_df.columns
    ]
    st.dataframe(history_df[show_cols].tail(40).iloc[::-1], use_container_width=True)


def render_pdf_analyzer_page():
    ensure_pdf_state()
    render_page_head("PDF Analyzer", "Analyze material complexity and effort weight.")
    uploaded_pdf = st.file_uploader("Upload study material PDF", type=["pdf"], key="study_pdf")

    if uploaded_pdf is not None and st.button("Analyze Document"):
        text = extract_pdf_text(uploaded_pdf)
        if not text:
            st.error("No readable text found in PDF.")
        else:
            st.session_state.pdf_text = text
            classifier = get_complexity_classifier()
            result = analyze_pdf_complexity(text, classifier)
            st.session_state.pdf_multiplier = result["effort_multiplier"]
            st.session_state.pdf_result = result

    result = st.session_state.pdf_result
    if result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Complexity", result["complexity_score"])
        c2.metric("Difficulty", result["difficulty_label"])
        c3.metric("Effort x", result["effort_multiplier"])
        st.write("Keywords:", ", ".join(result.get("keywords", [])) or "None")
    else:
        st.markdown("<p class='note'>Upload a PDF and analyze to enable adaptation.</p>", unsafe_allow_html=True)


def render_quiz_converter_page():
    ensure_pdf_state()
    render_page_head("Quiz Converter", "Convert analyzed PDF content into questions and export as PDF.")
    if not st.session_state.pdf_text:
        st.info("Analyze a PDF first in the PDF Analyzer page.")
        return

    count = st.slider("Number of quiz questions", 3, 15, 6, key="quiz_count")
    if st.button("Generate Questions"):
        st.session_state.pdf_quiz = generate_quiz_from_text(st.session_state.pdf_text, num_questions=count)

    quiz_items = st.session_state.pdf_quiz
    if not quiz_items:
        st.markdown("<p class='note'>Generate questions to preview and export.</p>", unsafe_allow_html=True)
        return
    quiz_df = pd.DataFrame(quiz_items)
    st.dataframe(quiz_df, use_container_width=True)
    pdf_bytes = quiz_to_pdf_bytes(quiz_items, title="AI Learns You - Quiz Set")
    st.download_button(
        "Download Quiz PDF",
        data=pdf_bytes,
        file_name="pdf_quiz_questions.pdf",
        mime="application/pdf",
    )


def toggle_technique(name):
    picked = st.session_state.selected_techniques
    if name in picked:
        picked.remove(name)
    else:
        picked.append(name)
    st.session_state.selected_techniques = picked


def render_technique_checker_page(full_df):
    ensure_technique_state()
    render_page_head("Technique Checker", "Evaluate technique stack with ratings and radar analysis.")

    subject_options = sorted(full_df["subject"].dropna().astype(str).unique().tolist()) if len(full_df) else DEFAULT_SUBJECTS
    selected_subject = st.selectbox("Subject", subject_options, key="technique_subject")

    st.markdown("<span class='label'>Select Techniques</span>", unsafe_allow_html=True)
    tech_names = list(TECHNIQUE_LIBRARY.keys())
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(1)
    cells = row1 + row2 + row3
    for idx, name in enumerate(tech_names):
        with cells[idx]:
            selected = name in st.session_state.selected_techniques
            button_text = f"{'Selected' if selected else 'Select'} | {name}"
            st.button(button_text, key=f"tech_pick_{name}", on_click=toggle_technique, args=(name,))
            st.markdown(
                f"""
                <div class="tech-card">
                  <strong>{name}</strong>
                  <p>{TECHNIQUE_LIBRARY[name]['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    routine_text = st.text_area("Current routine (optional)", key="technique_desc", height=100)

    if st.button("Analyze Technique Stack"):
        method_report = check_study_methods(full_df, subject=selected_subject)
        chosen = st.session_state.selected_techniques.copy()
        if not chosen:
            chosen = [method_name_to_library_key(method_report["methods"][0]["method"])]

        ratings = compute_technique_ratings(chosen)
        if len(routine_text.strip()) < 40:
            ratings["Science"] = round(max(1.0, ratings["Science"] - 0.3), 1)
            ratings["Efficiency"] = round(max(1.0, ratings["Efficiency"] - 0.2), 1)

        overall_score = round(sum(ratings.values()) / len(ratings), 2)
        top_method = method_report["methods"][0]
        summary = (
            f"Overall technique score: {overall_score}/5. "
            f"Strongest next move: {top_method['method']} ({top_method['score']}/100)."
        )
        st.session_state.technique_summary = summary
        st.session_state.technique_ratings = ratings
        st.session_state.technique_top_method = top_method
        st.session_state.technique_selection = chosen

    if "technique_ratings" not in st.session_state:
        st.markdown(
            "<p class='note'>Select techniques and click Analyze Technique Stack to see ratings.</p>",
            unsafe_allow_html=True,
        )
        return

    ratings = st.session_state.technique_ratings
    top_method = st.session_state.technique_top_method
    selected = st.session_state.technique_selection

    st.info(st.session_state.technique_summary)
    c1, c2 = st.columns([1, 1.1])
    with c1:
        radar = {
            "title": "Technique Quality Radar",
            "labels": list(ratings.keys()),
            "values": [round(v * 2.0, 2) for v in ratings.values()],
        }
        render_spider_chart(radar, color="#f59e0b")
    with c2:
        st.markdown("<span class='label'>Rating Breakdown</span>", unsafe_allow_html=True)
        for metric, value in ratings.items():
            filled = int(round(value))
            stars = ("*" * filled) + ("." * (5 - filled))
            st.markdown(
                f"<div><strong>{metric}</strong> &nbsp; <span class='stars'>{stars}</span> &nbsp; {value}/5</div>",
                unsafe_allow_html=True,
            )
        st.write("")
        st.write(f"Selected stack: {', '.join(selected)}")
        st.write(f"Best immediate change: {top_method['method']}")
        st.write(top_method["reason"])


def render_rl_planner_page(full_df, rl_memory):
    render_page_head("RL Planner", "Generate RL-driven study slot plans and track policy memory.")
    c1, c2, c3 = st.columns(3)
    with c1:
        hours_per_day = st.slider("Hours per day", 1.0, 10.0, 4.0, 0.5, key="rl_hours")
    with c2:
        days_to_exam = st.slider("Days to exam", 1, 30, 7, 1, key="rl_days")
    with c3:
        priority_subject = st.selectbox("Priority subject", ["Auto"] + DEFAULT_SUBJECTS, key="rl_priority")

    if st.button("Generate RL Study Plan"):
        plan_df = build_rl_study_plan(
            full_df=full_df,
            memory=rl_memory,
            hours_per_day=hours_per_day,
            days_to_exam=days_to_exam,
            priority_subject=priority_subject,
        )
        save_rl_memory(RL_MEMORY_PATH, rl_memory)
        st.session_state.rl_plan_df = plan_df

    if "rl_plan_df" not in st.session_state or not len(st.session_state.rl_plan_df):
        st.markdown("<p class='note'>Generate a plan to view RL-driven slots and actions.</p>", unsafe_allow_html=True)
        return

    plan_df = st.session_state.rl_plan_df
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Planned Days", int(plan_df["Day"].max()))
    m2.metric("Total Hours", round(float(plan_df["Hours"].sum()), 2))
    m3.metric("Top Subject", plan_df["Subject"].value_counts().idxmax())
    m4.metric("Avg Pred Score", round(float(plan_df["Predicted Productivity"].mean()), 2))
    st.dataframe(plan_df, use_container_width=True)


def render_focus_session_page():
    render_page_head("Focus Session", "Run distraction-free work blocks with a visual timer.")
    theme = THEMES.get(st.session_state.get("theme_name", "Neon Grid"), THEMES["Neon Grid"])

    c1, c2, c3 = st.columns(3)
    with c1:
        focus_minutes = st.slider("Focus minutes", 10, 90, 25, 5, key="focus_timer_minutes")
    with c2:
        short_break = st.slider("Short break", 3, 20, 5, 1, key="focus_timer_break")
    with c3:
        cycles = st.slider("Cycles", 1, 8, 4, 1, key="focus_timer_cycles")

    st.markdown(
        "<p class='note'>Start timer and keep this page open in your tab for uninterrupted focus cycles.</p>",
        unsafe_allow_html=True,
    )

    timer_html = f"""
    <div style="background:{theme['card']};border:1px solid {theme['border']};border-radius:12px;padding:1rem 1.1rem;color:{theme['text']};">
      <div style="font-family:Syne,sans-serif;font-size:.8rem;letter-spacing:.1em;text-transform:uppercase;color:{theme['muted']};margin-bottom:.45rem;">
        Focus Timer
      </div>
      <div id="phase" style="color:{theme['teal']};font-size:.95rem;margin-bottom:.35rem;font-weight:600;">Ready</div>
      <div id="clock" style="font-family:'JetBrains Mono',monospace;font-size:2.4rem;color:{theme['text']};margin-bottom:.8rem;">{focus_minutes:02d}:00</div>
      <div style="display:flex;gap:.55rem;flex-wrap:wrap;">
        <button id="startBtn" style="background:linear-gradient(135deg,{theme['accent']},#4338ca);color:#ffffff;border:none;border-radius:9px;padding:.55rem .95rem;cursor:pointer;font-weight:700;min-width:82px;">Start</button>
        <button id="pauseBtn" style="background:{theme['panel']};color:{theme['text']};border:1px solid {theme['border']};border-radius:9px;padding:.55rem .95rem;cursor:pointer;font-weight:700;min-width:82px;">Pause</button>
        <button id="resetBtn" style="background:{theme['panel']};color:{theme['text']};border:1px solid {theme['border']};border-radius:9px;padding:.55rem .95rem;cursor:pointer;font-weight:700;min-width:82px;">Reset</button>
      </div>
      <div id="status" style="margin-top:.7rem;color:{theme['soft']};font-size:.85rem;">Cycle 1 / {cycles}</div>
    </div>
    <script>
    (function() {{
      const focusMin = {focus_minutes};
      const breakMin = {short_break};
      const maxCycles = {cycles};
      let mode = "focus";
      let cycle = 1;
      let remaining = focusMin * 60;
      let timer = null;

      const phase = document.getElementById("phase");
      const clock = document.getElementById("clock");
      const status = document.getElementById("status");
      const startBtn = document.getElementById("startBtn");
      const pauseBtn = document.getElementById("pauseBtn");
      const resetBtn = document.getElementById("resetBtn");

      const format = (sec) => {{
        const m = Math.floor(sec / 60).toString().padStart(2, "0");
        const s = (sec % 60).toString().padStart(2, "0");
        return `${{m}}:${{s}}`;
      }};

      const repaint = () => {{
        phase.textContent = mode === "focus" ? "Focus Block" : "Short Break";
        phase.style.color = mode === "focus" ? "{theme['teal']}" : "{theme['amber']}";
        clock.textContent = format(remaining);
        status.textContent = `Cycle ${{cycle}} / ${{maxCycles}}`;
      }};

      const tick = () => {{
        if (remaining > 0) {{
          remaining -= 1;
          repaint();
          return;
        }}
        if (mode === "focus") {{
          mode = "break";
          remaining = breakMin * 60;
        }} else {{
          if (cycle >= maxCycles) {{
            clearInterval(timer);
            timer = null;
            phase.textContent = "Completed";
            phase.style.color = "{theme['green']}";
            return;
          }}
          cycle += 1;
          mode = "focus";
          remaining = focusMin * 60;
        }}
        repaint();
      }};

      startBtn.onclick = () => {{
        if (timer) return;
        timer = setInterval(tick, 1000);
      }};
      pauseBtn.onclick = () => {{
        if (!timer) return;
        clearInterval(timer);
        timer = null;
      }};
      resetBtn.onclick = () => {{
        if (timer) {{
          clearInterval(timer);
          timer = null;
        }}
        mode = "focus";
        cycle = 1;
        remaining = focusMin * 60;
        repaint();
      }};
      repaint();
    }})();
    </script>
    """
    components.html(timer_html, height=300)


def render_dashboard_page(base_df, history_df, full_df):
    if len(full_df) == 0:
        st.warning("No data found yet. Add at least one session.")
        return

    profile = update_profile(PROFILE_PATH, full_df)
    chart_df = get_dynamic_chart_df(history_df, full_df)
    render_page_head("Dashboard", "Trends, behavior signals, and per-subject diagnostics.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Productivity", round(float(full_df["productivity"].mean()), 2))
    m2.metric("Total Sessions", int(len(full_df)))
    m3.metric("Consistency Score", compute_consistency_score(full_df))

    st.subheader("Trend: You vs Dataset")
    trend_df = add_time_columns(full_df).sort_values("event_time")
    trend_df["session"] = np.arange(1, len(trend_df) + 1)
    trend_df["you_trend"] = trend_df["productivity"].rolling(3, min_periods=1).mean()
    dataset_avg = float(base_df["productivity"].mean()) if len(base_df) else float(trend_df["productivity"].mean())
    trend_view = trend_df[["session", "you_trend"]].copy()
    trend_view["dataset_avg"] = dataset_avg
    trend_view = trend_view.set_index("session")
    trend_view.columns = ["You", "Dataset Average"]
    st.line_chart(trend_view)

    st.subheader("Focus vs Time")
    render_focus_heatmap(full_df)

    st.subheader("Behavior Pattern Detection")
    insights = detect_behavior_patterns(full_df)
    st.write(insights["best_window"])
    st.write(insights["draining_subject"])
    st.write(insights["burnout_signal"])

    left, right = st.columns(2)
    with left:
        st.subheader("Spider Chart - Overall")
        render_spider_chart(compute_spider_metrics(chart_df, subject=None), color="#5b5ef4")
    with right:
        st.subheader("Spider Chart - Per Subject")
        subject_options = sorted(set(DEFAULT_SUBJECTS).union(set(chart_df["subject"].dropna().astype(str).unique().tolist())))
        selected_subject = st.selectbox(
            "Select subject",
            subject_options if subject_options else DEFAULT_SUBJECTS,
            key="dashboard_spider_subject",
        )
        render_spider_chart(compute_spider_metrics(chart_df, subject=selected_subject), color="#00d9c0")

    st.subheader("Learns-You Memory")
    st.write(f"Remembered sessions: {profile['total_sessions']}")
    st.write(f"Best performance window: {profile['best_time_bucket']}")
    st.write(f"Top subject trend: {profile['top_subject']}")


def main():
    st.set_page_config(page_title="AI Learns You", layout="wide")
    st.session_state.setdefault("theme_name", "Neon Grid")
    apply_demo_theme(st.session_state.theme_name)

    st.sidebar.markdown(
        """
        <div class="brand">
          <h3>AI That Learns You</h3>
          <p>Study Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<span class='label'>Theme</span>", unsafe_allow_html=True)
    st.sidebar.selectbox("Theme", list(THEMES.keys()), key="theme_name", label_visibility="collapsed")
    st.sidebar.markdown("<span class='label'>Navigate</span>", unsafe_allow_html=True)
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed", key="nav_page")

    st.markdown(
        """
        <div class="hero">
          <h2>Personal Study Intelligence</h2>
          <p>Predict, analyze, adapt, and plan with your own evolving data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    base_df = safe_read_csv(BASE_DATA_PATH)
    history_df = safe_read_csv(HISTORY_PATH)
    full_df = build_full_training_data(base_df, history_df)

    bootstrap_models_if_missing(base_df, history_df, MODEL_DIR)
    if "models" not in st.session_state:
        st.session_state.models = load_models(MODEL_DIR)
    models = st.session_state.models

    profile = load_profile(PROFILE_PATH)
    rl_memory = load_rl_memory(RL_MEMORY_PATH)
    ensure_pdf_state()
    ensure_technique_state()

    if page == "Predict":
        render_predict_page(base_df, full_df, models, profile, rl_memory)
    elif page == "History":
        render_history_page()
    elif page == "PDF Analyzer":
        render_pdf_analyzer_page()
    elif page == "Quiz Converter":
        render_quiz_converter_page()
    elif page == "Technique Checker":
        render_technique_checker_page(full_df)
    elif page == "RL Planner":
        render_rl_planner_page(full_df, rl_memory)
    elif page == "Focus Session":
        render_focus_session_page()
    elif page == "Dashboard":
        render_dashboard_page(base_df, history_df, full_df)


if __name__ == "__main__":
    main()

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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


def apply_demo_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=Mulish:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');
        :root{
          --bg:#070810;--panel:#0c0d1a;--card:#111224;--border:#1c1d35;
          --accent:#5b5ef4;--teal:#00d9c0;--amber:#f59e0b;--green:#10b981;
          --text:#eeeeff;--muted:#6b6b9a;--soft:#a0a0cc;
        }
        .stApp{
          background:
            radial-gradient(ellipse 800px 500px at 10% 20%, rgba(91,94,244,.08) 0%, transparent 70%),
            radial-gradient(ellipse 500px 300px at 90% 70%, rgba(0,217,192,.06) 0%, transparent 70%),
            var(--bg);
          color:var(--text);
          font-family:'Mulish', sans-serif;
        }
        h1,h2,h3{
          font-family:'Syne', sans-serif !important;
          color:var(--text) !important;
          letter-spacing:.02em;
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
        div[data-baseweb="tab-list"]{
          gap:.35rem;
          background:rgba(12,13,26,.8);
          padding:.3rem;
          border:1px solid var(--border);
          border-radius:10px;
        }
        button[data-baseweb="tab"]{
          background:transparent !important;
          border-radius:8px !important;
          border:1px solid transparent !important;
          color:var(--soft) !important;
          font-family:'Syne', sans-serif !important;
          font-size:.78rem !important;
          letter-spacing:.04em !important;
          text-transform:uppercase !important;
        }
        button[data-baseweb="tab"][aria-selected="true"]{
          background:var(--card) !important;
          border-color:var(--border) !important;
          color:var(--text) !important;
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
          font-size:.84rem;
          letter-spacing:.03em;
        }
        .stButton > button:hover{
          filter:brightness(1.05);
        }
        .label{
          font-size:.67rem;
          color:var(--muted);
          text-transform:uppercase;
          letter-spacing:.12em;
          margin-bottom:.3rem;
          display:block;
        }
        .note{
          color:var(--soft);
          font-size:.86rem;
          line-height:1.6;
        }
        </style>
        """,
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
    if subject_score < global_avg:
        return 1.1
    return 0.95


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

    fig = plt.figure(figsize=(5.5, 4.6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.22, color=color)
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(chart_data["title"])
    st.pyplot(fig)


def ensure_pdf_state():
    st.session_state.setdefault("pdf_multiplier", 1.0)
    st.session_state.setdefault("pdf_result", {})
    st.session_state.setdefault("pdf_text", "")
    st.session_state.setdefault("pdf_quiz", [])


def render_pdf_analyzer_tab():
    ensure_pdf_state()
    st.markdown("<span class='label'>PDF Analyzer</span>", unsafe_allow_html=True)
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
        st.markdown("<p class='note'>Upload and analyze a PDF to activate adaptive planning.</p>", unsafe_allow_html=True)


def render_quiz_converter_tab():
    ensure_pdf_state()
    st.markdown("<span class='label'>PDF to Quiz and Question Converter</span>", unsafe_allow_html=True)
    if st.session_state.pdf_text:
        count = st.slider("Number of quiz questions", 3, 15, 6, key="quiz_count")
        if st.button("Generate Questions"):
            st.session_state.pdf_quiz = generate_quiz_from_text(st.session_state.pdf_text, num_questions=count)

        quiz_items = st.session_state.pdf_quiz
        if quiz_items:
            quiz_df = pd.DataFrame(quiz_items)
            st.dataframe(quiz_df, use_container_width=True)
            pdf_bytes = quiz_to_pdf_bytes(quiz_items, title="AI Learns You - Quiz Set")
            st.download_button(
                "Download Quiz PDF",
                data=pdf_bytes,
                file_name="pdf_quiz_questions.pdf",
                mime="application/pdf",
            )
        else:
            st.markdown("<p class='note'>Generate questions to preview and export.</p>", unsafe_allow_html=True)
    else:
        st.info("Analyze a PDF first in the PDF Analyzer tab.")


def main():
    st.set_page_config(page_title="AI Learns You", layout="wide")
    apply_demo_theme()
    st.markdown(
        """
        <div class="hero">
          <h2>AI That Learns You</h2>
          <p>Predict, analyze, adapt, and plan with a personalized intelligence stack.</p>
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

    tabs = st.tabs(
        [
            "Predict",
            "History",
            "PDF Analyzer",
            "Quiz Converter",
            "Technique Checker",
            "RL Planner",
            "Dashboard",
        ]
    )

    with tabs[0]:
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

            predict_clicked = st.button("Predict Productivity + Learn")

        with right:
            if "last_prediction" in st.session_state:
                pred_data = st.session_state.last_prediction
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Productivity", f"{pred_data['pred']:.2f}/10")
                c2.metric("Recommended Hours", f"{pred_data['rec']:.2f}h")
                c3.metric("Forecast", f"{pred_data['forecast']:.2f}/10")
                st.info(pred_data["rl_tip"])
            else:
                st.markdown(
                    "<p class='note'>Run a prediction to see adaptive recommendation, RL action, and forecast.</p>",
                    unsafe_allow_html=True,
                )

        if predict_clicked:
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

            history_df = safe_read_csv(HISTORY_PATH)
            full_df = build_full_training_data(base_df, history_df)

            retrained = maybe_retrain_models(full_df, MODEL_DIR, min_rows=20)
            if retrained:
                st.session_state.models = load_models(MODEL_DIR)

            profile = update_profile(PROFILE_PATH, full_df)
            st.session_state.last_prediction = {
                "pred": pred_value,
                "rec": recommended_hours,
                "forecast": forecast,
                "rl_tip": f"RL action picked: {rl_action_name}. Recommendation used profile and PDF complexity signals.",
            }
            st.rerun()

    with tabs[1]:
        st.markdown("<span class='label'>Session History</span>", unsafe_allow_html=True)
        history_df = safe_read_csv(HISTORY_PATH)
        if len(history_df) == 0:
            st.info("No sessions logged yet.")
        else:
            show_cols = [col for col in ["date", "subject", "hours_studied", "focus_level", "distractions", "sleep_hours", "productivity", "recommended_hours"] if col in history_df.columns]
            st.dataframe(history_df[show_cols].tail(30).iloc[::-1], use_container_width=True)

    with tabs[2]:
        render_pdf_analyzer_tab()

    with tabs[3]:
        render_quiz_converter_tab()

    with tabs[4]:
        history_df = safe_read_csv(HISTORY_PATH)
        full_df = build_full_training_data(base_df, history_df)
        st.markdown("<span class='label'>Technique Checker</span>", unsafe_allow_html=True)
        subject_options = sorted(full_df["subject"].dropna().astype(str).unique().tolist()) if len(full_df) else DEFAULT_SUBJECTS
        selected_subject = st.selectbox("Subject", subject_options, key="technique_subject")
        technique_report = check_study_methods(full_df, subject=selected_subject)
        st.write(technique_report["summary"])

        technique_map = {item["method"]: item for item in technique_report["methods"]}
        selected_technique = st.selectbox("Technique fit breakdown", list(technique_map.keys()), key="technique_picker")
        selected_info = technique_map[selected_technique]
        st.metric("Fit Score", f"{selected_info['score']}/100")
        st.write(selected_info["reason"])
        st.write("Top technique suggestions:")
        for row in technique_report["methods"][:3]:
            st.write(f"- {row['method']}: {row['score']}/100")

    with tabs[5]:
        history_df = safe_read_csv(HISTORY_PATH)
        full_df = build_full_training_data(base_df, history_df)
        st.markdown("<span class='label'>RL Planner</span>", unsafe_allow_html=True)

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

        if "rl_plan_df" in st.session_state and len(st.session_state.rl_plan_df):
            plan_df = st.session_state.rl_plan_df
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Planned Days", int(plan_df["Day"].max()))
            c2.metric("Total Hours", round(float(plan_df["Hours"].sum()), 2))
            c3.metric("Top Subject", plan_df["Subject"].value_counts().idxmax())
            c4.metric("Avg Pred Score", round(float(plan_df["Predicted Productivity"].mean()), 2))
            st.dataframe(plan_df, use_container_width=True)
        else:
            st.markdown("<p class='note'>Generate a plan to view RL-driven slots and techniques.</p>", unsafe_allow_html=True)

    with tabs[6]:
        history_df = safe_read_csv(HISTORY_PATH)
        full_df = build_full_training_data(base_df, history_df)
        if len(full_df) == 0:
            st.warning("No data found yet. Add at least one session.")
        else:
            profile = update_profile(PROFILE_PATH, full_df)
            st.markdown("<span class='label'>Dashboard</span>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Productivity", round(float(full_df["productivity"].mean()), 2))
            m2.metric("Total Sessions", int(len(full_df)))
            m3.metric("Consistency Score", compute_consistency_score(full_df))

            st.subheader("Trend Comparison: You vs Dataset Average")
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
                render_spider_chart(compute_spider_metrics(full_df, subject=None), color="#5b5ef4")
            with right:
                st.subheader("Spider Chart - Per Subject")
                subject_options = sorted(full_df["subject"].dropna().astype(str).unique().tolist())
                selected_subject = st.selectbox(
                    "Select subject",
                    subject_options if subject_options else DEFAULT_SUBJECTS,
                    key="dashboard_spider_subject",
                )
                render_spider_chart(compute_spider_metrics(full_df, subject=selected_subject), color="#00d9c0")

            st.subheader("Learns-You Memory")
            st.write(f"Remembered sessions: {profile['total_sessions']}")
            st.write(f"Best performance window: {profile['best_time_bucket']}")
            st.write(f"Top subject trend: {profile['top_subject']}")


if __name__ == "__main__":
    main()

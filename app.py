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
    check_study_methods,
    compute_spider_metrics,
    compute_consistency_score,
    detect_behavior_patterns,
    extract_pdf_text,
    forecast_with_recommendation,
    generate_quiz_from_text,
    load_models,
    load_profile,
    make_productivity_input,
    make_recommendation_input,
    maybe_retrain_models,
    safe_read_csv,
    update_profile,
)


MODEL_DIR = Path("models")
BASE_DATA_PATH = Path("data/study_data.csv")
HISTORY_PATH = Path("data/user_history.csv")
PROFILE_PATH = Path("data/user_profile.json")


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


def render_spider_chart(chart_data):
    labels = chart_data["labels"]
    values = chart_data["values"]
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(chart_data["title"])
    st.pyplot(fig)


def run_pdf_analyzer():
    st.subheader("PDF Analyzer")
    uploaded_pdf = st.file_uploader("Upload study material PDF", type=["pdf"], key="study_pdf")
    if "pdf_multiplier" not in st.session_state:
        st.session_state.pdf_multiplier = 1.0
    if "pdf_result" not in st.session_state:
        st.session_state.pdf_result = {}
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    if "pdf_quiz" not in st.session_state:
        st.session_state.pdf_quiz = []

    if uploaded_pdf is not None and st.button("Analyze PDF"):
        text = extract_pdf_text(uploaded_pdf)
        if not text:
            st.error("No readable text found in PDF.")
            return
        st.session_state.pdf_text = text
        classifier = get_complexity_classifier()
        result = analyze_pdf_complexity(text, classifier)
        st.session_state.pdf_multiplier = result["effort_multiplier"]
        st.session_state.pdf_result = result

    result = st.session_state.pdf_result
    if result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Complexity Score", result["complexity_score"])
        c2.metric("Difficulty", result["difficulty_label"])
        c3.metric("Effort Multiplier", result["effort_multiplier"])
        st.write("Keywords:", ", ".join(result.get("keywords", [])) or "None")

    st.subheader("PDF to Quiz and Question Converter")
    if st.session_state.pdf_text:
        num_questions = st.slider("Number of quiz questions", 3, 15, 6, key="quiz_count")
        if st.button("Generate Quiz from PDF"):
            st.session_state.pdf_quiz = generate_quiz_from_text(
                st.session_state.pdf_text,
                num_questions=num_questions,
            )

        if st.session_state.pdf_quiz:
            quiz_df = pd.DataFrame(st.session_state.pdf_quiz)
            st.dataframe(quiz_df, use_container_width=True)
            st.download_button(
                "Download Questions CSV",
                data=quiz_df.to_csv(index=False).encode("utf-8"),
                file_name="pdf_quiz_questions.csv",
                mime="text/csv",
            )
    else:
        st.caption("Analyze a PDF first to enable quiz generation.")


def main():
    st.title("AI Learns You - Adaptive Study Intelligence")
    st.write("Prediction + tracking + pattern learning + adaptive recommendations.")

    base_df = safe_read_csv(BASE_DATA_PATH)
    history_df = safe_read_csv(HISTORY_PATH)
    full_df = build_full_training_data(base_df, history_df)

    bootstrap_models_if_missing(base_df, history_df, MODEL_DIR)
    if "models" not in st.session_state:
        st.session_state.models = load_models(MODEL_DIR)
    models = st.session_state.models

    profile = load_profile(PROFILE_PATH)

    run_pdf_analyzer()

    st.subheader("Session Input")
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

    if st.button("Predict + Learn"):
        prod_input = make_productivity_input(
            hours=hours,
            focus=focus,
            distractions=distractions,
            sleep=sleep,
            subject=subject,
            columns=models["productivity_columns"],
        )
        pred_value = float(models["productivity_model"].predict(prod_input)[0])
        st.success(f"Predicted Productivity: {pred_value:.2f} / 10")

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

        st.info(
            "ML recommendation: "
            f"{recommended_hours:.2f} hrs/day for {subject} "
            f"(raw={raw_recommendation:.2f}, profile={adaptive_factor:.2f}, pdf={pdf_factor:.2f})"
        )
        st.write(f"Forecast productivity if you follow this plan: {forecast:.2f} / 10")

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
            st.success("Auto-retraining complete: models updated with your latest behavior.")

        profile = update_profile(PROFILE_PATH, full_df)
        st.success("System memory updated.")

    history_df = safe_read_csv(HISTORY_PATH)
    full_df = build_full_training_data(base_df, history_df)
    if len(full_df) == 0:
        st.warning("No data found yet. Add at least one session.")
        return

    profile = update_profile(PROFILE_PATH, full_df)

    st.subheader("Intelligence Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Productivity", round(float(full_df["productivity"].mean()), 2))
    c2.metric("Total Sessions", int(len(full_df)))
    c3.metric("Consistency Score", compute_consistency_score(full_df))

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

    st.subheader("Spider Chart - Overall")
    overall_chart = compute_spider_metrics(full_df, subject=None)
    render_spider_chart(overall_chart)

    st.subheader("Spider Chart - Per Subject")
    subject_options = sorted(full_df["subject"].dropna().astype(str).unique().tolist())
    default_subject = st.session_state.get("subject", subject_options[0] if subject_options else DEFAULT_SUBJECTS[0])
    subject_for_spider = st.selectbox(
        "Select subject for radar",
        subject_options if subject_options else DEFAULT_SUBJECTS,
        index=(subject_options.index(default_subject) if subject_options and default_subject in subject_options else 0),
        key="spider_subject",
    )
    subject_chart = compute_spider_metrics(full_df, subject=subject_for_spider)
    render_spider_chart(subject_chart)

    st.subheader("Study Method Checker")
    method_report = check_study_methods(full_df, subject=subject_for_spider)
    st.write(method_report["summary"])
    method_lookup = {item["method"]: item for item in method_report["methods"]}
    selected_method = st.selectbox("Check method fit", list(method_lookup.keys()), key="method_fit_select")
    selected_info = method_lookup[selected_method]
    st.metric("Method Fit Score", selected_info["score"])
    st.write(selected_info["reason"])
    st.write("Top recommendations right now:")
    for item in method_report["methods"][:3]:
        st.write(f"- {item['method']}: {item['score']}/100")

    st.subheader("Learns-You Memory")
    st.write(f"Remembered sessions: {profile['total_sessions']}")
    st.write(f"Best performance window: {profile['best_time_bucket']}")
    st.write(f"Top subject trend: {profile['top_subject']}")

    subject_perf = full_df.groupby("subject")["productivity"].mean().sort_values(ascending=False)
    st.subheader("Subject Performance")
    st.bar_chart(subject_perf)

    st.subheader("Weekly Productivity Trend")
    week_df = add_time_columns(full_df)
    week_df["week"] = week_df["event_time"].dt.to_period("W").astype(str)
    weekly = week_df.groupby("week")["productivity"].mean()
    st.line_chart(weekly)


if __name__ == "__main__":
    main()

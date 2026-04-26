import json
import re
from io import BytesIO
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfReader
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DEFAULT_SUBJECTS = ["DSA", "OOP", "Maths", "Physics", "History"]


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def normalize_dataframe(df):
    expected = ["hours_studied", "focus_level", "distractions", "sleep_hours", "subject", "productivity"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df


def attach_event_time(df):
    df = df.copy()
    if "timestamp" in df.columns:
        df["event_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "date" in df.columns:
        df["event_time"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["event_time"] = pd.NaT
    missing = df["event_time"].isna()
    if missing.any():
        fallback = pd.date_range(end=datetime.now(), periods=len(df), freq="h")
        df.loc[missing, "event_time"] = fallback[: missing.sum()].values
    return df


def build_full_training_data(base_df, history_df):
    base_df = normalize_dataframe(base_df)
    history_df = normalize_dataframe(history_df)
    full_df = pd.concat([base_df, history_df], ignore_index=True)
    full_df = full_df.dropna(subset=["hours_studied", "focus_level", "distractions", "sleep_hours", "subject", "productivity"])
    full_df = attach_event_time(full_df)
    return full_df


def train_productivity_model(full_df):
    train_df = full_df[["hours_studied", "focus_level", "distractions", "sleep_hours", "subject", "productivity"]].copy()
    train_df["subject"] = train_df["subject"].astype(str)
    train_df = pd.get_dummies(train_df, columns=["subject"])
    x = train_df.drop(columns=["productivity"])
    y = train_df["productivity"]
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(x, y)
    return model, x.columns.tolist()


def build_recommendation_training_data(full_df):
    rec_df = full_df[["subject", "focus_level", "productivity", "hours_studied", "event_time"]].copy()
    rec_df = rec_df.sort_values("event_time").reset_index(drop=True)
    rec_df["past_productivity"] = rec_df["productivity"].shift(1)
    rec_df["past_productivity"] = rec_df["past_productivity"].fillna(rec_df["productivity"].mean())
    x = rec_df[["subject", "focus_level", "past_productivity"]].copy()
    x["subject"] = x["subject"].astype(str)
    x = pd.get_dummies(x, columns=["subject"])
    y = rec_df["hours_studied"].clip(lower=1, upper=10)
    return x, y


def train_recommendation_model(full_df):
    x, y = build_recommendation_training_data(full_df)
    model = RandomForestRegressor(n_estimators=220, random_state=42)
    model.fit(x, y)
    return model, x.columns.tolist()


def align_columns(frame, columns):
    aligned = frame.copy()
    for col in columns:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned[columns]


def make_productivity_input(hours, focus, distractions, sleep, subject, columns):
    row = pd.DataFrame(
        [
            {
                "hours_studied": hours,
                "focus_level": focus,
                "distractions": distractions,
                "sleep_hours": sleep,
                "subject": subject,
            }
        ]
    )
    row = pd.get_dummies(row, columns=["subject"])
    return align_columns(row, columns)


def make_recommendation_input(subject, focus, past_productivity, columns):
    row = pd.DataFrame(
        [{"subject": subject, "focus_level": focus, "past_productivity": past_productivity}]
    )
    row = pd.get_dummies(row, columns=["subject"])
    return align_columns(row, columns)


def maybe_retrain_models(full_df, model_dir, min_rows=20):
    if len(full_df) < min_rows:
        return False
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    productivity_model, productivity_columns = train_productivity_model(full_df)
    recommendation_model, recommendation_columns = train_recommendation_model(full_df)

    joblib.dump(productivity_model, model_dir / "model.pkl")
    joblib.dump(productivity_columns, model_dir / "columns.pkl")
    joblib.dump(recommendation_model, model_dir / "recommendation_model.pkl")
    joblib.dump(recommendation_columns, model_dir / "recommendation_columns.pkl")
    return True


def load_models(model_dir):
    model_dir = Path(model_dir)
    return {
        "productivity_model": joblib.load(model_dir / "model.pkl"),
        "productivity_columns": joblib.load(model_dir / "columns.pkl"),
        "recommendation_model": joblib.load(model_dir / "recommendation_model.pkl"),
        "recommendation_columns": joblib.load(model_dir / "recommendation_columns.pkl"),
    }


def bootstrap_models_if_missing(base_df, history_df, model_dir):
    model_dir = Path(model_dir)
    required = [
        model_dir / "model.pkl",
        model_dir / "columns.pkl",
        model_dir / "recommendation_model.pkl",
        model_dir / "recommendation_columns.pkl",
    ]
    if all(p.exists() for p in required):
        return
    full_df = build_full_training_data(base_df, history_df)
    maybe_retrain_models(full_df, model_dir, min_rows=1)


def build_complexity_classifier():
    simple_texts = [
        "A fraction has a numerator and denominator. You can simplify by dividing both.",
        "Photosynthesis is how plants make food using sunlight and water.",
        "Force equals mass times acceleration in basic physics examples.",
        "Variables store values in a program and can be updated with new values.",
        "A data structure can hold numbers in order, like a list or array.",
        "Practice loops and conditions to solve beginner coding questions.",
    ]
    complex_texts = [
        "Asymptotic analysis of recursive divide-and-conquer algorithms requires formal recurrence solving.",
        "Gradient descent convergence depends on curvature, conditioning, and adaptive learning rates.",
        "Quantum mechanical operators in Hilbert spaces demand linear algebraic rigor and abstraction.",
        "A rigorous proof of NP-completeness relies on polynomial-time reductions and verifier bounds.",
        "Thermodynamic equilibrium under non-ideal conditions requires partial derivative constraints.",
        "Eigenvalue decomposition of covariance matrices supports dimensionality reduction and optimization.",
    ]
    train_texts = simple_texts + complex_texts
    labels = [0] * len(simple_texts) + [1] * len(complex_texts)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(train_texts, labels)
    return pipeline


def extract_pdf_text(uploaded_pdf):
    try:
        reader = PdfReader(uploaded_pdf)
        chunks = []
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def _clean_text(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def analyze_pdf_complexity(text, complexity_classifier):
    text = _clean_text(text)
    if not text:
        return {
            "complexity_score": 0.0,
            "effort_multiplier": 1.0,
            "difficulty_label": "No text detected",
            "keywords": [],
            "stats": {},
        }

    words = re.findall(r"[a-zA-Z]+", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]

    avg_sentence_len = len(words) / max(len(sentences), 1)
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    long_word_ratio = sum(1 for w in words if len(w) >= 8) / max(len(words), 1)

    prob_complex = float(complexity_classifier.predict_proba([text])[0][1])
    stat_score = np.clip(
        (avg_sentence_len / 26.0) * 3.5 + (avg_word_len / 8.0) * 2.0 + unique_ratio * 2.5 + long_word_ratio * 2.0,
        0,
        10,
    )
    ml_score = prob_complex * 10.0
    complexity_score = round(float(0.65 * ml_score + 0.35 * stat_score), 2)

    if complexity_score < 4:
        difficulty = "Easy"
    elif complexity_score < 7:
        difficulty = "Moderate"
    else:
        difficulty = "Hard"

    effort_multiplier = round(float(np.clip(0.8 + (complexity_score / 10.0) * 0.8, 0.8, 1.6)), 2)

    vec = complexity_classifier.named_steps["tfidf"].transform([text]).toarray()[0]
    vocab = np.array(complexity_classifier.named_steps["tfidf"].get_feature_names_out())
    top_idx = vec.argsort()[-8:][::-1]
    keywords = [vocab[i] for i in top_idx if vec[i] > 0][:5]

    return {
        "complexity_score": complexity_score,
        "effort_multiplier": effort_multiplier,
        "difficulty_label": difficulty,
        "keywords": keywords,
        "stats": {
            "avg_sentence_len": round(float(avg_sentence_len), 2),
            "avg_word_len": round(float(avg_word_len), 2),
            "unique_word_ratio": round(float(unique_ratio), 3),
            "long_word_ratio": round(float(long_word_ratio), 3),
        },
    }


def compute_spider_metrics(full_df, subject=None):
    work = full_df.copy()
    if subject:
        subset = work[work["subject"] == subject]
        if len(subset) > 0:
            work = subset

    if len(work) == 0:
        return {
            "title": f"{subject or 'Overall'} Study Profile",
            "labels": ["Hours", "Focus", "Sleep", "Productivity", "Distraction Control"],
            "values": [0, 0, 0, 0, 0],
        }

    hours_score = float(np.clip(work["hours_studied"].mean(), 0, 10))
    focus_score = float(np.clip(work["focus_level"].mean(), 0, 10))
    sleep_score = float(np.clip(work["sleep_hours"].mean(), 0, 10))
    productivity_score = float(np.clip(work["productivity"].mean(), 0, 10))
    distraction_control = float(np.clip(10 - work["distractions"].mean(), 0, 10))

    return {
        "title": f"{subject or 'Overall'} Study Profile",
        "labels": ["Hours", "Focus", "Sleep", "Productivity", "Distraction Control"],
        "values": [
            round(hours_score, 2),
            round(focus_score, 2),
            round(sleep_score, 2),
            round(productivity_score, 2),
            round(distraction_control, 2),
        ],
    }


def check_study_methods(full_df, subject=None):
    work = full_df.copy()
    if subject:
        subject_df = work[work["subject"] == subject]
        if len(subject_df) > 0:
            work = subject_df

    if len(work) == 0:
        return {"methods": [], "summary": "No data to evaluate study methods yet."}

    avg_focus = float(work["focus_level"].mean())
    avg_distractions = float(work["distractions"].mean())
    avg_hours = float(work["hours_studied"].mean())
    avg_prod = float(work["productivity"].mean())
    consistency = compute_consistency_score(work) / 100.0

    active_recall = np.clip(50 + (7 - avg_prod) * 8 + (avg_focus - 5) * 3, 0, 100)
    spaced_repetition = np.clip(45 + (1 - consistency) * 35 + max(0, avg_hours - 4) * 5, 0, 100)
    pomodoro = np.clip(35 + avg_distractions * 6 + max(0, 6 - avg_focus) * 8, 0, 100)
    practice_testing = np.clip(40 + (7 - avg_prod) * 9 + max(0, 5 - avg_hours) * 4, 0, 100)
    feynman = np.clip(35 + max(0, 7 - avg_prod) * 7 + max(0, 6 - avg_focus) * 4, 0, 100)

    methods = [
        {
            "method": "Active Recall",
            "score": round(float(active_recall), 1),
            "reason": "Improves long-term retention and closes concept gaps quickly.",
        },
        {
            "method": "Spaced Repetition",
            "score": round(float(spaced_repetition), 1),
            "reason": "Best when consistency fluctuates and content load is large.",
        },
        {
            "method": "Pomodoro Blocks",
            "score": round(float(pomodoro), 1),
            "reason": "Useful when distractions are high or focus drops in long sessions.",
        },
        {
            "method": "Practice Testing",
            "score": round(float(practice_testing), 1),
            "reason": "Converts passive study into measurable performance gains.",
        },
        {
            "method": "Feynman Technique",
            "score": round(float(feynman), 1),
            "reason": "Great for hard topics where conceptual clarity is missing.",
        },
    ]
    methods = sorted(methods, key=lambda item: item["score"], reverse=True)
    summary = f"Top fit right now: {methods[0]['method']} ({methods[0]['score']}/100)."
    return {"methods": methods, "summary": summary}


_QUIZ_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "have",
    "your",
    "into",
    "using",
    "between",
    "their",
    "which",
    "where",
    "when",
    "while",
    "also",
    "such",
    "than",
    "there",
}


def _split_sentences(text):
    cleaned = _clean_text(text)
    chunks = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = []
    for sentence in chunks:
        words = sentence.split()
        if 8 <= len(words) <= 40:
            sentences.append(sentence.strip())
    return sentences


def _pick_keyword(sentence):
    candidates = re.findall(r"[A-Za-z]{5,}", sentence)
    candidates = [word for word in candidates if word.lower() not in _QUIZ_STOPWORDS]
    if not candidates:
        return ""
    candidates = sorted(candidates, key=len, reverse=True)
    return candidates[0]


def generate_quiz_from_text(text, num_questions=6):
    sentences = _split_sentences(text)
    if not sentences:
        return []

    questions = []
    for idx, sentence in enumerate(sentences):
        if len(questions) >= num_questions:
            break
        keyword = _pick_keyword(sentence)
        if keyword and idx % 2 == 0:
            blanked = re.sub(rf"\b{re.escape(keyword)}\b", "_____", sentence, count=1)
            questions.append(
                {
                    "type": "Fill in the blank",
                    "question": blanked,
                    "answer": keyword,
                }
            )
        else:
            prompt_seed = " ".join(sentence.split()[:10])
            questions.append(
                {
                    "type": "Short answer",
                    "question": f"Explain this statement: {prompt_seed}...",
                    "answer": sentence,
                }
            )
    return questions


def quiz_to_pdf_bytes(questions, title="PDF Quiz Export"):
    if not questions:
        return b""

    buffer = BytesIO()
    questions_per_page = 5

    with PdfPages(buffer) as pdf:
        total_pages = int(np.ceil(len(questions) / questions_per_page))
        for page_idx in range(total_pages):
            start = page_idx * questions_per_page
            end = min(len(questions), start + questions_per_page)
            page_questions = questions[start:end]

            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")

            y = 0.96
            ax.text(0.02, y, title, fontsize=16, fontweight="bold", va="top")
            y -= 0.04
            ax.text(0.02, y, f"Page {page_idx + 1} of {total_pages}", fontsize=9, color="#444444", va="top")
            y -= 0.04

            for local_idx, q in enumerate(page_questions, start=start + 1):
                q_type = q.get("type", "Question")
                question = q.get("question", "").strip()
                answer = q.get("answer", "").strip()
                block = (
                    f"{local_idx}. [{q_type}] {question}\n"
                    f"Answer: {answer}"
                )
                ax.text(0.02, y, block, fontsize=10, va="top", wrap=True)
                y -= 0.17

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


RL_ACTIONS = [
    {"name": "Deep Focus Block", "technique": "Active Recall", "slot_hours": 1.5, "intensity": 5},
    {"name": "Pomodoro Sprint", "technique": "Pomodoro", "slot_hours": 1.0, "intensity": 3},
    {"name": "Problem Solving Drill", "technique": "Practice Testing", "slot_hours": 1.5, "intensity": 4},
    {"name": "Concept Revision Loop", "technique": "Spaced Repetition", "slot_hours": 1.0, "intensity": 2},
    {"name": "Teach Back Session", "technique": "Feynman Technique", "slot_hours": 1.25, "intensity": 4},
]


def _bucket(value, step):
    return int(np.clip(np.floor(float(value) / step), 0, 9))


def build_rl_state(subject, focus_level, past_productivity, distractions):
    return f"{subject}|f{_bucket(focus_level, 2)}|p{_bucket(past_productivity, 2)}|d{_bucket(distractions, 2)}"


def load_rl_memory(path):
    p = Path(path)
    if not p.exists():
        return {"states": {}, "last_update": ""}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"states": {}, "last_update": ""}


def save_rl_memory(path, memory):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    memory["last_update"] = datetime.now().isoformat()
    p.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def _ensure_rl_state(memory, state):
    if "states" not in memory:
        memory["states"] = {}
    if state not in memory["states"]:
        memory["states"][state] = {action["name"]: 0.0 for action in RL_ACTIONS}


def choose_rl_action(memory, state, epsilon=0.12):
    _ensure_rl_state(memory, state)
    q_values = memory["states"][state]
    if np.random.random() < epsilon:
        picked = np.random.choice(list(q_values.keys()))
    else:
        picked = max(q_values.items(), key=lambda item: item[1])[0]
    return picked, q_values


def update_rl_q(memory, state, action_name, reward, alpha=0.25, gamma=0.8):
    _ensure_rl_state(memory, state)
    q_values = memory["states"][state]
    old_value = float(q_values.get(action_name, 0.0))
    max_future = max(q_values.values()) if q_values else 0.0
    updated = old_value + alpha * (float(reward) + gamma * max_future - old_value)
    q_values[action_name] = round(float(updated), 4)
    return q_values[action_name]


def _subject_need_scores(full_df, priority_subject=None):
    if len(full_df) == 0:
        base = {subj: 5.0 for subj in DEFAULT_SUBJECTS}
    else:
        subj_perf = full_df.groupby("subject")["productivity"].mean().to_dict()
        base = {}
        for subj in DEFAULT_SUBJECTS:
            score = float(subj_perf.get(subj, np.mean(list(subj_perf.values())) if subj_perf else 6.0))
            base[subj] = float(np.clip(10 - score, 0.5, 10))
    if priority_subject and priority_subject in base:
        base[priority_subject] = base[priority_subject] * 1.35
    return base


def _action_by_name(name):
    for action in RL_ACTIONS:
        if action["name"] == name:
            return action
    return RL_ACTIONS[0]


def build_rl_study_plan(
    full_df,
    memory,
    hours_per_day=4.0,
    days_to_exam=5,
    priority_subject="Auto",
):
    days = int(np.clip(days_to_exam, 1, 7))
    hpd = float(np.clip(hours_per_day, 1, 10))
    slots_per_day = max(1, int(round(hpd / 1.25)))
    needs = _subject_need_scores(full_df, None if priority_subject == "Auto" else priority_subject)

    plan_rows = []
    for day in range(1, days + 1):
        day_budget = hpd
        for slot in range(1, slots_per_day + 1):
            subjects = list(needs.keys())
            weights = np.array([needs[subj] for subj in subjects], dtype=float)
            weights = weights / weights.sum()
            subject = np.random.choice(subjects, p=weights)

            focus_est = int(np.clip(6 + np.random.randint(-2, 3), 1, 10))
            subject_hist = full_df[full_df["subject"] == subject]["productivity"] if len(full_df) else pd.Series(dtype=float)
            past_prod = float(subject_hist.tail(5).mean()) if len(subject_hist) else 6.0
            dist_est = int(np.clip(3 + np.random.randint(-1, 3), 0, 10))

            state = build_rl_state(subject, focus_est, past_prod, dist_est)
            action_name, q_values = choose_rl_action(memory, state)
            action = _action_by_name(action_name)

            allocated = min(action["slot_hours"], max(0.75, day_budget / max(1, slots_per_day - slot + 1)))
            reward_proxy = np.clip((past_prod / 10.0) + (focus_est / 20.0) - (dist_est / 30.0), 0, 1.5)
            updated_q = update_rl_q(memory, state, action_name, reward_proxy)

            predicted = float(np.clip((past_prod * 0.55) + (focus_est * 0.35) - (dist_est * 0.2) + 1.5, 1, 10))
            plan_rows.append(
                {
                    "Day": day,
                    "Slot": slot,
                    "Subject": subject,
                    "Action": action_name,
                    "Technique": action["technique"],
                    "Hours": round(float(allocated), 2),
                    "Intensity": int(action["intensity"]),
                    "Predicted Productivity": round(predicted, 2),
                    "Q Value": round(float(updated_q), 3),
                }
            )
            day_budget = max(0.0, day_budget - allocated)
            needs[subject] = max(0.4, needs[subject] * 0.8)
    return pd.DataFrame(plan_rows)


def compute_time_bucket(hour_value):
    if 5 <= hour_value <= 11:
        return "Morning"
    if 12 <= hour_value <= 16:
        return "Afternoon"
    if 17 <= hour_value <= 21:
        return "Evening"
    return "Night"


def add_time_columns(df):
    df = df.copy()
    df = attach_event_time(df)
    df["hour"] = df["event_time"].dt.hour.fillna(12).astype(int)
    df["time_bucket"] = df["hour"].apply(compute_time_bucket)
    return df


def compute_consistency_score(df):
    if len(df) < 3:
        return 50.0
    work = add_time_columns(df)
    prod_std = float(work["productivity"].std() or 0)
    stability = max(0.0, 1.0 - min(prod_std / 3.0, 1.0))

    gaps = work["event_time"].sort_values().diff().dt.days.dropna()
    if len(gaps) == 0:
        gap_stability = 0.7
    else:
        gap_std = float(gaps.std() or 0)
        gap_stability = max(0.0, 1.0 - min(gap_std / 4.0, 1.0))

    score = round((0.7 * stability + 0.3 * gap_stability) * 100, 1)
    return float(np.clip(score, 0, 100))


def detect_behavior_patterns(df):
    if len(df) < 5:
        return {
            "best_window": "Need at least 5 sessions for reliable time-window insights.",
            "draining_subject": "Need more sessions.",
            "burnout_signal": "Insufficient recent data.",
        }

    work = add_time_columns(df)
    by_time = work.groupby("time_bucket")["productivity"].mean().sort_values(ascending=False)
    best_window = f"Best window: {by_time.index[0]} (avg productivity {by_time.iloc[0]:.2f})."

    drain_metric = (work.groupby("subject")["productivity"].mean() / work.groupby("subject")["hours_studied"].mean()).sort_values()
    draining_subject = f"Most draining subject: {drain_metric.index[0]} (lowest productivity per study hour)."

    recent = work.sort_values("event_time").tail(6)
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent["productivity"], 1)[0] if len(recent) > 1 else 0.0
    heavy_hours = recent["hours_studied"].mean() >= work["hours_studied"].mean()
    if slope < -0.15 and heavy_hours:
        burnout = "Burnout risk detected: productivity trending down while study load stays high."
    else:
        burnout = "No strong burnout signal in recent sessions."

    return {
        "best_window": best_window,
        "draining_subject": draining_subject,
        "burnout_signal": burnout,
    }


def profile_template():
    return {
        "updated_at": "",
        "total_sessions": 0,
        "best_time_bucket": "Unknown",
        "top_subject": "Unknown",
        "subject_productivity": {},
    }


def load_profile(profile_path):
    p = Path(profile_path)
    if not p.exists():
        return profile_template()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return profile_template()


def update_profile(profile_path, full_df):
    profile = profile_template()
    if len(full_df) == 0:
        return profile

    work = add_time_columns(full_df)
    subject_perf = work.groupby("subject")["productivity"].mean().sort_values(ascending=False)
    bucket_perf = work.groupby("time_bucket")["productivity"].mean().sort_values(ascending=False)

    profile["updated_at"] = datetime.now().isoformat()
    profile["total_sessions"] = int(len(work))
    profile["best_time_bucket"] = bucket_perf.index[0]
    profile["top_subject"] = subject_perf.index[0]
    profile["subject_productivity"] = {k: round(float(v), 3) for k, v in subject_perf.items()}

    p = Path(profile_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def forecast_with_recommendation(productivity_model, productivity_columns, subject, focus, distractions, sleep, recommended_hours):
    row = make_productivity_input(
        hours=recommended_hours,
        focus=focus,
        distractions=distractions,
        sleep=sleep,
        subject=subject,
        columns=productivity_columns,
    )
    return float(productivity_model.predict(row)[0])

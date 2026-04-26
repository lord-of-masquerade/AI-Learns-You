import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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


def parse_voice_command(text, subjects=None):
    subjects = subjects or DEFAULT_SUBJECTS
    lowered = (text or "").lower()
    result = {}

    hours_match = re.search(r"(\d+(?:\.\d+)?)\s*hour", lowered)
    if hours_match:
        result["hours_studied"] = float(hours_match.group(1))

    focus_match = re.search(r"focus(?:\s*level)?(?:\s*is|\s*at)?\s*(\d+)", lowered)
    if focus_match:
        result["focus_level"] = int(focus_match.group(1))
    elif "low focus" in lowered:
        result["focus_level"] = 3
    elif "medium focus" in lowered:
        result["focus_level"] = 6
    elif "high focus" in lowered:
        result["focus_level"] = 8

    distraction_match = re.search(r"distraction(?:s)?(?:\s*is|\s*at)?\s*(\d+)", lowered)
    if distraction_match:
        result["distractions"] = int(distraction_match.group(1))

    sleep_match = re.search(r"(\d+(?:\.\d+)?)\s*hour(?:s)?\s*sleep", lowered)
    if sleep_match:
        result["sleep_hours"] = float(sleep_match.group(1))

    for subject in subjects:
        if subject.lower() in lowered:
            result["subject"] = subject
            break

    return result


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

print("Training started...")

from pathlib import Path

import pandas as pd
try:
    from src.intelligence import (
        build_full_training_data,
        build_recommendation_training_data,
        train_productivity_model,
        train_recommendation_model,
    )
except ImportError:
    from intelligence import (
        build_full_training_data,
        build_recommendation_training_data,
        train_productivity_model,
        train_recommendation_model,
    )


base_df = pd.read_csv("data/study_data.csv")
history_path = Path("data/user_history.csv")
history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()

full_df = build_full_training_data(base_df, history_df)

productivity_model, productivity_columns = train_productivity_model(full_df)
recommendation_model, recommendation_columns = train_recommendation_model(full_df)

# quick in-sample score for visibility
train_df = full_df[["hours_studied", "focus_level", "distractions", "sleep_hours", "subject", "productivity"]].copy()
train_df = pd.get_dummies(train_df, columns=["subject"])
x = train_df.drop(columns=["productivity"])
y = train_df["productivity"]
for col in productivity_columns:
    if col not in x.columns:
        x[col] = 0
x = x[productivity_columns]
score = productivity_model.score(x, y)

Path("models").mkdir(parents=True, exist_ok=True)

import joblib

joblib.dump(productivity_model, "models/model.pkl")
joblib.dump(productivity_columns, "models/columns.pkl")
joblib.dump(recommendation_model, "models/recommendation_model.pkl")
joblib.dump(recommendation_columns, "models/recommendation_columns.pkl")

print(f"Productivity model score: {score:.3f}")

rec_x, rec_y = build_recommendation_training_data(full_df)
for col in recommendation_columns:
    if col not in rec_x.columns:
        rec_x[col] = 0
rec_x = rec_x[recommendation_columns]
rec_score = recommendation_model.score(rec_x, rec_y)
print(f"Recommendation model score: {rec_score:.3f}")

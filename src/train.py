"""
src/train.py
Run from project ROOT:  python src/train.py

Saves:
  models/model.pkl           — trained Random Forest
  models/feature_columns.pkl — EXACT column list used at training time
  models/feature_importance.csv
  models/metrics.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib, os, json
from datetime import datetime

# ── Load & encode ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/study_data.csv")

# One-hot encode subject — produces subject_DSA, subject_History, etc.
df_enc = pd.get_dummies(df, columns=["subject"])

X = df_enc.drop("productivity", axis=1)
y = df_enc["productivity"]

# ── CRITICAL: save the exact column order ─────────────────────────────────────
feature_columns = X.columns.tolist()
print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train ──────────────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
r2        = r2_score(y_test, y_pred)
mae       = mean_absolute_error(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("=" * 44)
print("  Model Evaluation")
print("=" * 44)
print(f"  R²  (test)       : {r2:.4f}")
print(f"  MAE (test)       : {mae:.4f}")
print(f"  R²  (5-fold CV)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 44)

# ── Save everything ────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Model
joblib.dump(model, "models/model.pkl")
joblib.dump(model, f"models/model_{ts}.pkl")

# ✅ Feature column list — app.py reads this to build identical input DataFrames
joblib.dump(feature_columns, "models/feature_columns.pkl")

# Feature importance
pd.DataFrame({
    "feature":    feature_columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).to_csv(
    "models/feature_importance.csv", index=False
)

# Metrics
with open("models/metrics.json", "w") as f:
    json.dump({
        "r2_test":      round(r2,   4),
        "mae_test":     round(mae,  4),
        "cv_r2_mean":   round(cv_scores.mean(), 4),
        "cv_r2_std":    round(cv_scores.std(),  4),
        "trained_at":   ts,
        "n_estimators": model.n_estimators,
        "features":     feature_columns,
    }, f, indent=2)

print(f"\n✅ Saved:")
print(f"   models/model.pkl")
print(f"   models/feature_columns.pkl   ← used by app.py to match features exactly")
print(f"   models/feature_importance.csv")
print(f"   models/metrics.json")

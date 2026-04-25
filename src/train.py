"""
src/train.py — ML training pipeline for "AI That Learns You"
Run: python src/train.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib, os, json
from datetime import datetime

# ── Load & encode ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/study_data.csv")
df_enc = pd.get_dummies(df, columns=["subject"])

X = df_enc.drop("productivity", axis=1)
y = df_enc["productivity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train ─────────────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
r2        = r2_score(y_test, y_pred)
mae       = mean_absolute_error(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("=" * 42)
print("  Model Evaluation")
print("=" * 42)
print(f"  R²  (test)       : {r2:.4f}")
print(f"  MAE (test)       : {mae:.4f}")
print(f"  R²  (5-fold CV)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 42)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

joblib.dump(model, "models/model.pkl")
joblib.dump(model, f"models/model_{ts}.pkl")

pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}) \
  .sort_values("importance", ascending=False) \
  .to_csv("models/feature_importance.csv", index=False)

json.dump({"r2_test": round(r2,4), "mae_test": round(mae,4),
           "cv_r2_mean": round(cv_scores.mean(),4), "cv_r2_std": round(cv_scores.std(),4),
           "trained_at": ts, "features": X.columns.tolist()},
          open("models/metrics.json","w"), indent=2)

print(f"\n✅ Saved → models/model.pkl, models/feature_importance.csv, models/metrics.json")

print("Script started...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load dataset
df = pd.read_csv("data/study_data.csv")

# One-hot encoding
df = pd.get_dummies(df, columns=['subject'])

# Split features and target
X = df.drop('productivity', axis=1)
y = df['productivity']

# Create models folder FIRST
os.makedirs("models", exist_ok=True)

# Save columns (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "models/columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

# Accuracy
print("Accuracy:", model.score(X_test, y_test))
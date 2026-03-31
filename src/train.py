print("Script started...")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/study_data.csv")

X = df[['hours_studied', 'focus_level', 'distractions', 'sleep_hours']]
y = df['productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "models/model.pkl")

print("Model trained and saved!")
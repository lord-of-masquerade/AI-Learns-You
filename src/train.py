print("Script started...")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


df = pd.read_csv("data/study_data.csv")
df=pd.get_dummies(df, columns=['subject'])

X = df.drop('productivity', axis=1)
y = df['productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Accuracy:", model.score(X_test, y_test))

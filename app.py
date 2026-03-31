import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("models/model.pkl")

st.title("🧠 AI That Learns YOU")

st.write("Predict your productivity based on your habits")

# User input
hours = st.slider("Hours Studied", 0, 10)
focus = st.slider("Focus Level", 1, 10)
distractions = st.slider("Distractions", 0, 10)
sleep = st.slider("Sleep Hours", 0, 10)

if st.button("Predict"):
    data = np.array([[hours, focus, distractions, sleep]])
    prediction = model.predict(data)

    st.success(f"Predicted Productivity: {round(prediction[0], 2)} / 10")
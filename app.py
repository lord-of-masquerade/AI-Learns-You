import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/model.pkl")

st.title("🧠 AI That Learns YOU")

st.write("Predict your productivity based on your habits")

# User input
hours = st.slider("Hours Studied", 0, 10)
focus = st.slider("Focus Level", 1, 10)
distractions = st.slider("Distractions", 0, 10)
sleep = st.slider("Sleep Hours", 0, 10)

subject=st.selectbox(("Subject"), ["Math","Physics","DSA","History","OOP"])

input_data = {
    "hours_studied": hours,
    "focus_level": focus,
    "distractions": distractions,
    "sleep_hours": sleep,
    "subject_DSA": 1 if subject == "DSA" else 0,
    "subject_OOP": 1 if subject == "OOP" else 0,
    "subject_Math": 1 if subject == "Math" else 0,
    "subject_Physics": 1 if subject == "Physics" else 0,
    "subject_History": 1 if subject == "History" else 0,
}

if st.button("Predict"):
    data = np.array([[hours, focus, distractions, sleep]])
    prediction = model.predict(data)

    st.success(f"Predicted Productivity: {round(prediction[0], 2)} / 10")

import matplotlib.pyplot as plt
import pandas as pd

st.subheader("📊 Subject Focus Analysis")

df = pd.read_csv("data/study_data.csv")

# Average focus per subject
subject_focus = df.groupby("subject")["focus_level"].mean()

labels = list(subject_focus.index)
values = list(subject_focus.values)

# Close the loop
values += values[:1]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

ax.plot(angles, values)
ax.fill(angles, values, alpha=0.3)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

st.pyplot(fig)

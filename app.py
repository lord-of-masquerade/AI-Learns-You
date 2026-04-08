import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/model.pkl")

# Load dataset
df = pd.read_csv("data/study_data.csv")

st.title("🧠 AI That Learns YOU")
st.write("Adaptive Productivity Prediction System")

# ---------------- INPUT SECTION ---------------- #

hours = st.slider("Hours Studied", 0, 10)
focus = st.slider("Focus Level", 1, 10)
distractions = st.slider("Distractions", 0, 10)
sleep = st.slider("Sleep Hours", 0, 10)

subject = st.selectbox("Subject", ["DSA", "OOP", "Maths", "Physics", "History"])

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    # Prepare input for model
    input_data = {
        "hours_studied": hours,
        "focus_level": focus,
        "distractions": distractions,
        "sleep_hours": sleep,
        "subject_DSA": 1 if subject == "DSA" else 0,
        "subject_OOP": 1 if subject == "OOP" else 0,
        "subject_Maths": 1 if subject == "Maths" else 0,
        "subject_Physics": 1 if subject == "Physics" else 0,
        "subject_History": 1 if subject == "History" else 0
    }

    df_input = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(df_input)
    st.success(f"Predicted Productivity: {round(prediction[0],2)} / 10")

    # ---------------- DYNAMIC DATA ---------------- #

    new_row = {
        "hours_studied": hours,
        "focus_level": focus,
        "distractions": distractions,
        "sleep_hours": sleep,
        "subject": subject,
        "productivity": prediction[0]
    }

    df_temp = pd.concat([df, pd.DataFrame([new_row])])

    # ---------------- SPIDER CHART ---------------- #

    st.subheader("📊 Subject Focus Analysis")

    subject_focus = df_temp.groupby("subject")["focus_level"].mean()

    labels = list(subject_focus.index)
    values = list(subject_focus.values)

    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    st.pyplot(fig)

    # ---------------- WEAK SUBJECT ---------------- #

    st.subheader("📉 Weak Subject Analysis")

    subject_perf = df_temp.groupby("subject")["productivity"].mean()

    weak_subject = subject_perf.idxmin()
    weak_score = subject_perf.min()

    st.error(f"Weakest Subject: {weak_subject} (Score: {round(weak_score,2)})")

    # ---------------- RECOMMENDATION ---------------- #

    st.subheader("⏱️ Study Recommendation")

    if weak_score < 5:
        recommended_hours = 6
    elif weak_score < 7:
        recommended_hours = 5
    else:
        recommended_hours = 3

    st.info(f"Recommended study time for {weak_subject}: {recommended_hours} hrs/day")

    # ---------------- AI SUGGESTIONS ---------------- #

    st.subheader("🤖 AI Recommendations")

    if weak_score < 5:
        st.warning(f"Focus heavily on {weak_subject}. Revise basics.")
    elif weak_score < 7:
        st.warning(f"Practice more in {weak_subject}.")
    else:
        st.success("All subjects are balanced. Keep it up.")

    # ---------------- DAILY TASK PLAN ---------------- #

    st.subheader("📋 Daily Task Plan")

    for sub, score in subject_perf.items():
        if score < 5:
            st.write(f"🔴 {sub}: 2 intense sessions")
        elif score < 7:
            st.write(f"🟡 {sub}: 1 practice + 1 revision")
        else:
            st.write(f"🟢 {sub}: light revision")
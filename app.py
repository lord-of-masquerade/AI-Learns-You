import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")

# ---------------- LOAD DATA ---------------- #

df = pd.read_csv("data/study_data.csv")

# ---------------- SESSION MEMORY ---------------- #

if "user_data" not in st.session_state:
    st.session_state.user_data = []

# ---------------- UI ---------------- #

st.title("🧠 AI That Learns YOU")
st.write("Adaptive Productivity Prediction System")

# ---------------- INPUT ---------------- #

hours = st.slider("Hours Studied", 0, 10)
focus = st.slider("Focus Level", 1, 10)
distractions = st.slider("Distractions", 0, 10)
sleep = st.slider("Sleep Hours", 0, 10)

subject = st.selectbox("Subject", ["DSA","OOP","Maths","Physics","History"])

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    # Create empty input with all columns
    input_df = pd.DataFrame([0]*len(columns), index=columns).T

    # Fill values
    input_df.at[0, "hours_studied"] = hours
    input_df.at[0, "focus_level"] = focus
    input_df.at[0, "distractions"] = distractions
    input_df.at[0, "sleep_hours"] = sleep

    subject_col = f"subject_{subject}"
    if subject_col in input_df.columns:
        input_df.at[0, subject_col] = 1

    # Predict
    prediction = model.predict(input_df)
    pred_value = round(prediction[0], 2)

    st.success(f"Predicted Productivity: {pred_value} / 10")

    # ---------------- STORE USER DATA ---------------- #

    new_row = {
        "hours_studied": hours,
        "focus_level": focus,
        "distractions": distractions,
        "sleep_hours": sleep,
        "subject": subject,
        "productivity": pred_value
    }

    st.session_state.user_data.append(new_row)

    # Convert to DataFrame
    user_df = pd.DataFrame(st.session_state.user_data)

    # Optional: keep last 10 entries (smooth behavior)
    user_df = user_df.tail(10)

    # ---------------- SPIDER CHART ---------------- #

    st.subheader("📊 Subject Focus Analysis (Dynamic)")

    subject_focus = user_df.groupby("subject")["focus_level"].mean()

    labels = list(subject_focus.index)
    values = list(subject_focus.values)

    # Close the loop
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

    subject_perf = user_df.groupby("subject")["productivity"].mean()

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
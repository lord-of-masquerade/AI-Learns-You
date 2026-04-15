import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")

# ---------------- LOAD DATA ---------------- #

df = pd.read_csv("data/study_data.csv")

# ---------------- LOAD USER HISTORY ---------------- #

try:
    history_df = pd.read_csv("data/user_history.csv")
except:
    history_df = pd.DataFrame()

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

    # Create input
    input_df = pd.DataFrame([0]*len(columns), index=columns).T

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

    # ---------------- STORE DATA ---------------- #

    new_row = {
        "hours_studied": hours,
        "focus_level": focus,
        "distractions": distractions,
        "sleep_hours": sleep,
        "subject": subject,
        "productivity": pred_value,
        "date": datetime.now().date()
    }

    st.session_state.user_data.append(new_row)

    # Save to CSV
    pd.DataFrame([new_row]).to_csv(
        "data/user_history.csv",
        mode='a',
        header=not pd.io.common.file_exists("data/user_history.csv"),
        index=False
    )

    # ---------------- DATA PREP ---------------- #

    user_df = pd.DataFrame(st.session_state.user_data)
    user_df = user_df.tail(10)

    if not history_df.empty:
        full_df = pd.concat([history_df, user_df])
    else:
        full_df = user_df.copy()

    # ---------------- SPIDER CHART ---------------- #

    st.subheader("📊 Subject Focus Analysis")

    subject_focus = user_df.groupby("subject")["focus_level"].mean()

    labels = list(subject_focus.index)
    values = list(subject_focus.values)

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

    # ---------------- WEEKLY TREND ---------------- #

    st.subheader("📈 Weekly Productivity Trend")

    if "date" in full_df.columns:
        full_df["date"] = pd.to_datetime(full_df["date"])
        weekly = full_df.groupby("date")["productivity"].mean()
        st.line_chart(weekly)

    # ---------------- DASHBOARD ---------------- #

    st.subheader("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Productivity", round(full_df["productivity"].mean(), 2))
    col2.metric("Best Score", round(full_df["productivity"].max(), 2))
    col3.metric("Total Sessions", len(full_df))

    st.subheader("📊 Subject Performance")

    subject_perf_full = full_df.groupby("subject")["productivity"].mean()
    st.bar_chart(subject_perf_full)

    st.subheader("🧠 Focus vs Productivity")

    if "focus_level" in full_df.columns:
        st.scatter_chart(full_df[["focus_level", "productivity"]])

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
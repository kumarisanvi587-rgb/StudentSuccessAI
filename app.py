import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# Professional dashboard styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #151b35);
    color: #f4f7ff;
}
h1, h2, h3, p, label {
    color: #f4f7ff !important;
}
div[data-testid="stMetric"] {
    background: #202945;
    border: 1px solid #394568;
    padding: 18px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance AI")
st.caption("Machine Learning • Academic Analytics • Smart Study Planning")

@st.cache_data
def create_dataset():
    np.random.seed(42)
    n = 300

    df = pd.DataFrame({
        "Student_ID": range(1, n + 1),
        "Previous_Marks": np.random.randint(35, 96, n),
        "Attendance": np.random.randint(50, 101, n),
        "Study_Hours": np.round(np.random.uniform(1, 8, n), 1),
        "Assignment_Score": np.random.randint(40, 101, n),
        "Sleep_Hours": np.round(np.random.uniform(4, 10, n), 1)
    })

    noise = np.random.normal(0, 5, n)
    df["Next_Semester_Marks"] = np.clip(
        0.45 * df["Previous_Marks"]
        + 0.20 * df["Attendance"]
        + 2.5 * df["Study_Hours"]
        + 0.20 * df["Assignment_Score"]
        + noise,
        0, 100
    ).round(1)

    return df

df = create_dataset()

Path("data").mkdir(exist_ok=True)
df.to_csv("data/student_performance.csv", index=False)

# Train Linear Regression model
features = [
    "Previous_Marks", "Attendance", "Study_Hours",
    "Assignment_Score", "Sleep_Hours"
]
X = df[features]
y = df["Next_Semester_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
predictions = linear_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Train Logistic Regression to classify target achievement
target = (df["Next_Semester_Marks"] >= 75).astype(int)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, target, test_size=0.2, random_state=42, stratify=target
)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_c, y_train_c)
class_predictions = logistic_model.predict(X_test_c)
accuracy = accuracy_score(y_test_c, class_predictions)

# Sidebar navigation
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Marks Predictor", "Study Planner", "Dataset Explorer"]
)

if page == "Overview":
    st.header("📊 Performance Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(df))
    c2.metric("Average Marks", f"{df['Next_Semester_Marks'].mean():.1f}")
    c3.metric("Linear Regression R²", f"{r2:.3f}")
    c4.metric("Logistic Accuracy", f"{accuracy:.1%}")

    fig = px.histogram(
        df, x="Next_Semester_Marks", nbins=20,
        title="Next-Semester Marks Distribution",
        color_discrete_sequence=["#7c83ff"]
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        df, x="Study_Hours", y="Next_Semester_Marks",
        color="Attendance", title="Study Hours vs Predicted Outcome",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info("This project uses synthetic demo data, not real student records.")

elif page == "Marks Predictor":
    st.header("🤖 Predict Student Marks")
    col1, col2 = st.columns(2)

    with col1:
        previous = st.slider("Previous Semester Marks", 0, 100, 65)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        study = st.slider("Daily Study Hours", 0.0, 12.0, 3.0, 0.5)

    with col2:
        assignment = st.slider("Assignment Score", 0, 100, 70)
        sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
        goal = st.slider("Target Marks", 0, 100, 75)

    if st.button("✨ Predict My Performance", type="primary"):
        student = pd.DataFrame([{
            "Previous_Marks": previous,
            "Attendance": attendance,
            "Study_Hours": study,
            "Assignment_Score": assignment,
            "Sleep_Hours": sleep
        }])

        predicted = float(linear_model.predict(student)[0])
        probability = float(logistic_model.predict_proba(student)[0][1])

        st.subheader(f"Predicted Marks: {predicted:.1f} / 100")
        st.progress(int(np.clip(predicted, 0, 100)) / 100)
        st.metric("Estimated Chance of Reaching 75 Marks", f"{probability:.1%}")
        st.metric("Gap to Your Target", f"{max(0, goal - predicted):.1f} marks")

        if predicted >= goal:
            st.success("Your prediction meets your target. Keep practising!")
        else:
            st.warning("Your prediction is below your target. Consider a study plan.")

elif page == "Study Planner":
    st.header("📚 Smart Study Planner")
    current = st.number_input("Current Expected Marks", 0.0, 100.0, 60.0)
    goal = st.number_input("Target Marks", 0.0, 100.0, 80.0)
    available = st.slider("Available Study Hours per Day", 1, 12, 3)

    gap = max(0, goal - current)
    st.metric("Marks Gap", f"{gap:.1f}")

    if gap == 0:
        st.success("Your expected marks meet the target. Focus on revision.")
    else:
        st.write(f"Suggested starting plan: {available} hours daily.")
        st.write("- 40%: Core concepts and weak topics")
        st.write("- 30%: Practice questions")
        st.write("- 20%: Revision and mock tests")
        st.write("- 10%: Review mistakes")
        st.caption("This is a planning suggestion, not a guaranteed marks improvement.")

elif page == "Dataset Explorer":
    st.header("🗂️ Dataset Explorer")
    min_marks, max_marks = st.slider(
        "Filter by Next-Semester Marks",
        0, 100, (0, 100)
    )
    filtered = df[df["Next_Semester_Marks"].between(min_marks, max_marks)]
    st.write(f"Showing {len(filtered)} students")
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="student_performance.csv",
        mime="text/csv"
    )

    st.subheader("Feature Correlation")
    corr = df[features + ["Next_Semester_Marks"]].corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Correlation Heatmap", color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🔍 Model Evaluation"):
    st.write("**Linear Regression**")
    st.write(f"Mean Absolute Error: {mae:.2f} marks")
    st.write(f"R² Score: {r2:.3f}")
    st.write("**Logistic Regression**")
    st.write(f"Accuracy: {accuracy:.1%}")
    st.caption("Metrics are measured on a held-out test split of synthetic data.")

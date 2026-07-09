import streamlit as st
import joblib
import pandas as pd

# ==========================
# Load Model and Scaler
# ==========================
model = joblib.load("employee_salary_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Prediction")
st.write("Enter the employee information to predict the expected salary.")

# ==========================
# User Inputs
# ==========================

age = st.number_input("Age", 18, 65, 25)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education Level",
    [
        "High School",
        "Bachelor's",
        "Master's",
        "PhD"
    ]
)

job = st.selectbox(
    "Job Title",
    [
        "Software Engineer",
        "Data Scientist",
        "Manager",
        "Sales Executive",
        "Marketing Manager"
    ]
)

experience = st.slider(
    "Years of Experience",
    0,
    40,
    5
)

# ==========================
# Manual Encoding
# ==========================

gender = 1 if gender == "Male" else 0

education_dict = {
    "High School":0,
    "Bachelor's":1,
    "Master's":2,
    "PhD":3
}

education = education_dict[education]

job_dict = {
    "Software Engineer":0,
    "Data Scientist":1,
    "Manager":2,
    "Sales Executive":3,
    "Marketing Manager":4
}

job = job_dict[job]

# ==========================
# Prediction
# ==========================

if st.button("Predict Salary"):

    data = pd.DataFrame([[
        age,
        gender,
        education,
        job,
        experience
    ]],
    columns=[
        "Age",
        "Gender",
        "Education Level",
        "Job Title",
        "Years of Experience"
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(
        f"Predicted Salary: ${prediction[0]:,.2f}"
    )
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("columns.pkl")

st.title("Heart Stroke Prediction by Vabin ❤️")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])


chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholestrol = st.number_input("Cholestrol (mg/dL)", 100, 600, 200 )
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina= st.selectbox("Exercise_Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholestrol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0,
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

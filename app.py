import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and encoders
model = joblib.load("lung_cancer_survival_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title( "Lung Cancer Survival Predictor")
st.markdown("Enter the patient's information to predict whether they are likely to survive.")

# Input features
age = st.number_input("Age", min_value=0, max_value=120, value=60)
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
country = st.selectbox("Country", label_encoders['country'].classes_)
cancer_stage = st.selectbox("Cancer Stage", label_encoders['cancer_stage'].classes_)
family_history = st.selectbox("Family History of Cancer", label_encoders['family_history'].classes_)
smoking_status = st.selectbox("Smoking Status", label_encoders['smoking_status'].classes_)
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=22.5)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=500, value=180)
treatment_type = st.selectbox("Treatment Type", label_encoders['treatment_type'].classes_)

# Binary fields (not encoded, just 0/1)
hypertension = 1 if st.selectbox("Hypertension", ["Yes", "No"]) == "Yes" else 0
asthma = 1 if st.selectbox("Asthma", ["Yes", "No"]) == "Yes" else 0
cirrhosis = 1 if st.selectbox("Cirrhosis", ["Yes", "No"]) == "Yes" else 0
other_cancer = 1 if st.selectbox("Other Cancer", ["Yes", "No"]) == "Yes" else 0

# Encode input using LabelEncoders
def encode_input():
    return {
        "age": age,
        "gender": label_encoders['gender'].transform([gender])[0],
        "country": label_encoders['country'].transform([country])[0],
        "cancer_stage": label_encoders['cancer_stage'].transform([cancer_stage])[0],
        "family_history": label_encoders['family_history'].transform([family_history])[0],
        "smoking_status": label_encoders['smoking_status'].transform([smoking_status])[0],
        "bmi": bmi,
        "cholesterol_level": cholesterol,
        "hypertension": hypertension,
        "asthma": asthma,
        "cirrhosis": cirrhosis,
        "other_cancer": other_cancer,
        "treatment_type": label_encoders['treatment_type'].transform([treatment_type])[0]
    }

# Predict
if st.button("Predict Survival"):
    input_data = pd.DataFrame([encode_input()])
    prediction = model.predict(input_data)[0]
    outcome = " Patient Survived" if prediction == 1 else " Patient Did Not Survive"
    st.subheader(f"Prediction Result: {outcome}")

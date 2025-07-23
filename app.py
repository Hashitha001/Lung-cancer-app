
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and encoders
model = joblib.load("lung_cancer_survival_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Lung Cancer Survival Predictor")
st.markdown("Enter the patient's details to predict survival outcome.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
country = st.selectbox("Country", label_encoders['country'].classes_)
cancer_stage = st.selectbox("Cancer Stage", label_encoders['cancer_stage'].classes_)
family_history = st.selectbox("Family History of Cancer", label_encoders['family_history'].classes_)
smoking_status = st.selectbox("Smoking Status", label_encoders['smoking_status'].classes_)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=180)
hypertension = st.selectbox("Hypertension", label_encoders['hypertension'].classes_)
asthma = st.selectbox("Asthma", label_encoders['asthma'].classes_)
cirrhosis = st.selectbox("Cirrhosis", label_encoders['cirrhosis'].classes_)
other_cancer = st.selectbox("Other Cancers", label_encoders['other_cancer'].classes_)
treatment_type = st.selectbox("Treatment Type", label_encoders['treatment_type'].classes_)

# Encoding input using label encoders
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
        "hypertension": label_encoders['hypertension'].transform([hypertension])[0],
        "asthma": label_encoders['asthma'].transform([asthma])[0],
        "cirrhosis": label_encoders['cirrhosis'].transform([cirrhosis])[0],
        "other_cancer": label_encoders['other_cancer'].transform([other_cancer])[0],
        "treatment_type": label_encoders['treatment_type'].transform([treatment_type])[0]
    }

# Predict
if st.button("Predict Survival"):
    input_df = pd.DataFrame([encode_input()])
    prediction = model.predict(input_df)[0]
    survival = "Survived" if prediction == 1 else "Did not Survive"
    st.subheader(f"Prediction: {survival}")

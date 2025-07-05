import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and training feature names
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.title("🏥 Patient Readmission Predictor")
st.markdown("Predict whether a patient is likely to be readmitted.")

# User inputs
age = st.selectbox("Age Range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"])
time_in_hospital = st.slider("Days in Hospital", 1, 14, 3)
n_lab_procedures = st.slider("# Lab Procedures", 1, 100, 50)
n_procedures = st.slider("# Procedures", 0, 6, 1)
n_medications = st.slider("# Medications", 1, 50, 10)
n_outpatient = st.slider("# Outpatient Visits", 0, 10, 0)
n_inpatient = st.slider("# Inpatient Visits", 0, 10, 0)
n_emergency = st.slider("# Emergency Visits", 0, 10, 0)
change = st.radio("Medication Changed", ["yes", "no"])
diabetes_med = st.radio("On Diabetes Medication?", ["yes", "no"])
glucose_test = st.selectbox("Glucose Test Result", ["no", "normal", "high"])
A1Ctest = st.selectbox("A1C Test Result", ["no", "normal", "high"])

# Encode inputs
le_age = {"[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8}
change_map = {"yes": 1, "no": 0}
diabetes_map = {"yes": 1, "no": 0}
glucose_map = {"no": 0, "normal": 1, "high": 2}
A1C_map = {"no": 0, "normal": 1, "high": 2}

# Base input dict
input_dict = {
    'age': le_age[age],
    'time_in_hospital': time_in_hospital,
    'n_lab_procedures': n_lab_procedures,
    'n_procedures': n_procedures,
    'n_medications': n_medications,
    'n_outpatient': n_outpatient,
    'n_inpatient': n_inpatient,
    'n_emergency': n_emergency,
    'change': change_map[change],
    'diabetes_med': diabetes_map[diabetes_med],
    'glucose_test': glucose_map[glucose_test],
    'A1Ctest': A1C_map[A1Ctest]
}

# Create full feature frame with correct structure
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Output
if st.button("Predict"):
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"❌ Patient is likely to be readmitted. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Patient is not likely to be readmitted. (Probability: {1 - probability:.2f})")

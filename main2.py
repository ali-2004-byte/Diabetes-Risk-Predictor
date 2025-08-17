import streamlit as st
import uvicorn
import threading
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# ---------------- FastAPI Part ----------------
api = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class PatientData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    blood_glucose_level: int
    gender_score: int
    smoke_score: int

@api.post("/predict")
def predict_diabetes(data: PatientData):
    features = np.array([[ 
        data.age, data.hypertension, data.heart_disease,
        data.bmi, data.blood_glucose_level, data.gender_score,
        data.smoke_score
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return {"prediction": "Safe" if prediction[0] == 0 else "At risk"}

# Run FastAPI in a background thread
def run_api():
    uvicorn.run(api, host="127.0.0.1", port=8000)

threading.Thread(target=run_api, daemon=True).start()


# ---------------- Streamlit Part ----------------
# --- App title ---
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("Diabetes Predictor")
    st.markdown("Welcome to the DIabetes Predictor! Upload your vitals and get the predicted result.")
    st.markdown("---")
    st.markdown("Built with FastAPI + Streamlit")

st.markdown("<h1 style='text-align: center; color: #E694FF;'> Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'> Enter patient information to predict diabetes risk.</p>", unsafe_allow_html=True)
st.markdown("---")


# --- Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, step=1.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=600, step=1)

    with col2:
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        gender_score = st.selectbox("Gender", [1, 2, 3], format_func=lambda x: ["Male", "Female", "Other"][x - 1])
        smoke_score = st.selectbox("Smoking Status", [1, 2, 3, 4, 5, 6], format_func=lambda x: [
            "Current smoker", "Former smoker", "No info", "Never smoked", "Ever", "Not current"
        ][x - 1])

    submitted = st.form_submit_button("🧪 Predict")


# --- Predict Button ---
if submitted:
    payload = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "bmi": bmi,
        "blood_glucose_level": blood_glucose_level,
        "gender_score": gender_score,
        "smoke_score": smoke_score
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        st.success(f"✅ Prediction: {result['prediction']}")
    except Exception as e:

        st.error(f"❌ Error contacting the prediction API: {e}")

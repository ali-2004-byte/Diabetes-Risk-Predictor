from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

class PatientData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    blood_glucose_level: int
    gender_score: int
    smoke_score: int
@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running. Visit /docs to try it out."}
@app.post('/predict')
def predict_diabetes(data: PatientData):
    features = np.array([[
        data.age,
        data.hypertension,
        data.heart_disease,
        data.bmi,
        data.blood_glucose_level,
        data.gender_score,
        data.smoke_score
    ]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    result = 'Safe' if prediction[0] == 0 else 'At risk'
    return {'prediction': result}

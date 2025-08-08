import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('diabetes_prediction_dataset.csv (1).csv')

diabetes_df = df.drop(['gender', 'smoking_history', 'HbA1c_level'], axis=1)


X = diabetes_df[['age', 'hypertension', 'heart_disease', 'bmi', 'blood_glucose_level', 'gender_score', 'smoke_score']]
y = diabetes_df['diabetes']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train_scaled, y_train)
print(f'Accuracy of the model is {knn.score(X_test_scaled, y_test)}')

joblib.dump(knn, 'diabetesapi\model.pkl')
joblib.dump(scaler, 'diabetesapi\scaler.pkl')
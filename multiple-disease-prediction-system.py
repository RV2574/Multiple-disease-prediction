# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 23:06:49 2025

@author: Rhythm
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the models
try:
    diabetes_model = pickle.load(open('C:/Users/Rhythm/Desktop/nsp ml/mdps/saved models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open('C:/Users/Rhythm/Desktop/nsp ml/mdps/saved models/heart_model.sav', 'rb'))
    parkinsons_model = pickle.load(open('C:/Users/Rhythm/Desktop/nsp ml/mdps/saved models/parkinsons_model.sav', 'rb'))
except Exception as e:
    st.error("Error loading models: " + str(e))
    st.stop()

# Sidebar menu
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# --- Diabetes Prediction Page ---
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0)
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0)
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0)
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0)
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0)
    with col2:
        Age = st.number_input('Age of the Person', min_value=0)

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    st.success(diab_diagnosis)

# --- Heart Disease Prediction Page ---
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)

    features = [
        ("Age", float), ("Sex", float), ("Chest Pain types", float),
        ("Resting Blood Pressure", float), ("Serum Cholesterol (mg/dl)", float),
        ("Fasting Blood Sugar > 120 mg/dl", float), ("Resting ECG", float),
        ("Max Heart Rate Achieved", float), ("Exercise Induced Angina", float),
        ("ST Depression", float), ("Slope of ST", float),
        ("Vessels Colored by Fluoroscopy", float), ("Thal", float)
    ]
    values = []

    for i, (label, dtype) in enumerate(features):
        with [col1, col2, col3][i % 3]:
            val = st.number_input(label, format="%.2f" if dtype == float else "%d")
            values.append(val)

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            heart_prediction = heart_disease_model.predict([values])
            heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    st.success(heart_diagnosis)

# --- Parkinson's Prediction Page ---
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    parkinsons_features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
        'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    inputs = []
    cols = st.columns(5)
    for i, label in enumerate(parkinsons_features):
        with cols[i % 5]:
            val = st.number_input(label, format="%.6f")
            inputs.append(val)

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        try:
            parkinsons_prediction = parkinsons_model.predict([inputs])
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    st.success(parkinsons_diagnosis)

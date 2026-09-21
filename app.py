"""
Streamlit Web App: Titanic Passenger Survival Predictor
Task: Streamlit Interactive ML Web App
Due Date: 17 October 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_trained_model():
    return joblib.load("titanic_model.joblib")

try:
    model = load_trained_model()
except Exception as e:
    st.error("Model file 'titanic_model.joblib' not found! Please run 'train_model.py' first.")
    st.stop()

st.title("🚢 Titanic Passenger Survival Predictor")
st.markdown("Interactive Machine Learning Web Application built with **Streamlit** & **Scikit-Learn**.")
st.write("---")

st.sidebar.header("Input Passenger Attributes")
pclass = st.sidebar.selectbox("Passenger Ticket Class (Pclass)", [1, 2, 3], index=2)
sex_label = st.sidebar.selectbox("Gender", ["Male", "Female"])
sex = 1 if sex_label == "Female" else 0
age = st.sidebar.slider("Passenger Age (Years)", min_value=1, max_value=80, value=25)
sibsp = st.sidebar.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
parch = st.sidebar.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=6, value=0)
fare = st.sidebar.slider("Ticket Fare ($)", min_value=0.0, max_value=500.0, value=32.0, step=0.5)

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare
}])

st.subheader("Selected Passenger Attributes")
st.dataframe(input_df)

if st.button("Predict Survival Outcome", type="primary"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = probabilities[prediction] * 100
    
    if prediction == 1:
        st.success(f"🎉 Prediction: **SURVIVED** (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"⚠️ Prediction: **DID NOT SURVIVE** (Confidence: {confidence:.2f}%)")
    
    st.progress(float(probabilities[1]))
    st.caption(f"Survival Probability: {probabilities[1]*100:.2f}% | Non-Survival Probability: {probabilities[0]*100:.2f}%")

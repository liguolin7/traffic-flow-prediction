import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

# Load model and scaler
model = load_model('models/traffic_model.h5')
scaler = joblib.load('models/scaler.pkl')

st.title("Traffic Flow Prediction System")

# User input
traffic_volume = st.number_input("Traffic Volume:", min_value=0)
temperature = st.number_input("Temperature (°C):", min_value=-30.0, max_value=50.0)
precipitation = st.number_input("Precipitation (mm):", min_value=0.0)

if st.button("Predict"):
    # Prepare input data
    new_data = np.array([[traffic_volume, temperature, precipitation]])
    new_data_scaled = scaler.transform(new_data).reshape(1, new_data.shape[1], new_data.shape[1])
    
    # Make prediction
    prediction = model.predict(new_data_scaled)
    st.write(f"Predicted Traffic Volume: {prediction[0][0]:.2f}")

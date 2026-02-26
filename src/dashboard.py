import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Hospital Monitoring Dashboard", layout="wide")

st.title("Smart Hospital Patient Monitoring System")

# Safe Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "patient_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "logistic_model.pkl")

# Load Data
try:
    data = pd.read_csv(DATA_PATH)
    st.success("Patient data loaded successfully.")
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Load Model
try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Latest Vitals
latest = data.iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Heart Rate (BPM)", round(latest["heart_rate"], 2))
col2.metric("SpO₂ (%)", round(latest["spo2"], 2))
col3.metric("Temperature (°C)", round(latest["temperature"], 2))

# Prediction
features = latest[["heart_rate", "spo2", "temperature"]].values.reshape(1, -1)
prediction = model.predict(features)[0]

if prediction == 1:
    st.error("⚠️ Patient Status: CRITICAL")
else:
    st.success("Patient Status: NORMAL")

# Trend Chart
st.subheader("Heart Rate Trend")
st.line_chart(data["heart_rate"])

# Moving Average
st.subheader("Heart Rate Moving Average")
data["HR_MA"] = data["heart_rate"].rolling(window=5).mean()
st.line_chart(data[["heart_rate", "HR_MA"]])
import streamlit as st
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load('crop_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Streamlit app UI
st.title("🌾 Crop Recommendation System")
st.markdown("Enter the values below to get a crop recommendation:")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", step=0.1)
humidity = st.number_input("Humidity (%)", step=0.1)
ph = st.number_input("pH level", step=0.1)
rainfall = st.number_input("Rainfall (mm)", step=0.1)

# Predict button
if st.button("Get Crop Recommendation"):
    new_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    new_data_scaled = scaler.transform(new_data)
    predicted_label_encoded = model.predict(new_data_scaled)
    predicted_crop = le.inverse_transform(predicted_label_encoded)

    st.success(f"✅ Recommended Crop: **{predicted_crop[0]}**")

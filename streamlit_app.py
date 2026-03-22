import streamlit as st
import pickle
import pandas as pd

# Load model and encoder
model = pickle.load(open('insurance_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

st.title("Insurance Cost Prediction")

# Inputs
age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.number_input("Children", 0, 5)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    input_processed = encoder.transform(input_data)
    prediction = model.predict(input_processed)

    st.success(f"Predicted Insurance Cost: {max(prediction[0], 0):.2f}")
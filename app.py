import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Load model and encoders
model = pickle.load(open("car_model.pkl", "rb"))
le_fuel = pickle.load(open("le_fuel.pkl", "rb"))
le_seller = pickle.load(open("le_seller.pkl", "rb"))
le_trans = pickle.load(open("le_trans.pkl", "rb"))

st.title("🚗 Car Price Predictor")
st.write("Enter the car details below to predict its selling price (in Lakhs ₹).")

present_price = st.number_input("Present Price (in Lakhs ₹)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
driven_kms = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000, step=1000)
fuel_type = st.selectbox("Fuel Type", le_fuel.classes_)
selling_type = st.selectbox("Seller Type", le_seller.classes_)
transmission = st.selectbox("Transmission", le_trans.classes_)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
year = st.number_input("Year of Purchase", min_value=1990, max_value=2024, value=2015, step=1)

if st.button("Predict Price"):
    car_age = 2024 - year

    input_data = pd.DataFrame([[
        present_price,
        driven_kms,
        le_fuel.transform([fuel_type])[0],
        le_seller.transform([selling_type])[0],
        le_trans.transform([transmission])[0],
        owner,
        car_age
    ]], columns=["Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner", "Car_Age"])

    prediction = model.predict(input_data)[0]
    prediction = max(0, round(prediction, 2))

    st.success(f"💰 Estimated Selling Price: ₹ {prediction} Lakhs")

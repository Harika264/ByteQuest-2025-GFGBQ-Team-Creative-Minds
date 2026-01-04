import streamlit as st
import joblib
import pandas as pd
model = joblib.load("models/RandomForest_model.pkl")
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title(" Heart Disease Risk Prediction System")
st.write("Enter patient health details to estimate disease risk.")
input_data = {}
for feature in model.feature_names_in_:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0
    )
if st.button("Predict Risk"):
    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[0][1]
    risk_percent = round(prob * 100, 2)
    if risk_percent < 30:
        risk_level = "Low Risk 🟢"
    elif risk_percent < 60:
        risk_level = "Moderate Risk 🟡"
    else:
        risk_level = "High Risk 🔴"
    st.subheader("Prediction Result")
    st.metric("Risk Percentage", f"{risk_percent}%")
    st.write("Risk Level:", risk_level)

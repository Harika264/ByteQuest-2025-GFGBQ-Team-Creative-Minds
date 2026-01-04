import joblib
import numpy as np
import pandas as pd
model = joblib.load("models/RandomForest_model.pkl")
def predict_risk(input_data: dict):
    """
    input_data: dictionary of patient features
    returns: risk percentage and risk level
    """
    df = pd.DataFrame([input_data])
    risk_prob = model.predict_proba(df)[0][1]
    risk_percent = round(risk_prob * 100, 2)
    # Risk categorization
    if risk_percent < 30:
        risk_level = "Low Risk"
    elif risk_percent < 60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"
    return {
        "risk_percentage": risk_percent,
        "risk_level": risk_level
    }
# Example test
if __name__ == "__main__":
    sample_input = {
        col: 0 for col in model.feature_names_in_
    }
    result = predict_risk(sample_input)
    print(result)

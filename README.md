# 🫀 Silent Disease Early Detection Engine

## Team Name
Creative Minds

## Team Members
- Harika Velugubantla
- Team Member 2
- Team Member 3

## Problem Statement
Early detection of silent diseases like heart disease is difficult due to
absence of clear symptoms in early stages. Delayed diagnosis leads to
severe health complications.

## Our Solution
We built an AI-powered early disease detection system that analyzes health
and lifestyle parameters to predict disease risk before symptoms appear.

Instead of a binary output, our system provides:
- Risk percentage (0–100%)
- Risk category (Low / Moderate / High)

This enables preventive healthcare rather than reactive treatment.

## Dataset
- Heart disease dataset (originally sourced from Kaggle)
- Includes health attributes like age, cholesterol, blood pressure, heart rate

## AI / ML Pipeline
1. Data Cleaning & Preprocessing
2. Feature Encoding & Scaling
3. Model Training (Logistic Regression & Random Forest)
4. Model Evaluation & Selection
5. Risk Score Prediction
6. Interactive User Interface

## Models Used
- Logistic Regression (baseline)
- Random Forest Classifier (final model)

Random Forest was selected due to better performance and robustness.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- GitHub

## Project Structure
data/
src/
models/
ui/
notebooks/


## Demo
A Streamlit-based UI allows users to input health parameters and receive
real-time disease risk predictions.

## Impact
- Enables early intervention
- Assists preventive healthcare
- Scalable to other silent diseases

## Future Scope
- Integration with wearable device data
- Multi-disease prediction
- Cloud deployment

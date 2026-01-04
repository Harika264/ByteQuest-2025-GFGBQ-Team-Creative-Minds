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
The project follows a structured end-to-end AI/ML pipeline:
1. **Data Collection**
   - Heart disease dataset (`data/heart.csv`)
2. **Data Preprocessing**
   - Missing value handling
   - Categorical encoding
   - Feature scaling
   - Implemented in `preprocessing.py`
3. **Feature Engineering**
   - Numerical normalization
   - Label encoding for categorical variables
4. **Model Training**
   - Random Forest Classifier
   - Implemented in `train_model.py`
5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Implemented in `evaluation.py`
6. **Model Serialization**
   - Trained model saved as `models/heart_model.pkl`
7. **Inference & Risk Prediction**
   - User input converted into model-compatible format
   - Implemented in `predict_risk.py`
8. **User Interface**
   - Interactive Streamlit UI
   - Implemented in `app.py`
**Note:** Initial experimentation was done using Jupyter notebooks.  
Final implementation is provided as modular Python scripts for reproducibility.

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

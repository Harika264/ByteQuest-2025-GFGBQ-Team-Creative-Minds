Dataset: UCI Heart Disease Dataset
Source:
ORIGINALLY SOURCED FROM KAGGLE
Description:
This dataset contains patient health and lifestyle attributes such as age, sex, cholesterol, blood pressure, and chest pain type.
Purpose:
Used for early disease risk prediction before visible symptoms appear.

Processed Dataset: processed_heart.csv
Preprocessing Steps:
- Missing values handled using median (numerical) and mode (categorical)
- Categorical variables encoded using One-Hot Encoding
- Numerical features standardized using StandardScaler
Target Variable:
HeartDisease (1 = disease present, 0 = no disease)
Purpose:
Used for training machine learning models for early heart disease prediction.

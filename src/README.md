Data Preprocessing Module
Steps Performed:
1. Loaded raw health and lifestyle dataset
2. Handled missing values using median/mode strategy
3. Encoded categorical variables using Label Encoding
4. Scaled numerical features using StandardScaler
5. Saved cleaned dataset for model training
Output File:
data/processed_heart.csv




Risk Prediction Module

The trained model outputs a probability score which is converted into a
human-interpretable risk percentage and categorized into Low, Moderate,
or High risk levels.

This enables early preventive decision-making instead of binary diagnosis.

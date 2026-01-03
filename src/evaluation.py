import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("data/processed_heart.csv")
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
model = joblib.load("models/RandomForest_model.pkl")
y_pred = model.predict(X)
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))

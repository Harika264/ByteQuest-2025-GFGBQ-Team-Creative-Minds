import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("data/heart.csv")  # change filename if needed

print("Initial shape:", df.shape)

# Handle missing values
for column in df.columns:
    if df[column].dtype == "object":
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)

# Encode categorical columns
label_encoders = {}
for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Separate features and target
target_column = df.columns[-1]  # assuming last column is target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create processed dataframe
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df[target_column] = y.values

# Save processed data
processed_df.to_csv("data/processed_heart.csv", index=False)
print("Preprocessing completed successfully!")
print("Processed shape:", processed_df.shape)

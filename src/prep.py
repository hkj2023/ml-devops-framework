import pandas as pd
import os

RAW_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\ml_ready_dataset.csv"
NEW_DATA_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\outputs\hasfailure_feature_importance.csv"

os.makedirs("outputs", exist_ok=True)

# Load raw dataset
df = pd.read_csv(RAW_PATH)

# Clean and encode
df_clean = pd.get_dummies(df.dropna())

# Create fresh unseen data for inference (sample rows)
df_new = df_clean.sample(min(10, len(df_clean)), random_state=42)

# Ensure schema matches training features
df_new = df_new.reindex(columns=df_clean.columns, fill_value=0)

# Save new unseen dataset
df_new.to_csv(NEW_DATA_PATH, index=False)

print(f"New unseen data saved to {NEW_DATA_PATH}")
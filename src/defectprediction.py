import pandas as pd
import joblib
import os
import json

# =====================================================
# LOAD MODEL
# =====================================================

model_path = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\hasfailure_model.pkl"
model = joblib.load(model_path)

print("✅ Model loaded successfully")

# =====================================================
# LOAD FEATURE LIST (IMPORTANT FOR CONSISTENCY)
# =====================================================

feature_path = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\feature_names.json"

with open(feature_path, "r") as f:
    feature_names = json.load(f)

# =====================================================
# LOAD NEW DATA
# =====================================================

data_path = r"C:\Users\OLLRP\Desktop\Test\new_data.csv"
df = pd.read_csv(data_path)

print("📊 Input Shape:", df.shape)

# =====================================================
# REMOVE TARGET / LEAKAGE COLUMNS
# =====================================================

df = df.drop(columns=[
    "HasFailure",
    "DefectLabel",
    "DefectCount"
], errors="ignore")

# =====================================================
# ALIGN FEATURES (CRITICAL FIX)
# =====================================================

df = df.reindex(columns=feature_names, fill_value=0)

# =====================================================
# PREDICTION
# =====================================================

predictions = model.predict(df)
probabilities = model.predict_proba(df)[:, 1]

# =====================================================
# CONVERT TO JSON STRUCTURE
# =====================================================

results = []

for i in range(len(predictions)):
    results.append({
        "record_id": i,
        "prediction": int(predictions[i]),
        "failure_probability": float(probabilities[i])
    })

# =====================================================
# SAVE AS JSON FILE
# =====================================================

output_path = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\outputs\Random_Forest_Prediction_Result.json"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("\n💾 JSON prediction saved successfully!")
print("📁 Location:", output_path)

# =====================================================
# SHOW SAMPLE OUTPUT
# =====================================================

print("\n================ SAMPLE RESULTS ================\n")
print(results[:3])
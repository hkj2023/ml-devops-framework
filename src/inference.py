import pandas as pd
import joblib
import json

# =====================================================
# PATHS
# =====================================================

DATA_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\ml_ready_dataset.csv"

MODEL_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\hasfailure_model.pkl"

FEATURE_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\feature_names.json"

OUTPUT_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\outputs\Random_Forest_Prediction_result.json"

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully")

# =====================================================
# LOAD FEATURE SCHEMA (FIXES YOUR ERROR)
# =====================================================

print("Loading feature schema...")

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

print(f"Expected features: {len(feature_names)}")

# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")

df = pd.read_csv(DATA_PATH)

# Target (only for reference, not used in prediction)
target = "HasFailure"

# =====================================================
# ALIGN FEATURES (CRITICAL FIX)
# =====================================================

print("Aligning features...")

# Keep ONLY training features
X = df.reindex(columns=feature_names, fill_value=0)

# =====================================================
# PREDICTION
# =====================================================

print("Running predictions...")

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# =====================================================
# BUILD OUTPUT JSON
# =====================================================

results = []

for i in range(len(df)):
    results.append({
        "index": int(i),
        "prediction": int(predictions[i]),
        "probability": float(probabilities[i])
    })

output = {
    "model": "RandomForestClassifier",
    "total_predictions": len(results),
    "results": results
}

# =====================================================
# SAVE JSON OUTPUT
# =====================================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("\n================ DONE ================\n")
print(f"Prediction results saved to: {OUTPUT_PATH}")
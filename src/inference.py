import pandas as pd
import joblib
import json
import os

# =====================================================
# PATHS (RELATIVE PATHS)
# =====================================================

DATA_PATH = "data/ml_ready_dataset.csv"
MODEL_PATH = "models/hasfailure_model.pkl"
FEATURE_PATH = "models/feature_names.json"
OUTPUT_PATH = "outputs/Random_Forest_Prediction_result.json"

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully")

# =====================================================
# LOAD FEATURE SCHEMA
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

# =====================================================
# ALIGN FEATURES
# =====================================================

print("Aligning features...")

# Keep ONLY training features, fill missing columns with 0
X = df.reindex(columns=feature_names, fill_value=0)

# =====================================================
# RUN PREDICTIONS
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
# SAVE OUTPUT
# =====================================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=4)

# =====================================================
# DONE
# =====================================================

print("\n================ DONE ================\n")
print(f"Prediction results saved to: {OUTPUT_PATH}")
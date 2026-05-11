import pandas as pd
import joblib
import json
import os

# =====================================================
# BASE DIRECTORY (DOCKER + LOCAL SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_names.json")
METRICS_PATH = os.path.join(BASE_DIR, "..", "outputs", "metrics.json")
DATA_PATH = os.path.join(
    BASE_DIR, "..", "data", "current_commit_metrics.csv"
)
OUTPUT_PATH = os.path.join(
    BASE_DIR, "..", "outputs", "risk_prediction.json"
)

MODEL_PATH = os.path.normpath(MODEL_PATH)
FEATURE_PATH = os.path.normpath(FEATURE_PATH)
METRICS_PATH = os.path.normpath(METRICS_PATH)
DATA_PATH = os.path.normpath(DATA_PATH)
OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# =====================================================
# DISPLAY MODEL EVALUATION
# =====================================================

print("\n====== MODEL EVALUATION ======\n")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

print(f"Accuracy  : {metrics['accuracy']}")
print(f"Precision : {metrics['precision']}")
print(f"Recall    : {metrics['recall']}")
print(f"F1-score  : {metrics['f1_score']}")
print(f"AUROC     : {metrics['auroc']}")

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# =====================================================
# LOAD FEATURE SCHEMA
# =====================================================

print("Loading feature schema...")

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

# =====================================================
# LOAD CURRENT COMMIT METRICS
# =====================================================

print("Loading current commit metrics...")

df = pd.read_csv(DATA_PATH)

# align columns
X = df.reindex(columns=feature_names, fill_value=0)

# =====================================================
# PREDICT DEFECT RISK
# =====================================================

print("Predicting defect risk...")

prob = model.predict_proba(X)[0][1]

# Risk classification
if prob >= 0.70:
    risk = "HIGH"
elif prob >= 0.40:
    risk = "MEDIUM"
else:
    risk = "LOW"

# =====================================================
# SAVE OUTPUT
# =====================================================

result = {
    "defect_probability": round(float(prob), 2),
    "risk_level": risk
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=4)

# =====================================================
# DISPLAY RESULT
# =====================================================

print("\n====== DEFECT RISK PREDICTION ======\n")
print(f"Defect Probability : {result['defect_probability']}")
print(f"Risk Level         : {result['risk_level']}")

print("\nSaved:", OUTPUT_PATH)
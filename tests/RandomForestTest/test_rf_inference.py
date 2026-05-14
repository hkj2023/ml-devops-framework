import pandas as pd
import joblib
import json
import os
import numpy as np

# =====================================================
# BASE DIRECTORY (PROJECT ROOT SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model.pkl")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "models", "rf_feature_names.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "rf_metrics.json")

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "current_commit_metrics.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "rf_inference.json")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# =====================================================
# LOAD MODEL METRICS
# =====================================================

print("\n====== MODEL EVALUATION ======\n")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

for k in ["accuracy", "precision", "recall", "f1_score", "auroc"]:
    print(f"{k:<10}: {metrics.get(k, 'N/A')}")

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading Random Forest model...")
model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD FEATURE SCHEMA
# =====================================================

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

# =====================================================
# LOAD DATA (NO TARGET REQUIRED FOR INFERENCE)
# =====================================================

print("Loading current commit metrics...")

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("❌ Input data is empty")

# IMPORTANT FIX: inference uses ONLY features
X = df.reindex(columns=feature_names, fill_value=0)

# =====================================================
# PREDICTION
# =====================================================

print("Running Random Forest inference...")

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)

    # probability of class 1 (failure/defect)
    risk_score = float(np.mean(probs[:, 1]))
else:
    preds = model.predict(X)
    risk_score = float(np.mean(preds))

# safety clamp
risk_score = max(0.0, min(1.0, risk_score))

# =====================================================
# RISK CLASSIFICATION
# =====================================================

if risk_score < 0.40:
    risk_level = "LOW"
    risk_value = 1
    action = "APPROVED"

elif risk_score < 0.70:
    risk_level = "MEDIUM"
    risk_value = 2
    action = "MONITOR"

else:
    risk_level = "HIGH"
    risk_value = 3
    action = "RETEST_REQUIRED + PIPELINE BLOCK"

# =====================================================
# OUTPUT
# =====================================================

result = {
    "model": "RandomForest",
    "risk_score": round(risk_score, 4),
    "risk_value": risk_value,
    "risk_level": risk_level,
    "action": action,
    "total_samples": int(len(X))
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=4)

# =====================================================
# PRINT RESULTS
# =====================================================

print("\n====== RANDOM FOREST INFERENCE ======\n")
print(f"Model      : {result['model']}")
print(f"Samples    : {result['total_samples']}")
print(f"Risk Score : {result['risk_score']}")
print(f"Risk Level : {result['risk_level']}")
print(f"Risk Value : {result['risk_value']}")
print(f"Action     : {result['action']}")

print("\nSaved:", OUTPUT_PATH)
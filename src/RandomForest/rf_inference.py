import pandas as pd
import joblib
import json
import os
import sys

# =====================================================
# ADD PROJECT ROOT TO PYTHON PATH (fixes "No module src")
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

# =====================================================
# SAFE PATH IMPORT (fallback-safe)
# =====================================================

try:
    from src.common.paths import MODEL_DIR, DATA_DIR, OUTPUT_DIR
except:
    # fallback if src.common.paths not ready
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# =====================================================
# FILE PATHS (FIXED - NO DUPLICATION)
# =====================================================

MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "rf_feature_names.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "rf_metrics.json")
DATA_PATH = os.path.join(DATA_DIR, "current_commit_metrics.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "risk_prediction.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD METRICS (SAFE)
# =====================================================

print("\n====== MODEL EVALUATION ======\n")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    print(f"Accuracy  : {metrics.get('accuracy', 0)}")
    print(f"Precision : {metrics.get('precision', 0)}")
    print(f"Recall    : {metrics.get('recall', 0)}")
    print(f"F1-score  : {metrics.get('f1_score', 0)}")
    print(f"AUROC     : {metrics.get('auroc', 0)}")
else:
    print("Metrics file not found (skipping evaluation display).")

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading trained model...")
model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD FEATURES
# =====================================================

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading current commit metrics...")
df = pd.read_csv(DATA_PATH)

X = df.reindex(columns=feature_names, fill_value=0)

# =====================================================
# PREDICTION
# =====================================================

print("Predicting defect risk...")

prob = model.predict_proba(X)[0][1]

# =====================================================
# CLASSIFICATION (LOW / MEDIUM / HIGH)
# =====================================================

if prob < 0.40:
    risk_level = "LOW"
    risk_value = 1
    action = "APPROVED"

elif prob < 0.70:
    risk_level = "MEDIUM"
    risk_value = 2
    action = "MONITOR"

else:
    risk_level = "HIGH"
    risk_value = 3
    action = "RETEST_REQUIRED"

# =====================================================
# OUTPUT
# =====================================================

result = {
    "risk_score": round(float(prob), 4),
    "risk_value": risk_value,
    "risk_level": risk_level,
    "action": action
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=4)

# =====================================================
# DISPLAY
# =====================================================

print("\n====== DEFECT RISK PREDICTION ======\n")
print(f"Risk Score : {result['risk_score']}")
print(f"Risk Level : {result['risk_level']}")
print(f"Risk Value : {result['risk_value']}")
print(f"Action     : {result['action']}")

print("\nSaved:", OUTPUT_PATH)
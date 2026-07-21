import pandas as pd
import joblib
import json
import os
import numpy as np
import sys

# =====================================================
# PROJECT ROOT FIX (IMPORTANT FOR PYTEST)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.common.paths import MODEL_DIR, DATA_DIR, OUTPUT_DIR

# =====================================================
# PATHS
# =====================================================

MODEL_PATH = os.path.join(MODEL_DIR, "if_model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "if_feature_names.json")
DATA_PATH = os.path.join(DATA_DIR, "current_commit_metrics.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "if_inference.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL + FEATURES
# =====================================================

print("\nLoading Isolation Forest model...")
model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

X = df.reindex(columns=feature_names, fill_value=0)
X = X.fillna(0)

# =====================================================
# ANOMALY SCORING
# =====================================================

scores = model.decision_function(X)

# NORMALIZED anomaly score (CORRECT)
risk_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

# =====================================================
# CLASSIFICATION
# =====================================================

def classify(score):
    if score >= 0.70:
        return "HIGH_ANOMALY"
    elif score >= 0.40:
        return "MEDIUM_ANOMALY"
    else:
        return "LOW_ANOMALY"

results = []

for i, r in enumerate(risk_scores):
    level = classify(r)

    action = (
        "BLOCK_DEPLOYMENT" if level == "HIGH_ANOMALY"
        else "REVIEW_REQUIRED" if level == "MEDIUM_ANOMALY"
        else "ALLOW"
    )

    results.append({
        "sample_id": i,
        "anomaly_score": round(float(r), 4),
        "anomaly_level": level,
        "action": action
    })

# =====================================================
# SUMMARY
# =====================================================

summary = {
    "model": "IsolationForest",
    "total_samples": len(results),
    "high_anomalies": sum(r["anomaly_level"] == "HIGH_ANOMALY" for r in results),
    "medium_anomalies": sum(r["anomaly_level"] == "MEDIUM_ANOMALY" for r in results),
    "low_anomalies": sum(r["anomaly_level"] == "LOW_ANOMALY" for r in results),
    "results": results
}

# =====================================================
# SAVE OUTPUT
# =====================================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=4)

# =====================================================
# PRINT OUTPUT
# =====================================================

print("\n====== ISOLATION FOREST ANOMALY INFERENCE ======\n")
print(f"Total Samples   : {summary['total_samples']}")
print(f"High Anomalies  : {summary['high_anomalies']}")
print(f"Medium Anomalies: {summary['medium_anomalies']}")
print(f"Low Anomalies   : {summary['low_anomalies']}")

print("\nSample Output:")
for r in results[:5]:
    print(r)

print("\nSaved:", OUTPUT_PATH)
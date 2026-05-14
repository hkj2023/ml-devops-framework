import pandas as pd
import joblib
import json
import os
import numpy as np

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "if_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "..", "models", "if_feature_names.json")
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "current_commit_metrics.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "..", "outputs", "if_inference_test.json")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

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

if df.empty:
    raise ValueError("Input dataset is empty")

# Align features safely
X = df.reindex(columns=feature_names, fill_value=0)

# =====================================================
# ANOMALY SCORING
# =====================================================

scores = model.decision_function(X).astype(float)

# Safe normalization (avoid divide-by-zero)
min_s = np.min(scores)
max_s = np.max(scores)

if max_s - min_s == 0:
    risk_scores = np.zeros_like(scores)
else:
    risk_scores = 1 - (scores - min_s) / (max_s - min_s + 1e-9)

# =====================================================
# CLASSIFICATION RULES
# =====================================================

def classify(r):
    if r >= 0.70:
        return "HIGH_ANOMALY", 3, "BLOCK_DEPLOYMENT"
    elif r >= 0.40:
        return "MEDIUM_ANOMALY", 2, "REVIEW_REQUIRED"
    else:
        return "LOW_ANOMALY", 1, "ALLOW"

# =====================================================
# PREDICTION LOOP
# =====================================================

results = []

for i, score in enumerate(risk_scores):
    level, value, action = classify(float(score))

    results.append({
        "sample_id": int(i),
        "anomaly_score": round(float(score), 4),
        "anomaly_level": level,
        "anomaly_value": value,
        "action": action
    })

# =====================================================
# SUMMARY
# =====================================================

summary = {
    "model": "IsolationForest",
    "total_samples": int(len(results)),
    "high_anomalies": sum(r["anomaly_level"] == "HIGH_ANOMALY" for r in results),
    "medium_anomalies": sum(r["anomaly_level"] == "MEDIUM_ANOMALY" for r in results),
    "low_anomalies": sum(r["anomaly_level"] == "LOW_ANOMALY" for r in results),
    "execution_status": "SUCCESS",
    "results": results
}

# =====================================================
# SAVE OUTPUT
# =====================================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=4)

# =====================================================
# PRINT RESULTS
# =====================================================

print("\n====== ISOLATION FOREST INFERENCE TEST ======\n")

print(f"Total Samples   : {summary['total_samples']}")
print(f"High Anomalies  : {summary['high_anomalies']}")
print(f"Medium Anomalies: {summary['medium_anomalies']}")
print(f"Low Anomalies   : {summary['low_anomalies']}")

print("\nSample Outputs:")
for r in results[:5]:
    print(r)

print("\nExecution Status:", summary["execution_status"])
print("\nSaved:", OUTPUT_PATH)
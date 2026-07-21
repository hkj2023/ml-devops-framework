import pandas as pd
import joblib
import json
import os
import numpy as np

from sklearn.metrics import silhouette_score

# ============================================
# BASE DIRECTORY
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "models", "if_model.pkl")
)

SCALER_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "models", "if_scaler.pkl")
)

FEATURE_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "models", "if_feature_names.json")
)

DATA_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "data", "processed", "test.csv")
)

OUTPUT_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "outputs", "if_test_metrics.json")
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

TARGET = "HasFailure"

# ============================================
# LOAD DATA
# ============================================

print("\n====== LOADING TEST DATA ======\n")
df = pd.read_csv(DATA_PATH)

# ============================================
# LOAD MODEL + SCALER
# ============================================

print("Loading Isolation Forest model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ============================================
# LOAD FEATURES
# ============================================

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

# ============================================
# FEATURE ALIGNMENT
# ============================================

X = df.reindex(columns=feature_names, fill_value=0)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset")

y = df[TARGET].values

# ============================================
# SCALING
# ============================================

X_scaled = scaler.transform(X)

# ============================================
# PREDICTION
# ============================================

raw_pred = model.predict(X_scaled)

# Convert IsolationForest output:
# -1 = anomaly (failure)
#  1 = normal
y_pred = (raw_pred == -1).astype(int)

# ============================================
# METRICS
# ============================================

anomalies_detected = int(np.sum(y_pred))

anomaly_rate = anomalies_detected / len(y) if len(y) > 0 else 0.0

# silhouette score (only valid if both classes exist)
score = 0.0
try:
    if len(np.unique(y_pred)) > 1:
        score = silhouette_score(X_scaled, y_pred)
except Exception as e:
    print("Silhouette score failed:", str(e))
    score = 0.0

metrics = {
    "model": "IsolationForest",
    "total_samples": int(len(y)),
    "anomalies_detected": anomalies_detected,
    "anomaly_rate": round(anomaly_rate, 4),
    "silhouette_score": round(float(score), 4),
    "execution_status": "SUCCESS"
}

# ============================================
# SAVE RESULTS
# ============================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# ============================================
# PRINT RESULTS
# ============================================

print("\n====== ISOLATION FOREST TEST RESULTS ======\n")

for k, v in metrics.items():
    print(f"{k:<20}: {v}")

print("\nSaved:", OUTPUT_PATH)
import pandas as pd
import os
import json

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================
# PATHS
# ============================================

DATA_PATH = "data/processed/train.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ANOMALY_OUTPUT = os.path.join(OUTPUT_DIR, "anomaly_results.csv")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "anomaly_summary.json")

TARGET = "HasFailure"

# ============================================
# LOAD DATA
# ============================================

print("\nLoading training data...")

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)

# Remove target column
if TARGET in df.columns:
    X = df.drop(columns=[TARGET])
else:
    X = df.copy()

# ============================================
# SCALE FEATURES
# ============================================

print("\nScaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# RUN ISOLATION FOREST
# ============================================

print("\nRunning anomaly detection...")

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

labels = model.fit_predict(X_scaled)

# Convert labels
df["AnomalyFlag"] = labels
df["AnomalyFlag"] = df["AnomalyFlag"].map({
    -1: "ANOMALY",
     1: "NORMAL"
})

# ============================================
# SAVE RESULTS
# ============================================

df.to_csv(ANOMALY_OUTPUT, index=False)

anomaly_count = (df["AnomalyFlag"] == "ANOMALY").sum()

summary = {
    "model": "IsolationForest",
    "total_samples": len(df),
    "anomalies_detected": int(anomaly_count),
    "anomaly_rate": float(anomaly_count / len(df))
}

with open(SUMMARY_OUTPUT, "w") as f:
    json.dump(summary, f, indent=4)

# ============================================
# DONE
# ============================================

print("\n================ DONE ================\n")
print("Anomalies detected:", anomaly_count)
print("Saved:")
print("-", ANOMALY_OUTPUT)
print("-", SUMMARY_OUTPUT)
import pandas as pd
import joblib
import json
import os
import mlflow
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "if_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "..", "models", "if_scaler.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "..", "outputs", "if_metrics.json")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

TARGET = "HasFailure"

# ============================================
# LOAD DATA + MODEL
# ============================================
print("\nLoading test data...")
df = pd.read_csv(DATA_PATH)

print("Loading Isolation Forest model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X = df.drop(columns=[TARGET], errors="ignore")
y = df[TARGET]
X_scaled = scaler.transform(X)

# ============================================
# PREDICTIONS
# ============================================
scores = model.decision_function(X_scaled)
y_pred = (model.predict(X_scaled) == -1).astype(int)

# IMPORTANT: use scores directly (not -scores)
auc = roc_auc_score(y, scores)

# ============================================
# METRICS
# ============================================
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
total_samples = len(y)
anomalies_detected = int(y_pred.sum())
anomaly_rate = anomalies_detected / total_samples

precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

# Extra metrics
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
detection_rate = recall  # same as recall

metrics = {
    "model": "IsolationForest",
    "total_samples": total_samples,
    "anomalies_detected": anomalies_detected,
    "anomaly_rate": round(anomaly_rate, 4),

    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "auc_roc": round(auc, 4),

    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),

    "fpr": round(fpr, 4),
    "fnr": round(fnr, 4),
    "detection_rate": round(detection_rate, 4),

    "execution_status": "SUCCESS"
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# ============================================
# MLFLOW LOGGING
# ============================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("if-model")

with mlflow.start_run():
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

# ============================================
# PRINT
# ============================================
print("\n====== ISOLATION FOREST RESULTS ======\n")
print(f"Model              : {metrics['model']}")
print(f"Total Samples      : {metrics['total_samples']}")
print(f"Anomalies Detected : {metrics['anomalies_detected']}")
print(f"Anomaly Rate       : {metrics['anomaly_rate']}")

print(f"\nPrecision      : {metrics['precision']}")
print(f"Recall         : {metrics['recall']}")
print(f"F1-score       : {metrics['f1_score']}")
print(f"AUC-ROC        : {metrics['auc_roc']}")
print(f"FPR            : {metrics['fpr']}")
print(f"FNR            : {metrics['fnr']}")
print(f"Detection Rate : {metrics['detection_rate']}")

print("\nConfusion Matrix Metrics")
print(f"TN (True Negatives) : {metrics['true_negatives']}")
print(f"FP (False Positives): {metrics['false_positives']}")
print(f"FN (False Negatives): {metrics['false_negatives']}")
print(f"TP (True Positives) : {metrics['true_positives']}")

print("\nExecution Status:", metrics["execution_status"])
print("\nSaved:", OUTPUT_PATH)

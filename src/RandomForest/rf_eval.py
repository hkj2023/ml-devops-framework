import pandas as pd
import joblib
import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================
# BASE DIRECTORY
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "rf_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "..", "models", "rf_feature_names.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "..", "outputs", "rf_metrics.json")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

TARGET = "HasFailure"

# ============================================
# LOAD DATA
# ============================================

print("\n====== LOADING TEST DATA ======\n")
df = pd.read_csv(DATA_PATH)

# ============================================
# LOAD MODEL
# ============================================

print("Loading Random Forest model...")
model = joblib.load(MODEL_PATH)

# ============================================
# LOAD FEATURE SCHEMA
# ============================================

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

# ============================================
# SAFE FEATURE ALIGNMENT
# ============================================

X = df.reindex(columns=feature_names, fill_value=0)
y = df[TARGET]

X = X.fillna(0)

# ============================================
# PREDICTIONS
# ============================================

y_prob = model.predict_proba(X)[:, 1]

threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# ============================================
# CONFUSION MATRIX (USED INTERNALLY ONLY)
# ============================================

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

predicted_failure_rate = y_pred.mean()

# ============================================
# METRICS
# ============================================

metrics = {
    "model": "RandomForestClassifier",
    "total_samples": len(y),

    "predicted_failure_rate": round(predicted_failure_rate, 4),

    "accuracy": round(accuracy_score(y, y_pred), 4),
    "precision": round(precision_score(y, y_pred, zero_division=0), 4),
    "recall": round(recall_score(y, y_pred, zero_division=0), 4),
    "f1_score": round(f1_score(y, y_pred, zero_division=0), 4),

    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),

    "auroc": round(roc_auc_score(y, y_prob), 4),
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

print("\n====== RANDOM FOREST RESULTS ======\n")

print(f"Model                 : {metrics['model']}")
print(f"Total Samples         : {metrics['total_samples']}")
print(f"Predicted Failure Rate: {metrics['predicted_failure_rate']}")

print(f"\nAccuracy  : {metrics['accuracy']}")
print(f"Precision : {metrics['precision']}")
print(f"Recall    : {metrics['recall']}")
print(f"F1-score  : {metrics['f1_score']}")
print(f"AUROC     : {metrics['auroc']}")

print("\nExecution Status:", metrics["execution_status"])
print("\nSaved:", OUTPUT_PATH)
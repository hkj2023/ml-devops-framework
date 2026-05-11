import pandas as pd
import joblib
import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =====================================================
# BASE DIRECTORY (DOCKER + LOCAL SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_names.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "outputs", "metrics.json")

DATA_PATH = os.path.normpath(DATA_PATH)
MODEL_PATH = os.path.normpath(MODEL_PATH)
FEATURE_PATH = os.path.normpath(FEATURE_PATH)
OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

TARGET = "HasFailure"

# =====================================================
# LOAD TEST DATA
# =====================================================

print("\n====== LOADING TEST DATA ======\n")

df = pd.read_csv(DATA_PATH)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# =====================================================
# LOAD FEATURE SCHEMA
# =====================================================

print("Loading feature schema...")

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

print(f"Expected features: {len(feature_names)}")

# =====================================================
# PREPARE FEATURES
# =====================================================

X = df[feature_names]
y = df[TARGET]

# =====================================================
# RUN PREDICTIONS
# =====================================================

print("Running evaluation...")

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# =====================================================
# COMPUTE METRICS
# =====================================================

metrics = {
    "accuracy": round(accuracy_score(y, y_pred), 2),
    "precision": round(precision_score(y, y_pred), 2),
    "recall": round(recall_score(y, y_pred), 2),
    "f1_score": round(f1_score(y, y_pred), 2),
    "auroc": round(roc_auc_score(y, y_prob), 2)
}

# =====================================================
# SAVE METRICS
# =====================================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n====== MODEL EVALUATION RESULTS ======\n")
print(f"Accuracy  : {metrics['accuracy']}")
print(f"Precision : {metrics['precision']}")
print(f"Recall    : {metrics['recall']}")
print(f"F1-score  : {metrics['f1_score']}")
print(f"AUROC     : {metrics['auroc']}")

print("\nSaved:", OUTPUT_PATH)
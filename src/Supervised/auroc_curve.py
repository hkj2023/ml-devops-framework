import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# ============================================
# BASE DIRECTORY
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "rf_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "..", "models", "rf_feature_names.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "rf_metrics.json")
ROC_PLOT_PATH = os.path.join(OUTPUT_DIR, "rf_roc_curve.png")

TARGET = "HasFailure"

# ============================================
# LOAD TEST DATA
# ============================================

print("\n====== LOADING TEST DATA ======\n")

df = pd.read_csv(DATA_PATH)

# Fill missing values
df = df.fillna(0)

# ============================================
# LOAD MODEL
# ============================================

print("Loading Random Forest model...")
model = joblib.load(MODEL_PATH)

# ============================================
# LOAD FEATURE SCHEMA
# ============================================

print("Loading feature schema...")

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

# ============================================
# PREPARE TEST FEATURES
# ============================================

# Separate target
y = df[TARGET]

# Remove target column
X = df.drop(columns=[TARGET], errors="ignore")

# One-hot encode test data
X = pd.get_dummies(X)

# Align columns with training features
X = X.reindex(columns=feature_names, fill_value=0)

print("Feature shape:", X.shape)

# ============================================
# CONTINUOUS PREDICTIONS
# ============================================

print("\nGenerating predictions...")

# Continuous probabilities
y_prob = model.predict_proba(X)[:, 1]

# Binary predictions using threshold = 0.5
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# ============================================
# CONFUSION MATRIX
# ============================================

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

# ============================================
# PERFORMANCE METRICS
# ============================================

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
auroc = roc_auc_score(y, y_prob)

predicted_failure_rate = y_pred.mean()

# ============================================
# ROC CURVE
# ============================================

fpr, tpr, thresholds = roc_curve(y, y_prob)

# ============================================
# SAVE METRICS
# ============================================

metrics = {
    "model": "RandomForestClassifier",
    "total_samples": int(len(y)),

    "predicted_failure_rate": round(float(predicted_failure_rate), 4),

    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "auroc": round(float(auroc), 4),

    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),

    "execution_status": "SUCCESS"
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# ============================================
# PLOT ROC CURVE
# ============================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    color="blue",
    linewidth=2.5,
    label=f"Random Forest (AUC = {auroc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2,
    color="gray",
    label="Random Guess"
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Receiver Operating Characteristic (ROC)", fontsize=14)

plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()

plt.savefig(ROC_PLOT_PATH, dpi=300)
plt.show()

# ============================================
# PRINT RESULTS
# ============================================

print("\n========== RANDOM FOREST RESULTS ==========\n")

print(f"Model                  : {metrics['model']}")
print(f"Total Samples          : {metrics['total_samples']}")
print(f"Predicted Failure Rate : {metrics['predicted_failure_rate']:.4f}")

print("\nClassification Metrics")
print("----------------------")
print(f"Accuracy               : {metrics['accuracy']:.4f}")
print(f"Precision              : {metrics['precision']:.4f}")
print(f"Recall                 : {metrics['recall']:.4f}")
print(f"F1-score               : {metrics['f1_score']:.4f}")
print(f"AUROC                  : {metrics['auroc']:.4f}")

print("\nConfusion Matrix")
print("----------------")
print(f"True Negatives         : {metrics['true_negatives']}")
print(f"False Positives        : {metrics['false_positives']}")
print(f"False Negatives        : {metrics['false_negatives']}")
print(f"True Positives         : {metrics['true_positives']}")

print("\nExecution Status:", metrics["execution_status"])

print("\nMetrics saved to:")
print(OUTPUT_PATH)

print("\nROC curve saved to:")
print(ROC_PLOT_PATH)
import pandas as pd
import joblib
import json

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# PATHS
# =====================================================

DATA_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\ml_ready_dataset.csv"

MODEL_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\hasfailure_model.pkl"

FEATURE_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\feature_names.json"

OUTPUT_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\evaluation_metrics.json"

# =====================================================
# LOAD DATA
# =====================================================

print("\n================ LOADING DATA ================\n")

df = pd.read_csv(DATA_PATH)

target = "HasFailure"

# =====================================================
# LOAD MODEL
# =====================================================

print("\n================ LOADING MODEL ================\n")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# =====================================================
# LOAD FEATURE NAMES (CRITICAL FIX FOR YOUR ERROR)
# =====================================================

print("\n================ LOADING FEATURE SCHEMA ================\n")

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

print(f"Expected features: {len(feature_names)}")

# =====================================================
# SPLIT FEATURES / TARGET
# =====================================================

X = df[feature_names]   # ensures SAME ORDER + SAME COLUMNS
y = df[target]

# =====================================================
# PREDICTION
# =====================================================

print("\n================ RUNNING PREDICTION ================\n")

y_pred = model.predict(X)

y_prob = model.predict_proba(X)[:, 1]

# =====================================================
# METRICS
# =====================================================

print("\n================ COMPUTING METRICS ================\n")

accuracy = accuracy_score(y, y_pred)
balanced_acc = balanced_accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)

conf_matrix = confusion_matrix(y, y_pred).tolist()

class_report = classification_report(y, y_pred, output_dict=True)

# =====================================================
# SAVE EVALUATION METRICS (JSON ARTIFACT)
# =====================================================

metrics = {
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "roc_auc": float(roc_auc),
    "confusion_matrix": conf_matrix,
    "classification_report": class_report
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("\n================ SAVED =================")
print(f"Evaluation metrics saved to: {OUTPUT_PATH}")

# =====================================================
# SUMMARY
# =====================================================

print("\n================ FINAL RESULTS ================\n")
print("Accuracy:", accuracy)
print("Balanced Accuracy:", balanced_acc)
print("ROC-AUC:", roc_auc)
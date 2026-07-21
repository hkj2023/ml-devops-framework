"""
Isolation Forest Evaluation
ML-Enabled DevOps Framework

Evaluation:
    Test data only

Metrics:
    Precision
    Recall
    F1-score
    AUROC
    FPR
    FNR
    Confusion Matrix
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================
# BASE DIRECTORY
# ============================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


TEST_PATH = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "processed",
        "test.csv"
    )
)


MODEL_DIR = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "models"
    )
)


OUTPUT_DIR = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "outputs"
    )
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TARGET = "HasFailure"



# ============================================
# LOAD TEST DATA
# ============================================

print("Loading test data...")


df = pd.read_csv(
    TEST_PATH
)


df = df.fillna(0)



X_test = df.drop(
    columns=[TARGET],
    errors="ignore"
)


y_test = df[TARGET]



# ============================================
# LOAD MODEL
# ============================================

print("Loading Isolation Forest model...")


model_path = os.path.join(
    MODEL_DIR,
    "if_model.pkl"
)


scaler_path = os.path.join(
    MODEL_DIR,
    "if_scaler.pkl"
)


feature_path = os.path.join(
    MODEL_DIR,
    "if_feature_names.json"
)



model = joblib.load(
    model_path
)


scaler = joblib.load(
    scaler_path
)


with open(
    feature_path,
    "r"
) as f:

    feature_names = json.load(f)



# ============================================
# FEATURE PREPARATION
# ============================================

X_test = pd.get_dummies(
    X_test
)


# Align features with training schema

X_test = X_test.reindex(
    columns=feature_names,
    fill_value=0
)



# ============================================
# SCALING
# ============================================

X_test_scaled = scaler.transform(
    X_test
)



# ============================================
# PREDICTION
# ============================================

prediction = model.predict(
    X_test_scaled
)


# Isolation Forest:
#
# -1 = anomaly
#  1 = normal


anomaly_prediction = np.where(
    prediction == -1,
    1,
    0
)



# ============================================
# DECISION SCORES
# ============================================

decision_scores = model.decision_function(
    X_test_scaled
)


# IMPORTANT:
# Higher score = more normal
# ROC requires higher score = anomaly
#
# Therefore invert score

anomaly_scores = -decision_scores



# ============================================
# METRICS
# ============================================

precision = precision_score(
    y_test,
    anomaly_prediction,
    zero_division=0
)


recall = recall_score(
    y_test,
    anomaly_prediction,
    zero_division=0
)


f1 = f1_score(
    y_test,
    anomaly_prediction,
    zero_division=0
)


auc = roc_auc_score(
    y_test,
    anomaly_scores
)



# ============================================
# CONFUSION MATRIX
# ============================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    anomaly_prediction
).ravel()



fpr = fp / (fp + tn) if (fp + tn) else 0

fnr = fn / (fn + tp) if (fn + tp) else 0



# ============================================
# DISPLAY RESULTS
# ============================================

print("\n====== ISOLATION FOREST RESULTS ======\n")


print(
    "Model              : IsolationForest"
)


print(
    "Total Samples      :",
    len(y_test)
)


print(
    "Anomalies Detected :",
    anomaly_prediction.sum()
)


print(
    "Anomaly Rate       :",
    round(
        anomaly_prediction.mean(),
        4
    )
)


print()


print(
    "Precision      :",
    round(precision,4)
)


print(
    "Recall         :",
    round(recall,4)
)


print(
    "F1-score       :",
    round(f1,4)
)


print(
    "AUC-ROC        :",
    round(auc,4)
)


print(
    "FPR            :",
    round(fpr,4)
)


print(
    "FNR            :",
    round(fnr,4)
)


print(
    "Detection Rate :",
    round(recall,4)
)



print("\nConfusion Matrix Metrics")


print(
    "TN (True Negatives) :",
    tn
)


print(
    "FP (False Positives):",
    fp
)


print(
    "FN (False Negatives):",
    fn
)


print(
    "TP (True Positives) :",
    tp
)



# ============================================
# SAVE RESULTS
# ============================================

results = {

    "Model": "IsolationForest",

    "Total Samples": int(len(y_test)),

    "Anomalies Detected": int(anomaly_prediction.sum()),

    "Precision": round(float(precision),4),

    "Recall": round(float(recall),4),

    "F1-score": round(float(f1),4),

    "AUC-ROC": round(float(auc),4),

    "FPR": round(float(fpr),4),

    "FNR": round(float(fnr),4),

    "Detection Rate": round(float(recall),4),

    "TN": int(tn),

    "FP": int(fp),

    "FN": int(fn),

    "TP": int(tp)

}



output_file = os.path.join(
    OUTPUT_DIR,
    "if_metrics.json"
)


with open(
    output_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nExecution Status: SUCCESS")


print(
    "\nSaved:",
    output_file
)
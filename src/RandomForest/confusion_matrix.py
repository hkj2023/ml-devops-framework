"""
Confusion Matrix Generation
ML-Enabled DevOps Framework

Random Forest Defect Prediction

Model:
    models/rf_model.pkl

Test Data:
    data/processed/test.csv

Outputs:
    outputs/confusion_matrix.png
    outputs/confusion_matrix.json
"""


import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



# =====================================================
# PROJECT ROOT
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../"
    )
)



# =====================================================
# PATHS
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rf_model.pkl"
)


TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test.csv"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading Random Forest model...")


model = joblib.load(
    MODEL_PATH
)


print("Model loaded successfully")



# =====================================================
# LOAD TEST DATA
# =====================================================

print("\nLoading test data...")


df = pd.read_csv(
    TEST_DATA_PATH
)


print(
    "Test data shape:",
    df.shape
)



# =====================================================
# SEPARATE FEATURES AND LABEL
# =====================================================

# Change this if your target column has another name

TARGET_COLUMN = "HasFailure"


if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )



y_test = df[TARGET_COLUMN]


X_test = df.drop(
    columns=[TARGET_COLUMN]
)



# =====================================================
# MATCH MODEL FEATURES
# =====================================================

if hasattr(model, "feature_names_in_"):

    model_features = list(
        model.feature_names_in_
    )


    X_test = X_test[
        model_features
    ]



print(
    "Final test features:",
    X_test.shape
)



# =====================================================
# PREDICTION
# =====================================================

print("\nGenerating predictions...")


y_pred = model.predict(
    X_test
)



# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


tn, fp, fn, tp = cm.ravel()



print("\nConfusion Matrix:")
print(cm)


print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)



# =====================================================
# SAVE CONFUSION MATRIX IMAGE
# =====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Defect",
        "Defect"
    ]
)


disp.plot()


plt.title(
    "Random Forest Confusion Matrix - Defect Prediction"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# CALCULATE METRICS
# =====================================================

metrics = {

    "True Negative": int(tn),

    "False Positive": int(fp),

    "False Negative": int(fn),

    "True Positive": int(tp),

    "Accuracy": accuracy_score(
        y_test,
        y_pred
    ),

    "Precision": precision_score(
        y_test,
        y_pred
    ),

    "Recall": recall_score(
        y_test,
        y_pred
    ),

    "F1-score": f1_score(
        y_test,
        y_pred
    )
}



# =====================================================
# SAVE JSON
# =====================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )



print("\nMetrics:")

for key, value in metrics.items():

    print(
        f"{key}: {value}"
    )



print("\n====================================")
print("CONFUSION MATRIX GENERATED SUCCESSFULLY")
print("====================================")
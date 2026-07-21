"""
Final Unsupervised Anomaly Detection Model Comparison

ML-Enabled DevOps Framework

Training:
    Only normal software behavior (HasFailure = 0)

Testing:
    Full unseen test dataset

Models:
    1. Isolation Forest
    2. One-Class SVM
    3. Local Outlier Factor
    4. DBSCAN


Metrics:
    Detection Rate
    Precision
    F1 Score
    FPR
    FNR
    AUROC
    Training Time
"""


import os
import time
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler


from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)



# ==================================================
# PATH CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


TRAIN_PATH = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "processed",
        "train.csv"
    )
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



# ==================================================
# LOAD DATA
# ==================================================

print("Loading training data...")

train_df = pd.read_csv(
    TRAIN_PATH
)


print("Loading test data...")

test_df = pd.read_csv(
    TEST_PATH
)



train_df = train_df.fillna(0)

test_df = test_df.fillna(0)



# ==================================================
# NORMAL DATA ONLY FOR TRAINING
# ==================================================

normal_train = train_df[
    train_df[TARGET] == 0
]


print(
    "\nNormal samples used for training:",
    len(normal_train)
)



X_train = normal_train.drop(
    columns=[TARGET],
    errors="ignore"
)


X_test = test_df.drop(
    columns=[TARGET],
    errors="ignore"
)


y_test = test_df[TARGET]



# ==================================================
# FEATURE ENCODING
# ==================================================

X_train = pd.get_dummies(
    X_train
)


X_test = pd.get_dummies(
    X_test
)



# Match training features

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)



print(
    "Feature count:",
    X_train.shape[1]
)



# ==================================================
# FEATURE SCALING
# ==================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)



# ==================================================
# MODELS
# ==================================================

models = {


    "Isolation Forest":

        IsolationForest(

            n_estimators=300,

            contamination=0.10,

            random_state=42,

            n_jobs=-1

        ),



    "One-Class SVM":

        OneClassSVM(

            kernel="rbf",

            gamma="scale",

            nu=0.10

        ),



    "Local Outlier Factor":

        LocalOutlierFactor(

            n_neighbors=20,

            contamination=0.10,

            novelty=True

        ),



    "DBSCAN":

        DBSCAN(

            eps=2,

            min_samples=5

        )

}



# ==================================================
# TRAIN AND EVALUATE
# ==================================================

results = []


print("\n")

print("=" * 100)

print(
    "FINAL ANOMALY DETECTION MODEL COMPARISON"
)

print("=" * 100)



for name, model in models.items():


    print(
        "\nRunning:",
        name
    )


    start_time = time.time()



    # --------------------------------------
    # DBSCAN
    # --------------------------------------

    if name == "DBSCAN":


        prediction = model.fit_predict(
            X_test_scaled
        )


        anomaly_prediction = np.where(
            prediction == -1,
            1,
            0
        )


        auc = np.nan



    else:


        model.fit(
            X_train_scaled
        )


        prediction = model.predict(
            X_test_scaled
        )


        anomaly_prediction = np.where(
            prediction == -1,
            1,
            0
        )



        anomaly_score = model.decision_function(
            X_test_scaled
        )


        auc = roc_auc_score(
            y_test,
            -anomaly_score
        )



    elapsed_time = time.time() - start_time



    # --------------------------------------
    # METRICS
    # --------------------------------------

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



    tn, fp, fn, tp = confusion_matrix(

        y_test,

        anomaly_prediction

    ).ravel()



    fpr = (

        fp / (fp + tn)

        if (fp + tn) > 0

        else 0

    )



    fnr = (

        fn / (fn + tp)

        if (fn + tp) > 0

        else 0

    )



    results.append({


        "Model":

        name,


        "Detection Rate":

        recall,


        "Precision":

        precision,


        "F1 Score":

        f1,


        "FPR":

        fpr,


        "FNR":

        fnr,


        "AUROC":

        auc,


        "Training Time(sec)":

        elapsed_time

    })



# ==================================================
# RESULTS
# ==================================================

results_df = pd.DataFrame(
    results
)



# Sort by F1 score

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)



print("\n")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)



# ==================================================
# SAVE OUTPUTS
# ==================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "final_anomaly_model_comparison.csv"
)


json_path = os.path.join(
    OUTPUT_DIR,
    "final_anomaly_model_comparison.json"
)



results_df.to_csv(
    csv_path,
    index=False
)


results_df.to_json(
    json_path,
    orient="records",
    indent=4
)



print("\nSaved:")
print(csv_path)
print(json_path)
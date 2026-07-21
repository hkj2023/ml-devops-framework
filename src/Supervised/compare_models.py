"""
Model Comparison
ML-Enabled DevOps Framework

Software Defect Prediction

Algorithms:
    1. Tuned Random Forest
    2. Decision Tree
    3. Logistic Regression
    4. Support Vector Machine
    5. KNN
    6. Gaussian Naive Bayes

Evaluation Metrics:
    Accuracy
    Precision
    Recall
    F1-score
    ROC-AUC
    Training Time

Outputs:
    outputs/model_comparison.csv
    outputs/model_comparison.json
"""


import os
import json
import time
import warnings

import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


warnings.filterwarnings("ignore")


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "data",
    "processed",
    "train.csv"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "outputs"
)


DATA_PATH = os.path.normpath(DATA_PATH)
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TARGET = "HasFailure"



# =====================================================
# LOAD DATA
# =====================================================

print("Loading training data...")

df = pd.read_csv(
    DATA_PATH
)


df = df.fillna(0)



# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df.drop(
    columns=[TARGET],
    errors="ignore"
)


y = df[TARGET]



# =====================================================
# ENCODE FEATURES
# =====================================================

print("Encoding features...")

X = pd.get_dummies(
    X
)


print(
    "Feature shape:",
    X.shape
)



# =====================================================
# TRAIN TEST SPLIT
# =====================================================

print("\nSplitting data...")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)



# =====================================================
# SMOTE ONLY ON TRAINING DATA
# =====================================================

print("\nApplying SMOTE...")


smote = SMOTE(
    random_state=42
)


X_train, y_train = smote.fit_resample(

    X_train,

    y_train

)


print("\nBalanced distribution:")

print(
    pd.Series(y_train).value_counts()
)



# =====================================================
# MODELS
# =====================================================

models = {


    # Tuned Random Forest
    "Random Forest":

        RandomForestClassifier(

            n_estimators=500,

            max_depth=None,

            min_samples_split=5,

            min_samples_leaf=2,

            max_features="sqrt",

            random_state=42,

            n_jobs=-1,

            class_weight="balanced"

        ),



    "Decision Tree":

        DecisionTreeClassifier(

            random_state=42

        ),



    "Logistic Regression":

        LogisticRegression(

            max_iter=1000,

            random_state=42

        ),



    "Support Vector Machine":

        SVC(

            probability=True,

            random_state=42

        ),



    "KNN":

        KNeighborsClassifier(),



    "Gaussian Naive Bayes":

        GaussianNB()

}



# =====================================================
# TRAIN AND EVALUATE
# =====================================================


results = []


print("\n")
print("=" * 100)

print(
    "MODEL PERFORMANCE COMPARISON"
)

print("=" * 100)



for name, model in models.items():


    print(
        f"\nTraining {name}..."
    )


    start = time.time()


    model.fit(

        X_train,

        y_train

    )


    training_time = time.time() - start



    # Prediction

    y_pred = model.predict(

        X_test

    )



    # Probability for ROC-AUC

    if hasattr(
        model,
        "predict_proba"
    ):


        y_prob = model.predict_proba(

            X_test

        )[:,1]


    elif hasattr(
        model,
        "decision_function"
    ):


        y_prob = model.decision_function(

            X_test

        )


    else:


        y_prob = y_pred



    accuracy = accuracy_score(

        y_test,

        y_pred

    )


    precision = precision_score(

        y_test,

        y_pred,

        zero_division=0

    )


    recall = recall_score(

        y_test,

        y_pred,

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        y_pred,

        zero_division=0

    )


    auc = roc_auc_score(

        y_test,

        y_prob

    )



    results.append({


        "Model":

            name,


        "Accuracy":

            round(accuracy,4),


        "Precision":

            round(precision,4),


        "Recall":

            round(recall,4),


        "F1 Score":

            round(f1,4),


        "ROC AUC":

            round(auc,4),


        "Training Time(sec)":

            round(training_time,2)


    })




# =====================================================
# RESULTS TABLE
# =====================================================


results_df = pd.DataFrame(
    results
)



results_df = results_df.sort_values(

    by="F1 Score",

    ascending=False

)



print("\n")

print("=" * 100)


print(
    results_df.to_string(
        index=False
    )
)


print("=" * 100)



# =====================================================
# SAVE RESULTS
# =====================================================


csv_path = os.path.join(

    OUTPUT_DIR,

    "model_comparison.csv"

)


json_path = os.path.join(

    OUTPUT_DIR,

    "model_comparison.json"

)



results_df.to_csv(

    csv_path,

    index=False

)



with open(

    json_path,

    "w"

) as f:

    json.dump(

        results,

        f,

        indent=4

    )



print("\nSaved:")

print(csv_path)

print(json_path)



print("\nBest Model based on F1 Score:")

print(

    results_df.iloc[0]["Model"]

)
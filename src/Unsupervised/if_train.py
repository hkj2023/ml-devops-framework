"""
Isolation Forest Training
ML-Enabled DevOps Framework

Final Configuration:
    Training data:
        Only normal software behavior (HasFailure = 0)

    Model:
        Isolation Forest

    Parameters:
        n_estimators = 300
        contamination = 0.10
"""

import os
import json
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================
# BASE DIRECTORY
# ============================================

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


MODEL_DIR = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "models"
    )
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


TARGET = "HasFailure"



# ============================================
# LOAD DATA
# ============================================

print("Loading training data...")


df = pd.read_csv(
    TRAIN_PATH
)


df = df.fillna(0)


print(
    "Original training shape:",
    df.shape
)



# ============================================
# SELECT NORMAL BEHAVIOR ONLY
# ============================================

normal_df = df[
    df[TARGET] == 0
]


print(
    "\nNormal samples used for training:",
    len(normal_df)
)



X = normal_df.drop(
    columns=[TARGET],
    errors="ignore"
)



# ============================================
# ENCODE FEATURES
# ============================================

print("\nEncoding features...")


X = pd.get_dummies(
    X
)


print(
    "Feature shape:",
    X.shape
)



# ============================================
# SCALE FEATURES
# ============================================

print("\nScaling features...")


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)



# ============================================
# TRAIN ISOLATION FOREST
# ============================================

print("\nTraining Isolation Forest...")


model = IsolationForest(

    n_estimators=300,

    contamination=0.10,

    random_state=42,

    n_jobs=-1

)


model.fit(
    X_scaled
)


print(
    "Training complete."
)



# ============================================
# SAVE MODEL
# ============================================

model_path = os.path.join(
    MODEL_DIR,
    "if_model.pkl"
)


scaler_path = os.path.join(
    MODEL_DIR,
    "if_scaler.pkl"
)


joblib.dump(
    model,
    model_path
)


joblib.dump(
    scaler,
    scaler_path
)


print(
    "\nModel saved:"
)

print(
    model_path
)


print(
    "\nScaler saved:"
)

print(
    scaler_path
)



# ============================================
# SAVE FEATURE SCHEMA
# ============================================

feature_path = os.path.join(
    MODEL_DIR,
    "if_feature_names.json"
)


with open(
    feature_path,
    "w"
) as f:

    json.dump(
        X.columns.tolist(),
        f,
        indent=4
    )


print(
    "\nFeatures saved:"
)

print(
    feature_path
)



# ============================================
# MLFLOW TRACKING
# ============================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


mlflow.set_experiment(
    "if-model"
)


with mlflow.start_run():

    mlflow.log_param(
        "model",
        "IsolationForest"
    )


    mlflow.log_param(
        "n_estimators",
        300
    )


    mlflow.log_param(
        "contamination",
        0.10
    )


    mlflow.log_param(
        "training_strategy",
        "Normal behavior only (HasFailure=0)"
    )


    mlflow.log_param(
        "features",
        X.shape[1]
    )


    mlflow.sklearn.log_model(
        model,
        name="model"
    )



print(
    "\nIsolation Forest training completed successfully."
)
import pandas as pd
import joblib
import json
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ============================================
# BASE DIRECTORY
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "models")

TRAIN_PATH = os.path.normpath(TRAIN_PATH)
MODEL_DIR = os.path.normpath(MODEL_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "HasFailure"

# ============================================
# LOAD DATA
# ============================================

print("Loading training data...")
df = pd.read_csv(TRAIN_PATH)

# ============================================
# BASIC CLEANING
# ============================================

df = df.fillna(0)

# ============================================
# SPLIT FEATURES / TARGET
# ============================================

X = df.drop(columns=[TARGET], errors="ignore")
y = df[TARGET]

# ============================================
# ENCODE FEATURES (CRITICAL FIX)
# ============================================

print("Encoding features...")
X = pd.get_dummies(X)

print("Feature shape:", X.shape)

# ============================================
# SMOTE BALANCING
# ============================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

print("\nBalanced class distribution:")
print(pd.Series(y).value_counts())

# ============================================
# TRAIN RANDOM FOREST
# ============================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X, y)

print("Training complete.")

# ============================================
# SAVE MODEL
# ============================================

model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
joblib.dump(model, model_path)

# ============================================
# SAVE FEATURE SCHEMA
# ============================================

feature_path = os.path.join(MODEL_DIR, "rf_feature_names.json")

with open(feature_path, "w") as f:
    json.dump(X.columns.tolist(), f, indent=4)

print("Model + feature schema saved.")

# ============================================
# MLFLOW TRACKING
# ============================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("rf-model")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("sampling", "SMOTE")
    mlflow.log_param("features", X.shape[1])

    mlflow.sklearn.log_model(model, name="model")

print("RF training completed.")
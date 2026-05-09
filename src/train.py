import pandas as pd
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import mlflow

# ============================================
# PATHS
# ============================================

TRAIN_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\processed\train.csv"
MODEL_DIR = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================

print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)

TARGET = "HasFailure"

print("Training data shape:", train_df.shape)

# ============================================
# SPLIT FEATURES AND TARGET
# ============================================

print("Separating features and target...")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

print("Feature shape:", X_train.shape)
print("Target shape:", y_train.shape)

# ============================================
# TRAIN MODEL
# ============================================

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Model training complete.")

# ============================================
# EVALUATION (ON TRAIN SET OR TEST SET)
# ============================================

y_pred = model.predict(X_train)
y_prob = model.predict_proba(X_train)[:, 1]

accuracy = accuracy_score(y_train, y_pred)
roc_auc = roc_auc_score(y_train, y_prob)

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)

# ============================================
# MLFLOW TRACKING (CORRECT PLACE)
# ============================================

mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run():

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

# ============================================
# SAVE MODEL
# ============================================

model_path = os.path.join(MODEL_DIR, "hasfailure_model.pkl")
joblib.dump(model, model_path)

print("Model saved:", model_path)

# ============================================
# SAVE FEATURE NAMES
# ============================================

feature_names = X_train.columns.tolist()

feature_path = os.path.join(MODEL_DIR, "feature_names.json")

with open(feature_path, "w") as f:
    json.dump(feature_names, f, indent=4)

print("Feature names saved:", feature_path)

# ============================================
# DONE
# ============================================

print("Training pipeline completed successfully.")
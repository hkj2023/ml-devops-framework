import pandas as pd
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow

# ============================================
# BASE DIRECTORY (CRITICAL FIX)
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

TRAIN_PATH = os.path.normpath(TRAIN_PATH)
MODEL_DIR = os.path.normpath(MODEL_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# LOAD DATA (SAFETY CHECK)
# ============================================

print("Loading training data...")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

train_df = pd.read_csv(TRAIN_PATH)

TARGET = "HasFailure"

print("Training data shape:", train_df.shape)

# ============================================
# FEATURES / TARGET
# ============================================

print("Separating features and target...")

if TARGET not in train_df.columns:
    raise ValueError("Target column missing in training data")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

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
# METRICS
# ============================================

y_pred = model.predict(X_train)
y_prob = model.predict_proba(X_train)[:, 1]

accuracy = accuracy_score(y_train, y_pred)
roc_auc = roc_auc_score(y_train, y_prob)

print("Train Accuracy:", accuracy)
print("Train ROC-AUC:", roc_auc)

# ============================================
# MLFLOW
# ============================================

mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)

    mlflow.log_metric("train_accuracy", accuracy)
    mlflow.log_metric("train_roc_auc", roc_auc)

# ============================================
# SAVE MODEL
# ============================================

model_path = os.path.join(MODEL_DIR, "hasfailure_model.pkl")
joblib.dump(model, model_path)

print("Model saved:", model_path)

# ============================================
# SAVE FEATURES
# ============================================

feature_names = X_train.columns.tolist()

feature_path = os.path.join(MODEL_DIR, "feature_names.json")

with open(feature_path, "w") as f:
    json.dump(feature_names, f, indent=4)

print("Feature names saved:", feature_path)

print("Training pipeline completed successfully.")
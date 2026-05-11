import pandas as pd
import joblib
import json
import os
import mlflow

from sklearn.ensemble import RandomForestClassifier

# ============================================
# BASE DIRECTORY
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

TRAIN_PATH = os.path.normpath(TRAIN_PATH)
MODEL_DIR = os.path.normpath(MODEL_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# LOAD TRAINING DATA
# ============================================

print("Loading training data...")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

train_df = pd.read_csv(TRAIN_PATH)

TARGET = "HasFailure"

print("Training data shape:", train_df.shape)

if TARGET not in train_df.columns:
    raise ValueError(f"Target column '{TARGET}' missing")

# ============================================
# FEATURES / TARGET
# ============================================

print("Separating features and target...")

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
# SAVE MODEL
# ============================================

model_path = os.path.join(MODEL_DIR, "model.pkl")
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
# LOG TO MLFLOW
# ============================================

mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)

print("Training pipeline completed successfully.")
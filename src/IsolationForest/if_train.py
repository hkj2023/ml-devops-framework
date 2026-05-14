import pandas as pd
import joblib
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================
# BASE DIRECTORY
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "data", "processed", "train.csv")
)

MODEL_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "models")
)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "HasFailure"

# ============================================
# LOAD DATA
# ============================================
print("Loading training data...")
df = pd.read_csv(TRAIN_PATH)
X = df.drop(columns=[TARGET], errors="ignore")

print("\nTraining shape:", X.shape)

# ============================================
# SCALE FEATURES
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# TRAIN ISOLATION FOREST
# ============================================
print("\nTraining Isolation Forest...")
model = IsolationForest(
    n_estimators=300,
    contamination=0.1,   # tuned to balance precision/recall
    random_state=42
)
model.fit(X_scaled)
print("Training complete.")

# ============================================
# SAVE MODEL + SCALER
# ============================================
model_path = os.path.join(MODEL_DIR, "if_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "if_scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("\nModel saved successfully at:", model_path)

# ============================================
# SAVE FEATURES
# ============================================
feature_path = os.path.join(MODEL_DIR, "if_feature_names.json")
with open(feature_path, "w") as f:
    json.dump(X.columns.tolist(), f, indent=4)
print("Features saved at:", feature_path)

# ============================================
# MLFLOW TRACKING
# ============================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("if-model")

with mlflow.start_run():
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("contamination", 0.1)
    mlflow.sklearn.log_model(model, name="model")

print("\nIF training completed successfully.")

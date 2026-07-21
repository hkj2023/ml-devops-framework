import os
import joblib
import json
import pandas as pd

# =====================================================
# BASE DIRECTORY (robust for pytest execution)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "models", "rf_model.pkl"))
FEATURE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "models", "rf_feature_names.json"))
TRAIN_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data", "processed", "train.csv"))

TARGET = "HasFailure"

# =====================================================
# TEST 1: MODEL FILE EXISTS
# =====================================================

def test_rf_model_file_created():
    assert os.path.exists(MODEL_PATH), "❌ RF model file not found"


# =====================================================
# TEST 2: FEATURE FILE EXISTS
# =====================================================

def test_rf_feature_file_created():
    assert os.path.exists(FEATURE_PATH), "❌ Feature file not found"


# =====================================================
# TEST 3: MODEL LOADS
# =====================================================

def test_rf_model_loads_correctly():
    model = joblib.load(MODEL_PATH)
    assert model is not None, "❌ Model failed to load"


# =====================================================
# TEST 4: FEATURE SCHEMA VALID
# =====================================================

def test_rf_feature_schema_valid():
    with open(FEATURE_PATH, "r") as f:
        features = json.load(f)

    assert isinstance(features, list), "❌ Feature schema is not a list"
    assert len(features) > 0, "❌ Feature list is empty"


# =====================================================
# TEST 5: MODEL CAN PREDICT
# =====================================================

def test_rf_model_can_predict():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(TRAIN_PATH)

    assert TARGET in df.columns, f"❌ Target column '{TARGET}' missing in dataset"

    X = df.drop(columns=[TARGET], errors="ignore")

    # Load feature schema for consistency (IMPORTANT FIX)
    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)

    # Align features exactly like production
    X = X.reindex(columns=feature_names, fill_value=0)

    preds = model.predict(X)

    assert len(preds) == len(X), "❌ Prediction length mismatch"
    assert set(preds).issubset({0, 1}), "❌ Invalid prediction values (must be 0 or 1)"
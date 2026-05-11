import pandas as pd
import joblib
import json
import os

# =====================================================
# BASE DIRECTORY (CI + DOCKER SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_names.json")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "test.csv")

MODEL_PATH = os.path.normpath(MODEL_PATH)
FEATURE_PATH = os.path.normpath(FEATURE_PATH)
TEST_PATH = os.path.normpath(TEST_PATH)

# =====================================================
# TEST 1: Prediction runs successfully
# =====================================================

def test_prediction_runs():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)

    X = df.reindex(columns=feature_names, fill_value=0)

    preds = model.predict(X)

    assert len(preds) == len(X), "Prediction length mismatch"

# =====================================================
# TEST 2: Valid class output
# =====================================================

def test_predictions_valid_classes():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)

    X = df.reindex(columns=feature_names, fill_value=0)

    preds = model.predict(X)

    assert set(preds).issubset({0, 1}), "Invalid class labels detected"

# =====================================================
# TEST 3: Probability range validation (IMPORTANT)
# =====================================================

def test_prediction_probabilities():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)

    X = df.reindex(columns=feature_names, fill_value=0)

    probs = model.predict_proba(X)[:, 1]

    assert all(0.0 <= p <= 1.0 for p in probs), "Invalid probability values"
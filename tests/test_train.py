import joblib
import os
import numpy as np

# =====================================================
# BASE DIRECTORY (CI + DOCKER SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_names.json")

MODEL_PATH = os.path.normpath(MODEL_PATH)
FEATURE_PATH = os.path.normpath(FEATURE_PATH)

# =====================================================
# TESTS
# =====================================================

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Model file missing"


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Model failed to load"


def test_model_has_predict():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Model missing predict method"


def test_model_has_proba():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba"), "Model missing predict_proba method"


def test_model_prediction_range():
    """
    Ensures model outputs valid probability range [0, 1]
    """
    model = joblib.load(MODEL_PATH)

    # dummy input (safe structural test only)
    dummy_input = np.zeros((1, model.n_features_in_))

    prob = model.predict_proba(dummy_input)[0][1]

    assert 0.0 <= prob <= 1.0, "Probability out of range"
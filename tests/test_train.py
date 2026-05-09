import joblib
import os

MODEL_PATH = "models/hasfailure_model.pkl"

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)

def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None

def test_model_has_predict():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")

def test_model_has_proba():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba")
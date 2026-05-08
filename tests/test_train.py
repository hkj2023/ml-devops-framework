import os
import joblib

MODEL_PATH = "models/hasfailure_model.pkl"

def test_model_exists():
    assert os.path.exists(MODEL_PATH)

def test_model_load():
    model = joblib.load(MODEL_PATH)
    assert model is not None
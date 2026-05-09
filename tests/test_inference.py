import pandas as pd
import joblib

MODEL_PATH = "models/hasfailure_model.pkl"
TEST_PATH = "data/processed/test.csv"

def test_prediction_runs():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    X = df.drop(columns=["HasFailure"])
    preds = model.predict(X)

    assert len(preds) == len(X)

def test_predictions_valid_classes():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    X = df.drop(columns=["HasFailure"])
    preds = model.predict(X)

    assert set(preds).issubset({0, 1})
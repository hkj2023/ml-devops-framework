import pandas as pd
import joblib
import json

MODEL_PATH = "models/hasfailure_model.pkl"
FEATURE_PATH = "models/feature_names.json"

def test_inference():

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r") as f:
        features = json.load(f)

    df = pd.read_csv("data/ml_ready_dataset.csv")

    df = df.drop(columns=["HasFailure"], errors="ignore")

    df = df.reindex(columns=features, fill_value=0)

    preds = model.predict(df.head(5))

    assert len(preds) == 5
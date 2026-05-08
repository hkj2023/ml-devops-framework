import pandas as pd
import joblib
import json

MODEL_PATH = r"models/hasfailure_model.pkl"

FEATURE_PATH = r"models/feature_names.json"

def test_prediction_shape():

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Load feature names used during training
    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)

    # Load dataset
    df = pd.read_csv(
        r"data/ml_ready_dataset.csv"
    )

    # Remove target column
    X = df.drop(columns=["HasFailure"], errors="ignore")

    # Keep ONLY training features
    X = X[feature_names]

    # Predict
    preds = model.predict(X.head(5))

    # Assertions
    assert len(preds) == 5
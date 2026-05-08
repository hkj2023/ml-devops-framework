import os
import subprocess
import joblib

def test_training_creates_model():
    # Run training script
    subprocess.run(["python", "RandomForest.py"], check=True)
    # Check model file exists
    assert os.path.exists("outputs/defect_prediction.pkl")
    # Check model can be loaded
    saved = joblib.load("outputs/defect_prediction.pkl")
    assert "model" in saved and "features" in saved
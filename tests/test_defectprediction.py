import os
import subprocess
import pandas as pd

def test_inference_creates_predictions():
    # Run training first to ensure model exists
    subprocess.run(["python", "RandomForest.py"], check=True)

    # Then run inference
    subprocess.run(["python", "inference.py"], check=True)

    # Check predictions file exists
    assert os.path.exists("outputs/inference_output.csv")

    # Validate predictions file content
    df = pd.read_csv("outputs/inference_output.csv")
    assert "Prediction" in df.columns
    assert len(df) > 0
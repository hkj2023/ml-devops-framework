import os
import pandas as pd
import subprocess

def test_prep_creates_csv():
    # Run prep.py
    subprocess.run(["python", "prep.py"], check=True)

    # Check output file exists
    assert os.path.exists("outputs/new_data.csv")

    # Load the file
    df = pd.read_csv("outputs/new_data.csv")

    # Check for expected columns (adjust to your schema)
    expected_columns = ["BuildID", "BugID", "Duration(min)", "ExecutionTime"]
    for col in expected_columns:
        assert col in df.columns

    # Sanity check: file should have rows
    assert len(df) > 0
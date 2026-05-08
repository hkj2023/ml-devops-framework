import pandas as pd

def test_prep_output():
    df = pd.read_csv("data/ml_ready_dataset.csv")

    assert df is not None
    assert len(df) > 0
    assert "HasFailure" in df.columns
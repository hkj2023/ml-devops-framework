import pandas as pd

DATA_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\ml_ready_dataset.csv"

def test_data_load():
    df = pd.read_csv(DATA_PATH)
    assert df is not None
    assert len(df) > 0


def test_target_exists():
    df = pd.read_csv(DATA_PATH)
    assert "HasFailure" in df.columns


def test_no_all_null_dataframe():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().all().sum() < len(df.columns)
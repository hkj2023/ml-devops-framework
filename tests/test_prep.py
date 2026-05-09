import pandas as pd

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

def test_train_data_exists():
    df = pd.read_csv(TRAIN_PATH)
    assert df.shape[0] > 0

def test_test_data_exists():
    df = pd.read_csv(TEST_PATH)
    assert df.shape[0] > 0

def test_target_column_exists():
    df = pd.read_csv(TRAIN_PATH)
    assert "HasFailure" in df.columns

def test_no_missing_values():
    df = pd.read_csv(TRAIN_PATH)
    assert df.isnull().sum().sum() == 0
import pandas as pd
import os

# =====================================================
# BASE DIRECTORY (CI + DOCKER SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "test.csv")

TRAIN_PATH = os.path.normpath(TRAIN_PATH)
TEST_PATH = os.path.normpath(TEST_PATH)

TARGET = "HasFailure"

# =====================================================
# TESTS
# =====================================================

def test_train_data_exists():
    df = pd.read_csv(TRAIN_PATH)
    assert df.shape[0] > 0, "Train dataset is empty"


def test_test_data_exists():
    df = pd.read_csv(TEST_PATH)
    assert df.shape[0] > 0, "Test dataset is empty"


def test_target_column_exists():
    df = pd.read_csv(TRAIN_PATH)
    assert TARGET in df.columns, "Target column missing"


def test_no_missing_values_train():
    df = pd.read_csv(TRAIN_PATH)
    assert df.isnull().sum().sum() == 0, "Missing values found in train data"


def test_no_missing_values_test():
    df = pd.read_csv(TEST_PATH)
    assert df.isnull().sum().sum() == 0, "Missing values found in test data"
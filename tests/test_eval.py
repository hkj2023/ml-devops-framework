import os
import json

EVAL_PATH = "models/evaluation_metrics.json"

def test_eval_file_exists():

    assert os.path.exists(EVAL_PATH)

def test_eval_content():

    with open(EVAL_PATH, "r") as f:
        data = json.load(f)

    assert "accuracy" in data
    assert "roc_auc" in data
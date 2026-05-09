import json
import os

METRICS_PATH = "outputs/evaluation_metrics.json"

def test_metrics_file_exists():
    assert os.path.exists(METRICS_PATH)

def test_metrics_structure():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "balanced_accuracy" in metrics

def test_metrics_range():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
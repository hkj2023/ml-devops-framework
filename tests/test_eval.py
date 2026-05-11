import json
import os

# =====================================================
# BASE DIRECTORY (CI + DOCKER SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS_PATH = os.path.join(BASE_DIR, "..", "outputs", "metrics.json")

METRICS_PATH = os.path.normpath(METRICS_PATH)

# =====================================================
# TEST 1: File exists
# =====================================================

def test_metrics_file_exists():
    assert os.path.exists(METRICS_PATH), "metrics.json not found"

# =====================================================
# TEST 2: Required structure
# =====================================================

def test_metrics_structure():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    required_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc"
    ]

    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"

# =====================================================
# TEST 3: Value range validation
# =====================================================

def test_metrics_range():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert 0 <= metrics["auroc"] <= 1
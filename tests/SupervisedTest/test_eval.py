import os
import json

# =====================================================
# BASE PATH (CI/CD SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "outputs", "rf_metrics.json")
)

# =====================================================
# TEST 1: FILE EXISTS
# =====================================================

def test_rf_eval_output_file_exists():
    assert os.path.exists(OUTPUT_PATH), "❌ RF metrics file not found"


# =====================================================
# HELPER: SAFE LOAD JSON
# =====================================================

def _load_json():
    with open(OUTPUT_PATH, "r") as f:
        return json.load(f)


# =====================================================
# TEST 2: VALID JSON STRUCTURE
# =====================================================

def test_rf_eval_output_is_valid_json():
    data = _load_json()
    assert isinstance(data, dict), "❌ Output is not a JSON object"


# =====================================================
# TEST 3: REQUIRED KEYS
# =====================================================

def test_rf_eval_required_keys_exist():
    data = _load_json()

    required_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "execution_status"
    ]

    for key in required_keys:
        assert key in data, f"❌ Missing key: {key}"


# =====================================================
# TEST 4: METRIC VALUE RANGE
# =====================================================

def test_rf_eval_metrics_valid_range():
    data = _load_json()

    metric_keys = ["accuracy", "precision", "recall", "f1_score", "auroc"]

    for key in metric_keys:
        assert key in data, f"❌ Missing metric: {key}"

        value = data[key]

        assert isinstance(value, (int, float)), f"❌ {key} is not numeric"
        assert 0.0 <= value <= 1.0, f"❌ {key} out of range [0,1]: {value}"


# =====================================================
# TEST 5: EXECUTION STATUS
# =====================================================

def test_rf_eval_execution_status():
    data = _load_json()

    assert "execution_status" in data, "❌ Missing execution_status"
    assert data["execution_status"] == "SUCCESS", "❌ Evaluation did not complete successfully"
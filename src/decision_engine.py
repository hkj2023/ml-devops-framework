import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_PATH = os.path.join(BASE_DIR, "..", "outputs", "risk_prediction.json")

def decide_test_path(risk_level):
    if risk_level == "HIGH":
        return "tests/full/"
    elif risk_level == "MEDIUM":
        return "tests/critical/"
    else:
        return "tests/unit/"


def main():
    if not os.path.exists(RISK_PATH):
        raise FileNotFoundError("Run inference.py first")

    with open(RISK_PATH, "r") as f:
        data = json.load(f)

    # ✔ FIXED: correct key from your JSON
    risk = data["risk_level"]

    test_path = decide_test_path(risk)

    print(test_path)


if __name__ == "__main__":
    main()
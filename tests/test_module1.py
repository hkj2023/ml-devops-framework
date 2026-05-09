# tests/test_module1.py
import numpy as np
from src.module1 import train_and_predict, evaluate

def test_train_predict_evaluate():
    # Dummy dataset
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1.5], [2.5]])
    y_true = np.array([0, 1])

    # Train and predict
    preds, model = train_and_predict(X_train, y_train, X_test)

    # Evaluate
    acc = evaluate(y_true, preds)

    # Assertions
    assert preds.shape[0] == X_test.shape[0]
    assert 0.0 <= acc <= 1.0
    assert acc >= 0.5  # sanity threshold

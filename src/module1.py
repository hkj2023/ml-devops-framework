# src/module1.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_predict(X_train, y_train, X_test):
    """Train a logistic regression and predict."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model

def evaluate(y_true, y_pred):
    """Evaluate predictions with accuracy."""
    return accuracy_score(y_true, y_pred)

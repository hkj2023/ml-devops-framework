import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def test_model_training():
    df = pd.read_csv(r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\ml_ready_dataset.csv")

    target = "HasFailure"

    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    assert model is not None
    assert len(model.feature_importances_) == X_train.shape[1]
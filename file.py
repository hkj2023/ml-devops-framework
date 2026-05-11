import pandas as pd
import joblib

model = joblib.load("models/hasfailure_model.pkl")

import json
features = json.load(open("models/feature_names.json"))

imp = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(imp.head(15))
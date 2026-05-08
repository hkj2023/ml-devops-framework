import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)
from sklearn.utils import resample

# =====================================================
# LOAD DATASET
# =====================================================

print("\n================ LOADING DATASET ================\n")

df = pd.read_csv(
    r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\ml_ready_dataset.csv"
)

print("Dataset Shape:", df.shape)

# =====================================================
# REMOVE DUPLICATES
# =====================================================

print("\n================ REMOVING DUPLICATES ================\n")

df = df.drop_duplicates()
print("Dataset Shape after duplicates:", df.shape)

# =====================================================
# HANDLE NULL VALUES
# =====================================================

print("\n================ NULL VALUE CHECK ================\n")

df = df.fillna(0)

print("Null values handled")

# =====================================================
# TARGET COLUMN
# =====================================================

target = "HasFailure"

# =====================================================
# REMOVE LEAKAGE COLUMNS
# =====================================================

print("\n================ REMOVING LEAKAGE COLUMNS ================\n")

leakage_cols = [
    "HasFailure",
    "DefectLabel",
    "DefectCount",
    "Result_fail",
    "Result_pass",
    "FailureRate",
    "FailureRatio",
    "SuccessRate",
    "TestsFailed",
    "BugID",
    "Fix Time",
    "Status_y_failed",
    "Status_failed",
    "CommitID",
    "BuildID",
    "LogID",
    "TestID_x",
    "TestID_y",
    "Severity",
    "Status_x"
]

existing_leakage_cols = [c for c in leakage_cols if c in df.columns]

X = df.drop(columns=existing_leakage_cols, errors="ignore")
y = df[target]

print("Remaining Features:", X.shape[1])

# =====================================================
# SAVE FEATURE NAMES (IMPORTANT FOR INFERENCE FIX)
# =====================================================

feature_names = X.columns.tolist()

feature_path = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\feature_names.json"

with open(feature_path, "w") as f:
    json.dump(feature_names, f, indent=4)

print("Feature names saved")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# BALANCE TRAINING DATA ONLY
# =====================================================

train_df = pd.concat([X_train, y_train], axis=1)

majority = train_df[train_df[target] == 1]
minority = train_df[train_df[target] == 0]

minority_up = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

train_balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)

X_train = train_balanced.drop(columns=[target])
y_train = train_balanced[target]

# =====================================================
# MODEL TRAINING
# =====================================================

print("\n================ TRAINING MODEL ================\n")

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

print("Model trained")

# =====================================================
# EVALUATION
# =====================================================

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("Balanced Accuracy:", balanced_acc)
print("ROC-AUC:", roc_auc)

# =====================================================
# FEATURE IMPORTANCE (CSV + JSON)
# =====================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

# CSV
importance_csv = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\outputs\hasfailure_feature_importance.csv"
importance_df.to_csv(importance_csv, index=False)

# JSON
importance_json_path = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\hasfailure_feature_importance.json"

feature_importance_json = [
    {"feature": row["Feature"], "importance": float(row["Importance"])}
    for _, row in importance_df.iterrows()
]

with open(importance_json_path, "w") as f:
    json.dump(feature_importance_json, f, indent=4)

print("Feature importance saved (CSV + JSON)")

# =====================================================
# SAVE MODEL
# =====================================================

model_path = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\hasfailure_model.pkl"

joblib.dump(clf, model_path)

print("Model saved:", model_path)

# =====================================================
# SAVE TRAINING METRICS (IMPORTANT FOR DEVOPS)
# =====================================================

metrics = {
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "roc_auc": float(roc_auc)
}

metrics_path = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\training_metrics.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved")

# =====================================================
# SUMMARY
# =====================================================

print("\n================ DONE ================\n")
print("Training pipeline completed successfully")
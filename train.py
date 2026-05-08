import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

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
    r"C:\Users\OLLRP\Desktop\Test\ml_ready_dataset.csv"
)

print("Dataset Shape:", df.shape)

# =====================================================
# REMOVE DUPLICATES
# =====================================================

print("\n================ REMOVING DUPLICATES ================\n")

duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print("Duplicates Removed")
    print("New Dataset Shape:", df.shape)

# =====================================================
# HANDLE NULL VALUES
# =====================================================

print("\n================ NULL VALUE CHECK ================\n")

print(df.isnull().sum())

# Fill null values
df = df.fillna(0)

print("\nNull values handled successfully")

# =====================================================
# TARGET COLUMN
# =====================================================

target = "HasFailure"

# =====================================================
# TARGET DISTRIBUTION
# =====================================================

print("\n================ TARGET DISTRIBUTION ================\n")

print(df[target].value_counts())

print("\nPercentage Distribution:")

print(df[target].value_counts(normalize=True) * 100)

# =====================================================
# REMOVE LEAKAGE COLUMNS
# =====================================================

print("\n================ REMOVING LEAKAGE COLUMNS ================\n")

leakage_cols = [

    # Targets
    "HasFailure",
    "DefectLabel",
    "DefectCount",

    # Direct result columns
    "Result_fail",
    "Result_pass",

    # Failure-derived metrics
    "FailureRate",
    "FailureRatio",
    "SuccessRate",
    "TestsFailed",

    # Bug/Fix information
    "BugID",
    "Fix Time",

    # Status columns
    "Status_y_failed",
    "Status_failed",

    # IDs that may leak row patterns
    "CommitID",
    "BuildID",
    "LogID",
    "TestID_x",
    "TestID_y",

    # Potential leakage
    "Severity",
    "Status_x"
]

existing_leakage_cols = [
    col for col in leakage_cols
    if col in df.columns
]

print("Removed Columns:\n")
print(existing_leakage_cols)

# =====================================================
# FEATURE / TARGET SPLIT
# =====================================================

X = df.drop(
    columns=existing_leakage_cols,
    errors="ignore"
)

y = df[target]

print("\nRemaining Feature Count:", X.shape[1])

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

print("\n================ TRAIN TEST SPLIT ================\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# =====================================================
# BALANCE ONLY TRAINING DATA
# =====================================================

print("\n================ BALANCING TRAINING DATA ================\n")

train_df = pd.concat(
    [X_train, y_train],
    axis=1
)

df_majority = train_df[train_df[target] == 1]

df_minority = train_df[train_df[target] == 0]

print("Majority Class Size:", len(df_majority))

print("Minority Class Size:", len(df_minority))

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

train_balanced = pd.concat([
    df_majority,
    df_minority_upsampled
])

train_balanced = train_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train = train_balanced.drop(columns=[target])

y_train = train_balanced[target]

print("\nBalanced Training Distribution:")

print(y_train.value_counts())

# =====================================================
# MODEL TRAINING
# =====================================================

print("\n================ TRAINING MODEL ================\n")

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

print("✅ Model Training Complete")

# =====================================================
# PRINT TRAINED MODEL
# =====================================================

print("\n================ TRAINED MODEL DETAILS ================\n")

print(clf)

print("\nNumber of Trees:")
print(len(clf.estimators_))

print("\nClasses:")
print(clf.classes_)

print("\nNumber of Features Used:")
print(clf.n_features_in_)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = clf.predict(X_test)

y_prob = clf.predict_proba(X_test)[:, 1]

# =====================================================
# MODEL EVALUATION
# =====================================================

print("\n================ MODEL EVALUATION ================\n")

accuracy = accuracy_score(y_test, y_pred)

balanced_acc = balanced_accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("Accuracy Score:")
print(accuracy)

print("\nBalanced Accuracy Score:")
print(balanced_acc)

print("\nROC-AUC Score:")
print(roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    digits=6
))

# =====================================================
# CROSS VALIDATION
# =====================================================

print("\n================ CROSS VALIDATION ================\n")

cv_scores = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Cross Validation Scores:")
print(cv_scores)

print("\nAverage Cross Validation Score:")
print(cv_scores.mean())

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

print("\n================ FEATURE IMPORTANCE ================\n")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(20))

# =====================================================
# SAVE FEATURE IMPORTANCE
# =====================================================

importance_path = (
    r"C:\Users\OLLRP\Desktop\Test\hasfailure_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print("\n📁 Feature Importance Saved At:")
print(importance_path)

# =====================================================
# SAVE MODEL
# =====================================================

print("\n================ SAVING MODEL ================\n")

model_path = (
    r"C:\Users\OLLRP\Desktop\Test\hasfailure_model.pkl"
)

joblib.dump(clf, model_path)

print("💾 Model Saved Successfully At:")
print(model_path)

# =====================================================
# LOAD AND PRINT SAVED MODEL
# =====================================================

print("\n================ LOADED SAVED MODEL ================\n")

loaded_model = joblib.load(model_path)

print(loaded_model)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n================ FINAL SUMMARY ================\n")

print(f"Final Accuracy: {accuracy:.4f}")

print(f"Balanced Accuracy: {balanced_acc:.4f}")

print(f"ROC-AUC Score: {roc_auc:.4f}")

print(f"Cross Validation Mean: {cv_scores.mean():.4f}")

print("\n✅ HasFailure Prediction Process Complete")
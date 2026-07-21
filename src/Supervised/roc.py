import os
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

# ======================================================
# PATHS
# ======================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "HasFailure"

# ======================================================
# LOAD DATA
# ======================================================

print("Loading datasets...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train = train.fillna(0)
test = test.fillna(0)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# ======================================================
# ENCODE CATEGORICAL FEATURES
# ======================================================

combined = pd.concat([X_train, X_test], axis=0)

combined = pd.get_dummies(combined)

X_train = combined.iloc[:len(X_train), :]
X_test = combined.iloc[len(X_train):, :]

# ======================================================
# SCALE DATA
# ======================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================================
# APPLY SMOTE
# ======================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

# ======================================================
# MODELS
# ======================================================

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Gaussian NB":
        GaussianNB(),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Support Vector Machine":
        SVC(
            probability=True,
            random_state=42
        )
}

# ======================================================
# ROC CURVES
# ======================================================

plt.figure(figsize=(10,8))

for name, model in models.items():

    print(f"Training {name}")

    model.fit(
        X_train_smote,
        y_train_smote
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:,1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC={roc_auc:.4f})"
    )

# Random classifier

plt.plot(
    [0,1],
    [0,1],
    '--',
    color='black',
    label='Random Guess'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of Supervised Learning Models")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nROC comparison figure saved.")
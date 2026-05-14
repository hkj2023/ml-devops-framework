import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# =====================================================
# BASE DIRECTORY (PROJECT ROOT SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_PATH = os.path.join(DATA_DIR, "ml_ready_dataset.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_names.json")
CURRENT_METRICS_PATH = os.path.join(OUTPUT_DIR, "current_commit_metrics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# CONFIGURATION
# =====================================================

TARGET = "HasFailure"

LEAKAGE_COLUMNS = [
    "FailureRate",
    "FailureRatio",
    "SuccessRate",
    "TestsFailed",
    "DefectLabel",
    "DefectCount",
    "Result_fail",
    "Result_pass",
    "Status_failed",
    "Status_success",
    "BugID",
    "Fix Time"
]

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading raw dataset...")

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)

print("Original shape:", df.shape)

# =====================================================
# CLEANING
# =====================================================

df = df.drop_duplicates()
df = df.fillna(0)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found!")

# =====================================================
# REMOVE LEAKAGE
# =====================================================

existing_leakage = [c for c in LEAKAGE_COLUMNS if c in df.columns]
df = df.drop(columns=existing_leakage, errors="ignore")

print("Removed leakage columns:", existing_leakage)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

if "LinesAdded" in df.columns and "LinesRemoved" in df.columns:
    df["CodeChurn"] = df["LinesAdded"] + df["LinesRemoved"]

if "LinesAdded" in df.columns and "FilesChanged" in df.columns:
    df["ChangeDensity"] = df["LinesAdded"] / (df["FilesChanged"] + 1)

# =====================================================
# ENCODING
# =====================================================

df_encoded = pd.get_dummies(df)

if TARGET not in df_encoded.columns:
    raise ValueError("Target column lost during processing!")

# =====================================================
# FEATURE SCHEMA
# =====================================================

features_df = df_encoded.drop(columns=[TARGET])
feature_names = features_df.columns.tolist()

with open(SCHEMA_PATH, "w") as f:
    json.dump(feature_names, f, indent=4)

print(f"Feature schema saved: {len(feature_names)} features")

# =====================================================
# SPLIT DATASET
# =====================================================

train_df, test_df = train_test_split(
    df_encoded,
    test_size=0.2,
    random_state=42,
    stratify=df_encoded[TARGET]
)

train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# =====================================================
# CURRENT COMMIT METRICS
# =====================================================

current = {
    "LinesAdded": 45,
    "LinesRemoved": 12,
    "FilesChanged": 3
}

current["CodeChurn"] = current["LinesAdded"] + current["LinesRemoved"]
current["ChangeDensity"] = current["LinesAdded"] / (current["FilesChanged"] + 1)

current_df = pd.DataFrame([current])
current_df = pd.get_dummies(current_df)

current_df = current_df.reindex(columns=feature_names, fill_value=0)

current_df.to_csv(CURRENT_METRICS_PATH, index=False)

print("Saved:", CURRENT_METRICS_PATH)
print("Shape:", current_df.shape)

# =====================================================
# FINAL REPORT
# =====================================================

print("\n=========== PREP COMPLETE ===========")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Feature count:", len(feature_names))
print("Data saved safely.")
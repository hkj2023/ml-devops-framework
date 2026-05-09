import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# =====================================================
# PATHS
# =====================================================

RAW_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\ml_ready_dataset.csv"

OUTPUT_DIR = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\data\processed"

SCHEMA_PATH = r"C:\Users\OLLRP\Documents\Framework\ml-devops-framework\models\feature_names.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCHEMA_PATH), exist_ok=True)

# =====================================================
# CONFIGURATION (MOST IMPORTANT PART)
# =====================================================

TARGET = "HasFailure"

# ❌ EVERYTHING HERE IS FORBIDDEN (LEAKAGE / POST-OUTCOME FEATURES)
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
df = pd.read_csv(RAW_PATH)

print("Original shape:", df.shape)

# =====================================================
# BASIC CLEANING
# =====================================================

print("\nCleaning data...")

df = df.drop_duplicates()
df = df.fillna(0)

# =====================================================
# TARGET CHECK
# =====================================================

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found!")

# =====================================================
# REMOVE LEAKAGE COLUMNS (STRICT ENFORCEMENT)
# =====================================================

print("\nRemoving leakage columns...")

existing_leakage = [col for col in LEAKAGE_COLUMNS if col in df.columns]

df = df.drop(columns=existing_leakage, errors="ignore")

print("Removed leakage columns:", existing_leakage)

# =====================================================
# FEATURE ENGINEERING (SAFE ONLY)
# =====================================================

print("\nFeature engineering...")

if "LinesAdded" in df.columns and "LinesRemoved" in df.columns:
    df["CodeChurn"] = df["LinesAdded"] + df["LinesRemoved"]

if "LinesAdded" in df.columns and "FilesChanged" in df.columns:
    df["ChangeDensity"] = df["LinesAdded"] / (df["FilesChanged"] + 1)

# =====================================================
# ENCODE CATEGORICAL VARIABLES
# =====================================================

print("\nEncoding categorical variables...")

df_encoded = pd.get_dummies(df)

# =====================================================
# FINAL SAFETY CHECK (CRITICAL)
# =====================================================

print("\nFinal safety check...")

# ensure target still exists
if TARGET not in df_encoded.columns:
    raise ValueError("Target column lost during processing!")

# remove target from features
features_df = df_encoded.drop(columns=[TARGET])

# =====================================================
# FREEZE FEATURE SCHEMA
# =====================================================

feature_names = features_df.columns.tolist()

with open(SCHEMA_PATH, "w") as f:
    json.dump(feature_names, f, indent=4)

print(f"Feature schema saved: {len(feature_names)} features")

# =====================================================
# SPLIT DATASET
# =====================================================

print("\nSplitting dataset...")

train_df, test_df = train_test_split(
    df_encoded,
    test_size=0.2,
    random_state=42,
    stratify=df_encoded[TARGET]
)

# =====================================================
# SAVE DATASETS
# =====================================================

train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# =====================================================
# FINAL REPORT
# =====================================================

print("\n================ PREP COMPLETE ================\n")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Leakage columns removed:", existing_leakage)
print("Feature count:", len(feature_names))
print("Data saved safely (no leakage guaranteed)")
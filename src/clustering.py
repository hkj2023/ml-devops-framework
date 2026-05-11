import pandas as pd
import os
import json

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================
# PATHS
# ============================================

DATA_PATH = "data/processed/train.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLUSTER_OUTPUT = os.path.join(OUTPUT_DIR, "cluster_results.csv")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "cluster_summary.json")

TARGET = "HasFailure"

# ============================================
# LOAD DATA
# ============================================

print("\nLoading training data...")

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)

# Remove target column
if TARGET in df.columns:
    X = df.drop(columns=[TARGET])
else:
    X = df.copy()

# ============================================
# SCALE FEATURES
# ============================================

print("\nScaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# RUN KMEANS
# ============================================

print("\nRunning clustering...")

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

# ============================================
# SAVE RESULTS
# ============================================

df.to_csv(CLUSTER_OUTPUT, index=False)

cluster_counts = df["Cluster"].value_counts().to_dict()

summary = {
    "model": "KMeans",
    "total_samples": len(df),
    "n_clusters": 3,
    "cluster_distribution": {
        str(k): int(v)
        for k, v in cluster_counts.items()
    }
}

with open(SUMMARY_OUTPUT, "w") as f:
    json.dump(summary, f, indent=4)

# ============================================
# DONE
# ============================================

print("\n================ DONE ================\n")
print("Cluster distribution:", cluster_counts)
print("Saved:")
print("-", CLUSTER_OUTPUT)
print("-", SUMMARY_OUTPUT)
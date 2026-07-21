"""
SHAP Explainability Analysis
ML-Enabled DevOps Framework

Isolation Forest Anomaly Detection

Model:
    models/if_model.pkl

Test Data:
    data/processed/test.csv

Outputs:
    outputs/if_shap/
"""


import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap



# =====================================================
# PROJECT ROOT DIRECTORY
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../"
    )
)



# =====================================================
# PATHS
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "if_model.pkl"
)


TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test.csv"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "if_shap"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =====================================================
# START
# =====================================================

print("\n====================================")
print("SHAP ISOLATION FOREST EXPLANATION")
print("====================================")


print("\nModel Path:")
print(MODEL_PATH)


print("\nTest Data Path:")
print(TEST_DATA_PATH)



# =====================================================
# CHECK FILES
# =====================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nIsolation Forest model not found:\n{MODEL_PATH}"
    )


if not os.path.exists(TEST_DATA_PATH):

    raise FileNotFoundError(
        f"\nTest data not found:\n{TEST_DATA_PATH}"
    )



# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading Isolation Forest model...")


model = joblib.load(
    MODEL_PATH
)


print(
    "Isolation Forest model loaded successfully"
)



# =====================================================
# LOAD TEST DATA
# =====================================================

print("\nLoading test dataset...")


test_df = pd.read_csv(
    TEST_DATA_PATH
)


print(
    "Test dataset shape:",
    test_df.shape
)



# =====================================================
# REMOVE TARGET COLUMN
# =====================================================

TARGET_COLUMNS = [
    "HasFailure",
    "defect",
    "Defect",
    "label",
    "target",
    "bug"
]


X_test = test_df.copy()



for col in TARGET_COLUMNS:

    if col in X_test.columns:

        print(
            "Removing target column:",
            col
        )

        X_test = X_test.drop(
            columns=[col]
        )



# =====================================================
# MATCH MODEL FEATURES
# =====================================================

if hasattr(model, "feature_names_in_"):

    model_features = list(
        model.feature_names_in_
    )


    extra_features = set(
        X_test.columns
    ) - set(
        model_features
    )


    missing_features = set(
        model_features
    ) - set(
        X_test.columns
    )


    if extra_features:

        print(
            "Removing extra columns:",
            extra_features
        )


    if missing_features:

        raise ValueError(
            f"Missing features:\n{missing_features}"
        )


    X_test = X_test[
        model_features
    ]



print(
    "SHAP input shape:",
    X_test.shape
)



# =====================================================
# CREATE SHAP EXPLAINER
# =====================================================

print("\nCalculating SHAP values...")


explainer = shap.TreeExplainer(
    model
)


shap_values = explainer.shap_values(
    X_test
)


print(
    "SHAP calculation completed"
)


print(
    "SHAP output shape:",
    np.array(shap_values).shape
)



# =====================================================
# HANDLE SHAP OUTPUT
# =====================================================

if isinstance(
    shap_values,
    list
):

    shap_values_used = shap_values[0]


else:

    shap_values_used = shap_values



print(
    "Final SHAP shape:",
    shap_values_used.shape
)
# =====================================================
# 1. SHAP SUMMARY PLOT
# =====================================================

print(
    "\nGenerating SHAP summary plot..."
)


plt.figure(
    figsize=(12, 8)
)


shap.summary_plot(
    shap_values_used,
    X_test,
    show=False
)


plt.title(
    "SHAP Summary Plot - Isolation Forest Anomaly Detection"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "if_shap_summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# 2. SHAP FEATURE IMPORTANCE BAR PLOT
# =====================================================

print(
    "Generating SHAP feature importance plot..."
)


plt.figure(
    figsize=(10, 7)
)


shap.summary_plot(
    shap_values_used,
    X_test,
    plot_type="bar",
    show=False
)


plt.title(
    "SHAP Feature Importance - Isolation Forest"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "if_shap_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# FEATURE RANKING
# =====================================================

print(
    "\nCreating SHAP feature ranking..."
)


mean_shap = np.abs(
    shap_values_used
).mean(
    axis=0
)



ranking = pd.DataFrame(
    {
        "Feature": X_test.columns,
        "Mean_SHAP_Value": mean_shap
    }
)



ranking = ranking.sort_values(
    by="Mean_SHAP_Value",
    ascending=False
)



ranking.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "if_shap_feature_ranking.csv"
    ),
    index=False
)



print(
    "\nTop 10 SHAP Features:"
)


print(
    ranking.head(10)
)



# =====================================================
# 3. SHAP DEPENDENCE PLOT
# =====================================================

top_feature = ranking.iloc[0]["Feature"]


print(
    "\nGenerating dependence plot:",
    top_feature
)



shap.dependence_plot(
    top_feature,
    shap_values_used,
    X_test,
    show=False
)



plt.title(
    f"SHAP Dependence Plot - {top_feature}"
)



plt.tight_layout()



plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "if_shap_dependence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)



plt.close()



# =====================================================
# 4. SHAP WATERFALL PLOT
# =====================================================

print(
    "\nGenerating waterfall plot..."
)


try:

    explanation = explainer(
        X_test
    )


    shap.plots.waterfall(
        explanation[0],
        max_display=15,
        show=False
    )


except Exception as e:

    print(
        "Waterfall fallback:",
        e
    )


    expected_value = explainer.expected_value


    if isinstance(
        expected_value,
        np.ndarray
    ):

        expected_value = expected_value.item()



    shap.plots._waterfall.waterfall_legacy(
        expected_value,
        shap_values_used[0],
        X_test.iloc[0],
        max_display=15
    )



plt.title(
    "SHAP Waterfall Plot - Isolation Forest Anomaly"
)



plt.tight_layout()



plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "if_shap_waterfall.png"
    ),
    dpi=300,
    bbox_inches="tight"
)



plt.close()



# =====================================================
# COMPLETE
# =====================================================

print("\n====================================")
print("ISOLATION FOREST SHAP COMPLETED")
print("====================================")


print("\nGenerated files:")


for file in os.listdir(
    OUTPUT_DIR
):

    print(
        " -",
        file
    )
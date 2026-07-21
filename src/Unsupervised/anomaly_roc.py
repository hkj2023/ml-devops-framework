"""
ROC Curve Comparison - Unsupervised Anomaly Detection Models

Models:
    - DBSCAN
    - One-Class SVM
    - Isolation Forest
    - Local Outlier Factor

Based on reported AUROC values.

Output:
    outputs/anomaly_roc_comparison.png
"""


import os
import numpy as np
import matplotlib.pyplot as plt



# =====================================================
# OUTPUT PATH
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../"
    )
)


OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "anomaly_roc_comparison.png"
)



# =====================================================
# MODEL AUROC RESULTS
# =====================================================

model_auc = {

    "DBSCAN": None,

    "One-Class SVM": 0.905328,

    "Isolation Forest": 0.934239,

    "Local Outlier Factor": 0.896725

}



# =====================================================
# GENERATE APPROXIMATE ROC CURVES
# =====================================================

plt.figure(
    figsize=(10,8)
)



for model, auc in model_auc.items():


    # DBSCAN has no AUROC
    if auc is None:

        continue


    # Generate illustrative ROC points
    # Higher AUROC produces better curve

    fpr = np.linspace(
        0,
        1,
        100
    )


    tpr = fpr ** (
        (1/auc)-1
    )


    # Highlight Isolation Forest

    if model == "Isolation Forest":

        plt.plot(
            fpr,
            tpr,
            linewidth=4,
            label=f"{model} (AUROC={auc:.4f})"
        )

    else:

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model} (AUROC={auc:.4f})"
        )



# Random classifier

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    label="Random Guess"
)



# =====================================================
# GRAPH FORMAT
# =====================================================

plt.xlabel(
    "False Positive Rate (FPR)"
)


plt.ylabel(
    "True Positive Rate (Detection Rate)"
)



plt.title(
    "ROC Curve Comparison of Unsupervised Anomaly Detection Models"
)



plt.legend(
    loc="lower right"
)



plt.grid()



plt.tight_layout()



plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)



plt.close()



print(
    "ROC comparison saved:"
)

print(
    OUTPUT_PATH
)
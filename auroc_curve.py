import numpy as np
import matplotlib.pyplot as plt

# Example: ground truth labels and predicted probabilities
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])   # replace with your test labels
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.3, 0.7])  # predicted probabilities

# Sort thresholds
thresholds = np.sort(np.unique(y_scores))[::-1]

tpr_list = []
fpr_list = []

for thresh in thresholds:
    y_pred = (y_scores >= thresh).astype(int)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    tpr_list.append(TPR)
    fpr_list.append(FPR)

# Convert to arrays
tpr = np.array(tpr_list)
fpr = np.array(fpr_list)

# Numerical integration (trapezoidal rule)
auc = np.trapz(tpr, fpr)
print("AUC (computed by formula):", auc)

# Plot ROC curve
plt.plot(fpr, tpr, marker='o', label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (from formula)")
plt.legend()
plt.show()

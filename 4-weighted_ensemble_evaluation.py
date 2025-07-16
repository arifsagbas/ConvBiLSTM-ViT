import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# === Ayarlar ===
root_dir = r"\SLEEP"
result_dir = os.path.join(root_dir, "results_ensemble_multibranch")

# Bireysel model başarılarına göre normalize edilmiş ağırlıklar
weights = {
    "mel": 0.3394,
    "mfcc": 0.3300,
    "cqt": 0.3306
}

# === y_prob ve y_true dosyalarını yükle ===
probs = {}
y_true = None
for branch in ["mel", "mfcc", "cqt"]:
    prob_path = os.path.join(result_dir, f"y_prob_{branch}.npy")
    probs[branch] = np.load(prob_path)

    if y_true is None:
        y_true = np.load(os.path.join(result_dir, f"y_true_{branch}.npy"))

# === Ensemble softmax birleşimi ===
ensemble_prob = sum(weights[b] * probs[b] for b in probs)
y_pred = np.argmax(ensemble_prob, axis=1)

# === Raporlama ===
report = classification_report(y_true, y_pred, digits=4)
print(report)

with open(os.path.join(result_dir, "report_weighted_ensemble.txt"), "w") as f:
    f.write(report)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Weighted Ensemble")
plt.savefig(os.path.join(result_dir, "cm_weighted_ensemble.png"))
plt.close()

# === ROC Curve (macro average) ===
try:
    from sklearn.preprocessing import label_binarize
    n_classes = len(np.unique(y_true))
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], ensemble_prob[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], ensemble_prob[:, i])

    # Makro ortalama ROC
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve - Weighted Ensemble")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "roc_weighted_ensemble.png"))
    plt.close()
except Exception as e:
    print("ROC çizimi sırasında hata:", e)

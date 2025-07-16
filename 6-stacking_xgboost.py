import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# === Ayarlar ===
result_dir = r"\SLEEP\results_ensemble_multibranch"
branches = ["mel", "mfcc", "cqt"]
n_classes = 7

# === Tüm y_prob'ları yükle ve birleştir ===
X_meta = []
for branch in branches:
    prob = np.load(os.path.join(result_dir, f"y_prob_{branch}.npy"))
    X_meta.append(prob)
X_meta = np.concatenate(X_meta, axis=1)  # (N, 21)

# === Ortak y_true'yu yükle ===
y_true = np.load(os.path.join(result_dir, f"y_true_mel.npy"))  # hepsi aynı

# === Meta-learner: XGBoost ===
meta_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss')
meta_model.fit(X_meta, y_true)
y_pred = meta_model.predict(X_meta)
y_prob = meta_model.predict_proba(X_meta)

# === Değerlendirme ===
print("=== STACKING RESULTS (XGBoost) ===")
report = classification_report(y_true, y_pred, digits=4)
print(report)

with open(os.path.join(result_dir, "report_stacking_xgboost.txt"), "w") as f:
    f.write(report)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Confusion Matrix - Stacking (XGBoost)")
plt.savefig(os.path.join(result_dir, "cm_stacking_xgboost.png"))
plt.close()

# === ROC Curve (Macro Average) ===
y_true_bin = label_binarize(y_true, classes=range(n_classes))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])

plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC={roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Stacking (XGBoost)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(result_dir, "roc_stacking_xgboost.png"))
plt.close()

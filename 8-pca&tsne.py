import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Yollar
base_path = r"\SLEEP\results_ensemble_multibranch"
mfcc = np.load(base_path + r"\y_prob_mfcc.npy")
mel = np.load(base_path + r"\y_prob_mel.npy")
cqt = np.load(base_path + r"\y_prob_cqt.npy")
y_true = np.load(base_path + r"\y_true.npy")

# Softmax çıktılarının birleştirilmesi (3 x 7 = 21 boyut)
X = np.hstack([mfcc, mel, cqt])

# Normalize et (PCA ve t-SNE için önerilir)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Etiket isimleri
labels = ['Cough', 'Laugh', 'Scream', 'Sneeze', 'Snore', 'Sniffle', 'Farting']
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(len(labels)):
    plt.scatter(X_pca[y_true == i, 0], X_pca[y_true == i, 1], label=labels[i], alpha=0.7, s=40, c=colors[i])
plt.title("PCA Projection of Stacked Softmax Vectors")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(base_path + r"\pca_scatter.png")
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(len(labels)):
    plt.scatter(X_tsne[y_true == i, 0], X_tsne[y_true == i, 1], label=labels[i], alpha=0.7, s=40, c=colors[i])
plt.title("t-SNE Projection of Stacked Softmax Vectors")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend()
plt.tight_layout()
plt.savefig(base_path + r"\tsne_scatter.png")
plt.show()

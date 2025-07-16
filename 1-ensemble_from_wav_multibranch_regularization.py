# Başlık: Ensemble CNN + BiLSTM + Attention + ViT (Mel, MFCC, CQT) with SpecAugment


import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Yol ayarları
wav_dir = r"\SLEEP\nocturnal_wav"
out_root = r"\SLEEP"
os.makedirs(out_root, exist_ok=True)

# Çıktı klasörleri
out_dirs = {
    "mel": os.path.join(out_root, "mel"),
    "mfcc": os.path.join(out_root, "mfcc"),
    "cqt": os.path.join(out_root, "cqt"),
}
for path in out_dirs.values():
    os.makedirs(path, exist_ok=True)

# Öznitelik çıkarım fonksiyonu
def save_spectrogram(y, sr, path, spec_type="mel"):
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis("off")
    if spec_type == "mel":
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    elif spec_type == "mfcc":
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        librosa.display.specshow(mfcc, sr=sr, x_axis=None, y_axis=None)
    elif spec_type == "cqt":
        C = librosa.cqt(y=y, sr=sr)
        C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        librosa.display.specshow(C_dB, sr=sr, x_axis=None, y_axis=None)
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# WAV dosyalarını işle
for fname in os.listdir(wav_dir):
    if not fname.endswith(".wav"):
        continue
    try:
        wav_path = os.path.join(wav_dir, fname)
        y, sr = librosa.load(wav_path, sr=48000)
        for spec_type in ["mel", "mfcc", "cqt"]:
            out_path = os.path.join(out_dirs[spec_type], fname.replace(".wav", ".png"))
            save_spectrogram(y, sr, out_path, spec_type)
    except Exception as e:
        print(f"Hata: {fname} → {e}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import random
import numpy as np
import os
from PIL import Image
from timm import create_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Deterministik Sabitleme ====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== SpecAugment ====
def spec_augment(img_tensor, time_mask_width=20, freq_mask_width=20):
    img = img_tensor.clone()
    c, h, w = img.shape
    t = random.randint(0, time_mask_width)
    t0 = random.randint(0, max(0, w - t))
    img[:, :, t0:t0 + t] = 0
    f = random.randint(0, freq_mask_width)
    f0 = random.randint(0, max(0, h - f))
    img[:, f0:f0 + f, :] = 0
    return img

# ==== Dataset Sınıfı ====
class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, specaug=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.specaug = specaug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform: img = self.transform(img)
        if self.specaug: img = spec_augment(img)
        return img, label

# ==== Görsel Dönüşüm ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==== Attention Katmanı ====
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: [B, T, F]
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)

# ==== Hybrid Model: CNN + BiLSTM + Attention + ViT ====
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.bilstm = nn.LSTM(64 * 56, 128, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(0.5)
        self.attn = Attention(256)
        self.vit = create_model("vit_base_patch16_224", pretrained=True, num_classes=256)
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        cnn_feat = self.cnn(x)                          # [B, 64, 56, 56]
        seq = cnn_feat.permute(0, 3, 1, 2)              # [B, 56, 64, 56]
        seq = seq.flatten(2)                            # [B, 56, 64*56]
        lstm_out, _ = self.bilstm(seq)                  # [B, 56, 256]
        lstm_out = self.dropout_lstm(lstm_out)          # Apply dropout
        attn_out = self.attn(lstm_out)                  # [B, 256]
        vit_feat = self.vit(x)                          # [B, 256]
        fused = torch.cat([attn_out, vit_feat], dim=1)  # [B, 512]
        return self.classifier(fused)


# ==== Eğitim Parametreleri ====
epochs = 30
patience = 30
batch_size = 16
root_dir = r"C:\Users\Ensar\Desktop\SLEEP"
result_dir = os.path.join(root_dir, "results_ensemble_multibranch")
os.makedirs(result_dir, exist_ok=True)

branches = ["mel", "mfcc", "cqt"]
branch_models = {}
branch_preds = {}
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# === Fold Başlat ===
image_dict = {}
label_dict = {}
for b in branches:
    folder = os.path.join(root_dir, b)
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    labels = [int(os.path.basename(p)[0]) for p in paths]
    image_dict[b] = paths
    label_dict[b] = labels

all_true, all_pred, all_prob = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(image_dict["mel"], label_dict["mel"]), 1):
    print(f"\n🔁 Fold {fold}/10")
    preds_softmax = []
    for branch in branches:
        print(f"🌿 Training on: {branch}")
        x_paths = image_dict[branch]
        y_labels = label_dict[branch]
        model = HybridModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_ds = SpectrogramDataset([x_paths[i] for i in train_idx], [y_labels[i] for i in train_idx], transform, True)
        test_ds = SpectrogramDataset([x_paths[i] for i in test_idx], [y_labels[i] for i in test_idx], transform, False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        best_acc, counter = 0, 0
        for epoch in range(epochs):
            model.train()
            correct, total = 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total += y.size(0)
                correct += (out.argmax(1) == y).sum().item()
            acc = correct / total
            print(f"  Epoch {epoch+1:02d}: acc={acc:.4f}")

            # Validation
            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    correct += (out.argmax(1) == y).sum().item()
            val_acc = correct / len(test_loader.dataset)
            if val_acc > best_acc:
                best_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), os.path.join(result_dir, f"best_{branch}_fold{fold}.pth"))
            else:
                counter += 1
                if counter >= patience:
                    print("  ⏹️ Early stopping.")
                    break

        # Predict & Softmax
        model.load_state_dict(torch.load(os.path.join(result_dir, f"best_{branch}_fold{fold}.pth")))
        model.eval()
        probs = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = model(x)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                probs.extend(prob)
        preds_softmax.append(np.array(probs))

    # ==== Ensemble Softmax ====
    avg_probs = np.mean(preds_softmax, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    true_labels = [label_dict["mel"][i] for i in test_idx]

    all_true.extend(true_labels)
    all_pred.extend(final_preds)
    all_prob.extend(avg_probs)

# ==== Kaydet ====
np.save(os.path.join(result_dir, "y_true.npy"), np.array(all_true))
np.save(os.path.join(result_dir, "y_pred.npy"), np.array(all_pred))
np.save(os.path.join(result_dir, "y_prob.npy"), np.array(all_prob))

# ==== Rapor ve Grafikler ====
cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(result_dir, "conf_matrix.png"))

y_true_bin = label_binarize(all_true, classes=list(range(7)))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(7):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(all_prob)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure()
for i in range(7):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC={roc_auc[i]:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(result_dir, "roc_curve.png"))

report = classification_report(all_true, all_pred, digits=4)
with open(os.path.join(result_dir, "report.txt"), "w") as f:
    f.write(report)
print("\n✅ Ensemble 10-fold CV tamamlandı.")


import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from timm import create_model
from sklearn.model_selection import StratifiedKFold

# === Ayarlar ===
root_dir = r"\SLEEP"
result_dir = os.path.join(root_dir, "results_ensemble_multibranch")
branches = ["mel", "mfcc", "cqt"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transform ve Dataset ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = transform(img)
        return img, self.labels[idx]

# === Model Tanımı ===
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.bilstm = nn.LSTM(64 * 56, 128, batch_first=True, bidirectional=True)
        class Attention(nn.Module):
            def __init__(self, dim): super().__init__(); self.attn = nn.Linear(dim, 1)
            def forward(self, x): w = torch.softmax(self.attn(x), dim=1); return torch.sum(x * w, dim=1)
        self.attn = Attention(256)
        self.vit = create_model("vit_base_patch16_224", pretrained=True, num_classes=256)
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        cnn_feat = self.cnn(x)
        seq = cnn_feat.permute(0, 3, 1, 2).flatten(2)
        lstm_out, _ = self.bilstm(seq)
        attn_out = self.attn(lstm_out)
        vit_feat = self.vit(x)
        fused = torch.cat([attn_out, vit_feat], dim=1)
        return self.classifier(fused)

# === Fold Başına Test ===
for branch in branches:
    print(f"\n📊 Değerlendiriliyor: {branch.upper()}")
    folder = os.path.join(root_dir, branch)
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    labels = [int(os.path.basename(p)[0]) for p in paths]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels), 1):
        model = HybridModel().to(device)
        fold_model_path = os.path.join(result_dir, f"best_{branch}_fold{fold}.pth")
        if not os.path.exists(fold_model_path):
            print(f"❌ {fold_model_path} bulunamadı, atlanıyor.")
            continue
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()

        test_paths = [paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        loader = DataLoader(SimpleDataset(test_paths, test_labels), batch_size=16)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                out = model(x)
                pred = out.argmax(1).cpu().numpy()
                all_pred.extend(pred)
                all_true.extend(y.numpy())

    # === Rapor ve Görsel ===
    report = classification_report(all_true, all_pred, digits=4)
    with open(os.path.join(result_dir, f"report_{branch}_10fold.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {branch.upper()} (10-Fold)")
    plt.savefig(os.path.join(result_dir, f"cm_{branch}_10fold.png"))
    plt.close()
    print(report)

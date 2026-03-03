import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# ===== 基本配置 =====
DATA_DIR = r"D:\desktop\3DUNET\dataset_luna"
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# ===== Dataset =====
class LunaDataset(Dataset):
    def __init__(self, pos_npz, neg_npz):
        pos = np.load(pos_npz)["patches"]
        neg = np.load(neg_npz)["patches"]

        self.data = np.concatenate([pos, neg], axis=0)
        self.labels = np.concatenate([
            np.ones(len(pos), dtype=np.int64),
            np.zeros(len(neg), dtype=np.int64)
        ])

        # 简单归一化到 [-1,1]
        self.data = np.clip(self.data, -1024, 1024) / 1024.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx][None, ...]  # (1,64,64,64)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)

# ===== 加载数据 =====
dataset = LunaDataset(
    os.path.join(DATA_DIR, "pos_patches.npz"),
    os.path.join(DATA_DIR, "neg_patches.npz")
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== 最小 3D CNN =====
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = Simple3DCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===== 训练 =====
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

print("Training finished.")
torch.save(model.state_dict(), "baseline_3dcnn.pth")


import numpy as np, os, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images = np.load(os.path.join(DATA_DIR, "images.npy"))
masks  = np.load(os.path.join(DATA_DIR, "masks.npy"))

# 取一个前景最多的样本，更容易过拟合
fg = masks.sum(axis=(1,2,3,4))
idx = int(fg.argmax())
x0 = torch.tensor(images[idx:idx+1], dtype=torch.float32).to(DEVICE)
y0 = torch.tensor(masks[idx:idx+1], dtype=torch.float32).to(DEVICE)
print("overfit idx:", idx, "fg vox:", int(fg[idx]))

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).view(1, -1)
    targets = targets.view(1, -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()

class UNet3DLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.up = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv3d(16, 8, 3, padding=1), nn.ReLU(),
            nn.Conv3d(8, 1, 1)
        )
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        return self.dec(torch.cat([u, e1], dim=1))

model = UNet3DLite().to(DEVICE)
bce = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, 301):
    opt.zero_grad()
    logits = model(x0)
    loss = 0.7*bce(logits, y0) + 0.3*dice_loss_from_logits(logits, y0)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).float()
            inter = (pred * y0).sum().item()
            denom = pred.sum().item() + y0.sum().item()
            dice = (2*inter + 1e-6) / (denom + 1e-6)
        print(f"step {step} loss {loss.item():.4f} dice {dice:.4f}")

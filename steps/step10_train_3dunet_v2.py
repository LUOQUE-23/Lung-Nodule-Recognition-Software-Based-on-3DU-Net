import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
BATCH_SIZE = 1
EPOCHS = 40
LR = 1e-3
VAL_RATIO = 0.15
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

class SegDataset(Dataset):
    def __init__(self, images, masks, indices, augment=False):
        self.images = images
        self.masks = masks
        self.indices = indices
        self.augment = augment
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = torch.tensor(self.images[i], dtype=torch.float32)
        y = torch.tensor(self.masks[i], dtype=torch.float32)
        if self.augment:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[3])
                y = torch.flip(y, dims=[3])
        return x, y
def dice_coeff_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    return (2 * inter + eps) / (denom + eps)
def dice_coeff_binary(pred_bin, gt_bin, eps=1e-6):
    pred = pred_bin.view(pred_bin.size(0), -1)
    gt = gt_bin.view(gt_bin.size(0), -1)
    inter = (pred * gt).sum(dim=1)
    denom = pred.sum(dim=1) + gt.sum(dim=1)
    return (2 * inter + eps) / (denom + eps)
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.out = nn.Conv3d(16, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)
images = np.load(os.path.join(DATA_DIR, "images.npy"))
masks = np.load(os.path.join(DATA_DIR, "masks.npy"))
n = len(images)
rng = np.random.default_rng(SEED)
indices = np.arange(n)
rng.shuffle(indices)
split = int(n * (1 - VAL_RATIO))
train_idx = indices[:split]
val_idx = indices[split:]
print("Samples:", n, "Train:", len(train_idx), "Val:", len(val_idx))
train_ds = SegDataset(images, masks, train_idx, augment=True)
val_ds = SegDataset(images, masks, val_idx, augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
train_masks = masks[train_idx]
pos = float(train_masks.sum())
neg = float(train_masks.size - pos)
pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
print("pos_weight:", float(pos_weight.item()))
model = UNet3D().to(DEVICE)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=3, min_lr=1e-5
)
best_dice = -1.0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        dice_loss = 1.0 - dice_coeff_from_logits(logits, y).mean()
        loss = 0.5 * bce(logits, y) + 0.5 * dice_loss
        loss.backward()
        opt.step()
        train_loss += loss.item()
    train_loss /= max(1, len(train_loader))
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            dice_loss = 1.0 - dice_coeff_from_logits(logits, y).mean()
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss
            val_loss += loss.item()
            pred = (torch.sigmoid(logits) > 0.5).float()
            val_dice += dice_coeff_binary(pred, y).mean().item()
    val_loss /= max(1, len(val_loader))
    val_dice /= max(1, len(val_loader))
    scheduler.step(val_loss)
    lr = opt.param_groups[0]["lr"]
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"val_dice={val_dice:.4f} lr={lr:.2e}"
    )
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "best_model_unet3d.pth")
torch.save(model.state_dict(), "unet3d_v3.pth")
print("Training finished. Best val dice:", best_dice)

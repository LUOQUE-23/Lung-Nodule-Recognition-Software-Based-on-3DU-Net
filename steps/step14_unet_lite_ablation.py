import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 统一阈值（用于评估 top20）
AUTO_THRESH = True
THRESH = 0.1
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]

# 训练配置
EPOCHS = 60
LR = 1e-4
BATCH = 1
EVAL_EVERY = 5  # 每隔多少轮评估并保存 best

# 你需要确保这个权重文件存在且结构匹配（UNet3D）
PRETRAIN_PATH = r"best_model_unet3d.pth"

# 全局 best 指标
best_dice = 0.0


class SegDataset(Dataset):
    def __init__(self, img_path, mask_path):
        self.images = np.load(img_path)
        self.masks = np.load(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx], dtype=torch.float32)
        y = torch.tensor(self.masks[idx], dtype=torch.float32)
        return x, y


loader = DataLoader(
    SegDataset(os.path.join(DATA_DIR, "images.npy"), os.path.join(DATA_DIR, "masks.npy")),
    batch_size=BATCH,
    shuffle=True
)


def dice_coeff(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return (2 * inter + eps) / (denom + eps)


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


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
        return self.out(d1)  # logits


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def train_and_eval():
    """
    使用新训练模型作为底座，冻结 encoder+瓶颈，只微调 decoder
    """
    global best_dice

    # ---------- build model ----------
    model = UNet3D().to(DEVICE)

    # ---------- load pretrained ----------
    if not os.path.exists(PRETRAIN_PATH):
        raise FileNotFoundError(f"Pretrained weights not found: {PRETRAIN_PATH}")

    ckpt = torch.load(PRETRAIN_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)

    # ---------- freeze strategy ----------
    for p in model.enc1.parameters():
        p.requires_grad = True
    for p in model.enc2.parameters():
        p.requires_grad = True
    for p in model.bottleneck.parameters():
        p.requires_grad = True

    # ---------- loss / optimizer ----------
    masks = np.load(os.path.join(DATA_DIR, "masks.npy"))
    pos = float(masks.sum())
    neg = float(masks.size - pos)
    pos_weight_val = min(50.0, neg / max(pos, 1.0))
    pos_weight = torch.tensor([pos_weight_val], device=DEVICE)
    print("pos_weight:", float(pos_weight.item()))

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # ---------- prepare eval subset (Top20) ----------
    images = np.load(os.path.join(DATA_DIR, "images.npy"))
    masks = np.load(os.path.join(DATA_DIR, "masks.npy"))
    fg = masks.sum(axis=(1, 2, 3, 4))
    topk = np.argsort(-fg)[:20]

    def eval_top20_mean_dice():
        model.eval()
        ds = []
        soft_ds = []
        mean_probs = []
        with torch.no_grad():
            for j in topk:
                xx = torch.tensor(images[j:j+1], dtype=torch.float32).to(DEVICE)
                gg = torch.tensor(masks[j:j+1], dtype=torch.float32).to(DEVICE)
                prob = torch.sigmoid(model(xx))
                soft_ds.append(dice_coeff(prob, gg).item())
                mean_probs.append(float(prob.mean().item()))
                pred = (prob > THRESH).float()
                ds.append(dice_coeff(pred, gg).item())
        model.train()
        return float(np.mean(ds)), float(np.mean(soft_ds)), float(np.mean(mean_probs))

    # ---------- train loop ----------
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss_from_logits(logits, y)
            loss.backward()
            opt.step()
            tot += loss.detach().item()

        if (ep + 1) % EVAL_EVERY == 0:
            if AUTO_THRESH:
                mean_dices = []
                with torch.no_grad():
                    probs_list = []
                    gts_list = []
                    for j in topk:
                        xx = torch.tensor(images[j:j+1], dtype=torch.float32).to(DEVICE)
                        gg = torch.tensor(masks[j:j+1], dtype=torch.float32).to(DEVICE)
                        probs_list.append(torch.sigmoid(model(xx)))
                        gts_list.append(gg)
                    for t in THRESHOLDS:
                        ds = []
                        for prob, gg in zip(probs_list, gts_list):
                            pred = (prob > t).float()
                            ds.append(dice_coeff(pred, gg).item())
                        mean_dices.append(float(np.mean(ds)))
                best_idx = int(np.argmax(mean_dices))
                THRESH = THRESHOLDS[best_idx]
            mean_dice, soft_dice, mean_prob = eval_top20_mean_dice()
            avg_loss = tot / len(loader)
            print(
                f"  epoch {ep+1}/{EPOCHS} loss {avg_loss:.4f} "
                f"top20_dice@{THRESH:.2f} {mean_dice:.4f} "
                f"soft_dice {soft_dice:.4f} mean_prob {mean_prob:.4f}"
            )

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), "best_model.pth")
                print(f"    -> saved best_model.pth (best_dice={best_dice:.4f})")

    # ---------- speed test (use current model) ----------
    xx = torch.tensor(images[topk[0]:topk[0] + 1], dtype=torch.float32).to(DEVICE)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = model(xx)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / 20

    return {
        "params": count_params(model),
        "best_top20_mean_dice": best_dice,
        "forward_ms": ms
    }


if __name__ == "__main__":
    print("\n=== Training config: UNet3D (freeze encoder) ===")
    results = [train_and_eval()]

    print("\n=== Summary (THRESH=%.2f) ===" % THRESH)
    for r in results:
        print(r)
    print("\nSaved best model to: best_model.pth")

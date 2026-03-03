import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ----------------- config -----------------
DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
LOG_PATH = r"D:\desktop\3DUNET\EXPERIMENT_LOG.md"
OUT_DIR = r"D:\desktop\3DUNET\seg_outputs_thresh"

BATCH_SIZE = 1
EPOCHS = 40
LR = 1e-3
VAL_RATIO = 0.15
SEED = 0

TOPK = 20
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
FIXED_THRESH = 0.30
UNCERTAINTY_SUPPRESS = True
U_Q_LOW = 0.50
U_Q_HIGH = 0.90
EPS = 1e-6

BASE_CH = 4
SMALL_CH = 2

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


class UNet3DLite(nn.Module):
    def __init__(self, base_ch=8):
        super().__init__()
        c1 = base_ch
        c2 = base_ch * 2
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, c1, 3, padding=1), nn.ReLU(),
            nn.Conv3d(c1, c1, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(c1, c2, 3, padding=1), nn.ReLU(),
            nn.Conv3d(c2, c2, 3, padding=1), nn.ReLU()
        )
        self.up = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv3d(c1 + c1, c1, 3, padding=1), nn.ReLU(),
            nn.Conv3d(c1, 1, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        return self.dec(torch.cat([u, e1], dim=1))


def dice_coeff_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    return (2 * inter + eps) / (denom + eps)


def entropy_from_prob(prob):
    return -(prob * torch.log(prob + EPS) + (1 - prob) * torch.log(1 - prob + EPS))


def compute_u_params(u_list):
    if not u_list:
        return None, None
    u_all = torch.cat([u.view(-1) for u in u_list], dim=0)
    u0 = float(torch.quantile(u_all, U_Q_LOW).item())
    u1 = float(torch.quantile(u_all, U_Q_HIGH).item())
    return u0, u1


def suppress_prob(prob, u, u0, u1):
    if u0 is None or u1 is None or u1 <= u0:
        return prob
    g = (u - u0) / (u1 - u0)
    g = torch.clamp(g, 0.0, 1.0)
    return prob * (1.0 - g)


def seg_metrics(pred_bin, gt_bin, eps=1e-6):
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def prepare_loaders(images, masks):
    n = len(images)
    rng = np.random.default_rng(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(n * (1 - VAL_RATIO))
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_ds = SegDataset(images, masks, train_idx, augment=True)
    val_ds = SegDataset(images, masks, val_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    train_masks = masks[train_idx]
    pos = float(train_masks.sum())
    neg = float(train_masks.size - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    return train_loader, val_loader, pos_weight


def train_model(model, train_loader, val_loader, pos_weight, loss_mode):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    best_dice = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            if loss_mode == "bce_only":
                loss = bce(logits, y)
            else:
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
                if loss_mode == "bce_only":
                    loss = bce(logits, y)
                else:
                    dice_loss = 1.0 - dice_coeff_from_logits(logits, y).mean()
                    loss = 0.5 * bce(logits, y) + 0.5 * dice_loss
                val_loss += loss.item()
                val_dice += dice_coeff_from_logits(logits, y).mean().item()

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
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_state, best_dice


def eval_with_thresholds(model, images, masks, topk_idx):
    model.eval()
    with torch.no_grad():
        probs_list = []
        gts_list = []
        u_list = []
        for j in topk_idx:
            xx = torch.tensor(images[j:j+1], dtype=torch.float32).to(DEVICE)
            gg = torch.tensor(masks[j:j+1], dtype=torch.float32).to(DEVICE)
            prob = torch.sigmoid(model(xx))
            probs_list.append(prob)
            gts_list.append(gg)
            if UNCERTAINTY_SUPPRESS:
                u_list.append(entropy_from_prob(prob))

        if UNCERTAINTY_SUPPRESS:
            u0, u1 = compute_u_params(u_list)
            probs_list = [
                suppress_prob(prob, u, u0, u1)
                for prob, u in zip(probs_list, u_list)
            ]

        all_metrics = []
        for t in THRESHOLDS:
            metrics_list = []
            for prob, gg in zip(probs_list, gts_list):
                pred = (prob > t).float()
                metrics_list.append(seg_metrics(pred, gg))
            mean_metrics = {
                k: float(np.mean([m[k].item() for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            all_metrics.append((t, mean_metrics))

    best = max(all_metrics, key=lambda x: x[1]["dice"])
    return best[0], best[1]


def eval_fixed_threshold(model, images, masks, topk_idx, thresh):
    model.eval()
    with torch.no_grad():
        probs_list = []
        gts_list = []
        u_list = []
        metrics_list = []
        for j in topk_idx:
            xx = torch.tensor(images[j:j+1], dtype=torch.float32).to(DEVICE)
            gg = torch.tensor(masks[j:j+1], dtype=torch.float32).to(DEVICE)
            prob = torch.sigmoid(model(xx))
            probs_list.append(prob)
            gts_list.append(gg)
            if UNCERTAINTY_SUPPRESS:
                u_list.append(entropy_from_prob(prob))

        if UNCERTAINTY_SUPPRESS:
            u0, u1 = compute_u_params(u_list)
            probs_list = [
                suppress_prob(prob, u, u0, u1)
                for prob, u in zip(probs_list, u_list)
            ]

        for prob, gg in zip(probs_list, gts_list):
            pred = (prob > thresh).float()
            metrics_list.append(seg_metrics(pred, gg))
        mean_metrics = {
            k: float(np.mean([m[k].item() for m in metrics_list]))
            for k in metrics_list[0].keys()
        }
    return mean_metrics


def save_figure(rows, out_dir, fname, title):
    os.makedirs(out_dir, exist_ok=True)
    labels = [r["name"] for r in rows]
    metrics = ["dice", "iou", "precision", "recall", "f1"]
    metric_labels = ["Dice", "IoU", "Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    width = 0.15
    plt.figure(figsize=(8, 4))
    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        vals = [r[m] for r in rows]
        plt.bar(x + (i - 2) * width, vals, width=width, label=ml)
        for xi, v in zip(x + (i - 2) * width, vals):
            plt.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, labels)
    max_val = max([r[m] for r in rows for m in metrics]) if rows else 1.0
    plt.ylim(0, max_val * 1.2 if max_val > 0 else 1.0)
    plt.ylabel("TopK mean")
    plt.title(title)
    plt.legend(ncol=3, fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, fname)
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def append_log(section_name, rows, topk_count, fig_path, notes):
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"\n## Ablation: {section_name}\n")
    lines.append(f"### {ts}\n")
    lines.append(f"- Dataset: dataset_luna_seg (Top{topk_count} fg)\n")
    lines.append(f"- Thresholds: {THRESHOLDS}\n")
    lines.append("- Metric: TopK mean Dice/IoU/Precision/Recall/F1\n")
    lines.append(f"- Figure: {fig_path}\n")
    lines.append(f"- Notes: {notes}\n\n")
    lines.append("| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |\n")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['weights']} | {r['best_thr']:.2f} | "
            f"{r['dice']:.4f} | {r['iou']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} |\n"
        )
    Path(LOG_PATH).write_text(
        Path(LOG_PATH).read_text(encoding="utf-8") + "".join(lines),
        encoding="utf-8",
    )


# ----------------- load data -----------------
images = np.load(os.path.join(DATA_DIR, "images.npy"))
masks = np.load(os.path.join(DATA_DIR, "masks.npy"))

fg = masks.sum(axis=(1, 2, 3, 4))
if float(fg.max()) <= 0:
    raise SystemExit("No foreground found in masks.npy (fg max <= 0).")

ranked = np.argsort(-fg)
ranked = ranked[fg[ranked] > 0]
if len(ranked) == 0:
    raise SystemExit("No fg>0 samples found after filtering.")

k = min(TOPK, len(ranked))
topk_idx = ranked[:k]
print(f"Using Top{k} fg samples for evaluation.")

# ----------------- baseline (pretrained) -----------------
baseline_weights = r"D:\desktop\3DUNET\ablation_unet3d_lite_ch4.pth"
baseline_model = UNet3DLite(base_ch=BASE_CH).to(DEVICE)
if not os.path.exists(baseline_weights):
    raise SystemExit(f"Baseline weights not found: {baseline_weights}")
baseline_model.load_state_dict(torch.load(baseline_weights, map_location=DEVICE))

best_thr, baseline_metrics = eval_with_thresholds(
    baseline_model, images, masks, topk_idx
)
baseline_row = {
    "name": "Baseline",
    "weights": os.path.basename(baseline_weights),
    "best_thr": best_thr,
    **baseline_metrics,
}

# ----------------- Ablation 1: threshold strategy -----------------
fixed_metrics = eval_fixed_threshold(
    baseline_model, images, masks, topk_idx, FIXED_THRESH
)
rows = [
    baseline_row,
    {
        "name": f"FixedThr{FIXED_THRESH:.2f}",
        "weights": os.path.basename(baseline_weights),
        "best_thr": FIXED_THRESH,
        **fixed_metrics,
    },
]
fig_path = save_figure(
    rows, OUT_DIR, "ablation_threshold.png", "Ablation: Threshold Strategy"
)
append_log(
    "Threshold Strategy",
    rows,
    len(topk_idx),
    fig_path,
    f"Baseline uses best Dice threshold; ablation uses fixed {FIXED_THRESH:.2f}.",
)
print("Ablation 1 done.")

# ----------------- Ablation 2: channel width -----------------
train_loader, val_loader, pos_weight = prepare_loaders(images, masks)
small_model = UNet3DLite(base_ch=SMALL_CH).to(DEVICE)
best_state, best_dice = train_model(
    small_model, train_loader, val_loader, pos_weight, loss_mode="mix"
)
small_weights = r"D:\desktop\3DUNET\ablation_unet3d_lite_ch2.pth"
if best_state is not None:
    torch.save(best_state, small_weights)
small_model.load_state_dict(torch.load(small_weights, map_location=DEVICE))
small_thr, small_metrics = eval_with_thresholds(
    small_model, images, masks, topk_idx
)
rows = [
    baseline_row,
    {
        "name": "UNet3DLite-Ch2",
        "weights": os.path.basename(small_weights),
        "best_thr": small_thr,
        **small_metrics,
    },
]
fig_path = save_figure(
    rows, OUT_DIR, "ablation_channels.png", "Ablation: Channel Width"
)
append_log(
    "Channel Width",
    rows,
    len(topk_idx),
    fig_path,
    "Baseline uses base_ch=4; ablation uses base_ch=2.",
)
print("Ablation 2 done.")

# ----------------- Ablation 3: loss function -----------------
train_loader, val_loader, pos_weight = prepare_loaders(images, masks)
bce_model = UNet3DLite(base_ch=BASE_CH).to(DEVICE)
best_state, best_dice = train_model(
    bce_model, train_loader, val_loader, pos_weight, loss_mode="bce_only"
)
bce_weights = r"D:\desktop\3DUNET\ablation_unet3d_lite_ch4_bce_only.pth"
if best_state is not None:
    torch.save(best_state, bce_weights)
bce_model.load_state_dict(torch.load(bce_weights, map_location=DEVICE))
bce_thr, bce_metrics = eval_with_thresholds(
    bce_model, images, masks, topk_idx
)
rows = [
    baseline_row,
    {
        "name": "BCE-only",
        "weights": os.path.basename(bce_weights),
        "best_thr": bce_thr,
        **bce_metrics,
    },
]
fig_path = save_figure(
    rows, OUT_DIR, "ablation_loss.png", "Ablation: Loss Function"
)
append_log(
    "Loss Function",
    rows,
    len(topk_idx),
    fig_path,
    "Baseline uses BCE+Dice (Ch4); ablation uses BCE-only (Ch4).",
)
print("Ablation 3 done.")

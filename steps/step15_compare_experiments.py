import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------- config -----------------
DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
LOG_PATH = r"D:\desktop\3DUNET\EXPERIMENT_LOG.md"
OUT_DIR = r"D:\desktop\3DUNET\seg_outputs_thresh"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
TOPK = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ----------------- models -----------------
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
        return self.dec(torch.cat([u, e1], dim=1))  # logits


MODELS = [
    {
        "name": "UNet3D",
        "weights": r"D:\desktop\3DUNET\best_model_unet3d.pth",
        "builder": UNet3D,
    },
    {
        "name": "UNet3DLite-Ch4",
        "weights": r"D:\desktop\3DUNET\ablation_unet3d_lite_ch4.pth",
        "builder": UNet3DLite,
        "base_ch": 4,
    },
    {
        "name": "UNet3DLite-Ch8",
        "weights": r"D:\desktop\3DUNET\unet3d_lite_v2.pth",
        "builder": UNet3DLite,
        "base_ch": 8,
    },
]


def dice_coeff(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return (2 * inter + eps) / (denom + eps)


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


def eval_model(model, images, masks, topk_idx):
    model.eval()
    with torch.no_grad():
        probs_list = []
        gts_list = []
        for j in topk_idx:
            xx = torch.tensor(images[j:j+1], dtype=torch.float32).to(DEVICE)
            gg = torch.tensor(masks[j:j+1], dtype=torch.float32).to(DEVICE)
            probs_list.append(torch.sigmoid(model(xx)))
            gts_list.append(gg)

        mean_dices = []
        mean_metrics = []
        for t in THRESHOLDS:
            ds = []
            ious = []
            precisions = []
            recalls = []
            f1s = []
            for prob, gg in zip(probs_list, gts_list):
                pred = (prob > t).float()
                metrics = seg_metrics(pred, gg)
                ds.append(metrics["dice"].item())
                ious.append(metrics["iou"].item())
                precisions.append(metrics["precision"].item())
                recalls.append(metrics["recall"].item())
                f1s.append(metrics["f1"].item())
            mean_dices.append(float(np.mean(ds)))
            mean_metrics.append({
                "dice": float(np.mean(ds)),
                "iou": float(np.mean(ious)),
                "precision": float(np.mean(precisions)),
                "recall": float(np.mean(recalls)),
                "f1": float(np.mean(f1s)),
            })

    best_idx = int(np.argmax(mean_dices))
    return THRESHOLDS[best_idx], mean_metrics[best_idx]


def append_log(rows, topk_count, fig_path):
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("\n## Dice Comparison Experiments\n")
    lines.append(f"### {ts}\n")
    lines.append(f"- Dataset: dataset_luna_seg (Top{topk_count} fg)\n")
    lines.append(f"- Thresholds: {THRESHOLDS}\n")
    lines.append("- Metric: TopK mean Dice/IoU/Precision/Recall/F1\n")
    lines.append("- Threshold selection: best Dice among thresholds\n")
    lines.append(f"- Figure: {fig_path}\n\n")
    lines.append("| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |\n")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['weights']} | {r['best_thr']:.2f} | "
            f"{r['dice']:.4f} | {r['iou']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} |\n"
        )

    Path(LOG_PATH).write_text(Path(LOG_PATH).read_text(encoding='utf-8') + "".join(lines), encoding='utf-8')


def save_figure(rows, out_dir):
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
    plt.title("Comparison Metrics (TopK FG)")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "compare_metrics_topk.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


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
print(f"Using Top{k} fg samples for Dice comparison.")

rows = []
for m in MODELS:
    weights = m["weights"]
    if not os.path.exists(weights):
        print(f"[Skip] Weights not found: {weights}")
        continue

    if "base_ch" in m:
        model = m["builder"](base_ch=m["base_ch"]).to(DEVICE)
    else:
        model = m["builder"]().to(DEVICE)
    state = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(state)

    best_thr, metrics = eval_model(model, images, masks, topk_idx)
    print(
        f"{m['name']} best_thr={best_thr:.2f} top{len(topk_idx)} "
        f"dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )

    rows.append({
        "name": m["name"],
        "weights": os.path.basename(weights),
        "best_thr": best_thr,
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    })

if rows:
    fig_path = save_figure(rows, OUT_DIR)
    append_log(rows, len(topk_idx), fig_path)
    print("Saved figure to:", fig_path)
    print("Appended results to:", LOG_PATH)
else:
    print("No results to log.")

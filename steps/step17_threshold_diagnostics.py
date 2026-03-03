import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------- config -----------------
DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
WEIGHTS = r"D:\desktop\3DUNET\ablation_unet3d_lite_ch4.pth"
OUT_DIR = r"D:\desktop\3DUNET\seg_outputs_thresh"
TOPK = 20
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
HIST_BINS = 50
UNCERTAINTY_SUPPRESS = True
U_Q_LOW = 0.50
U_Q_HIGH = 0.90
EPS = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ----------------- model -----------------
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
print(f"Using Top{k} fg samples for diagnostics.")

# ----------------- load model -----------------
if not os.path.exists(WEIGHTS):
    raise SystemExit(f"Weights not found: {WEIGHTS}")
model = UNet3DLite(base_ch=4).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

# ----------------- collect probabilities -----------------
probs_list = []
gts_list = []
u_list = []

with torch.no_grad():
    for j in topk_idx:
        xx = torch.tensor(images[j:j+1], dtype=torch.float32).to(DEVICE)
        gg = torch.tensor(masks[j:j+1], dtype=torch.float32).to(DEVICE)
        prob = torch.sigmoid(model(xx))
        probs_list.append(prob)
        gts_list.append(gg)
        if UNCERTAINTY_SUPPRESS:
            u_list.append(entropy_from_prob(prob))

# ----------------- probability histogram -----------------
if UNCERTAINTY_SUPPRESS:
    u0, u1 = compute_u_params(u_list)
    probs_list = [
        suppress_prob(prob, u, u0, u1) for prob, u in zip(probs_list, u_list)
    ]

metrics_by_t = {t: [] for t in THRESHOLDS}
for t in THRESHOLDS:
    for prob, gg in zip(probs_list, gts_list):
        pred = (prob > t).float()
        metrics_by_t[t].append(seg_metrics(pred, gg))

all_probs = np.concatenate(
    [prob.detach().cpu().numpy().ravel() for prob in probs_list], axis=0
)
os.makedirs(OUT_DIR, exist_ok=True)
plt.figure(figsize=(6, 4))
plt.hist(all_probs, bins=HIST_BINS, color="#4C78A8", alpha=0.85)
hist_title = "Probability Histogram (TopK FG Samples)"
if UNCERTAINTY_SUPPRESS:
    hist_title = "Probability Histogram (TopK FG Samples, Suppressed)"
plt.title(hist_title)
plt.xlabel("Predicted probability")
plt.ylabel("Voxel count")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
hist_path = os.path.join(OUT_DIR, "probability_histogram_topk.png")
plt.savefig(hist_path, dpi=200)
plt.close()
print("Saved histogram:", hist_path)

# ----------------- threshold curves -----------------
metrics = ["dice", "iou", "precision", "recall", "f1"]
values = {m: [] for m in metrics}
for t in THRESHOLDS:
    entries = metrics_by_t[t]
    for m in metrics:
        values[m].append(float(np.mean([e[m].item() for e in entries])))

plt.figure(figsize=(7, 4))
for m in metrics:
    plt.plot(THRESHOLDS, values[m], marker="o", label=m.capitalize())
plt.title("Metrics vs Threshold (TopK FG)")
plt.xlabel("Threshold")
plt.ylabel("TopK mean")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
curve_path = os.path.join(OUT_DIR, "metrics_vs_threshold_topk.png")
plt.savefig(curve_path, dpi=200)
plt.close()
print("Saved curves:", curve_path)

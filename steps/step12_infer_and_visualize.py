import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------- config -----------------
DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
# 建议用你最新训练保存的 best 模型
WEIGHTS  = r"D:\desktop\3DUNET\unet3d_lite_v2.pth"
OUT_DIR  = r"D:\desktop\3DUNET\seg_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

AUTO_THRESH = True
THRESH = 0.30
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
UNCERTAINTY_SUPPRESS = True
U_Q_LOW = 0.50
U_Q_HIGH = 0.90
EPS = 1e-6

# Visualization controls
VIS_NUM = 6  # number of samples to visualize
SLICE_OFFSETS = [-2,-1,0, 1,2]  # extra slices around the center per view

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


def dice_coeff(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
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


# ----------------- load data -----------------
images = np.load(os.path.join(DATA_DIR, "images.npy"))  # (N,1,64,64,64)
masks  = np.load(os.path.join(DATA_DIR, "masks.npy"))   # (N,1,64,64,64)

fg = masks.sum(axis=(1, 2, 3, 4))
ranked = np.argsort(-fg)
idx = int(ranked[0])
print("Selected idx:", idx, "fg_vox:", int(fg[idx]))

x  = torch.tensor(images[idx:idx+1], dtype=torch.float32).to(DEVICE)
gt = torch.tensor(masks[idx:idx+1], dtype=torch.float32).to(DEVICE)

# ----------------- load weights & infer -----------------
model = UNet3DLite(base_ch=8).to(DEVICE)

if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"WEIGHTS not found: {WEIGHTS}")

state = torch.load(WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ----------------- auto threshold + uncertainty suppression (Top20) -----------------
topk = ranked[:20]
u0 = u1 = None
mean_dices = []
with torch.no_grad():
    probs_list = []
    gts_list = []
    u_list = []
    for j in topk:
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
            suppress_prob(prob, u, u0, u1) for prob, u in zip(probs_list, u_list)
        ]

    if AUTO_THRESH:
        for t in THRESHOLDS:
            ds = []
            for prob, gg in zip(probs_list, gts_list):
                pred = (prob > t).float()
                ds.append(dice_coeff(pred, gg).item())
            mean_dices.append(float(np.mean(ds)))
        best_idx = int(np.argmax(mean_dices))
        THRESH = THRESHOLDS[best_idx]
        print("Auto THRESH:", THRESH, "Top20 mean Dice:", mean_dices[best_idx])

with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)
    if UNCERTAINTY_SUPPRESS:
        u = entropy_from_prob(prob)
        prob = suppress_prob(prob, u, u0, u1)
    pred = (prob > THRESH).float()

# ----------------- compute dice -----------------
dice = dice_coeff(pred, gt).item()
print(f"Dice@{THRESH:.2f} on selected sample:", dice)

# ----------------- visualize (3 views) -----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def save_triplet(view_name, img2d, gt2d, pr2d, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img2d, cmap="gray")
    axes[0].set_title(f"{view_name}: Patch")

    axes[1].imshow(img2d, cmap="gray")
    axes[1].imshow(gt2d, alpha=0.35)
    axes[1].set_title(f"{view_name}: GT overlay")

    axes[2].imshow(img2d, cmap="gray")
    axes[2].imshow(pr2d, alpha=0.35)
    axes[2].set_title(f"{view_name}: Pred overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

vis_indices = ranked[fg[ranked] > 0]
vis_indices = vis_indices[:VIS_NUM]
if len(vis_indices) < VIS_NUM:
    print(f"Only {len(vis_indices)} samples have fg>0 (VIS_NUM={VIS_NUM}).")
print("Visualizing indices:", vis_indices.tolist())

with torch.no_grad():
    for vis_idx in vis_indices:
        patch = images[vis_idx, 0]                 # (64,64,64)  z,y,x
        gtm = masks[vis_idx, 0].astype(np.uint8)   # GT mask
        xx = torch.tensor(images[vis_idx:vis_idx+1], dtype=torch.float32).to(DEVICE)
        gg = torch.tensor(masks[vis_idx:vis_idx+1], dtype=torch.float32).to(DEVICE)
        prob_v = torch.sigmoid(model(xx))
        if UNCERTAINTY_SUPPRESS:
            u_v = entropy_from_prob(prob_v)
            prob_v = suppress_prob(prob_v, u_v, u0, u1)
        pred_v = (prob_v > THRESH).float()
        dice_v = dice_coeff(pred_v, gg).item()
        prm = pred_v.detach().cpu().numpy()[0, 0].astype(np.uint8)

        coords = np.argwhere((gtm > 0) | (prm > 0))
        if len(coords) > 0:
            zc, yc, xc = coords.mean(axis=0)
            zc, yc, xc = int(round(zc)), int(round(yc)), int(round(xc))
        else:
            zc = yc = xc = patch.shape[0] // 2

        for dz in SLICE_OFFSETS:
            z = clamp(zc + dz, 0, patch.shape[0] - 1)
            y = clamp(yc + dz, 0, patch.shape[1] - 1)
            x = clamp(xc + dz, 0, patch.shape[2] - 1)

            save_triplet(
                "Axial",
                patch[z, :, :], gtm[z, :, :], prm[z, :, :],
                os.path.join(
                    OUT_DIR,
                    f"axial_idx{vis_idx}_z{z}_thr{THRESH:.2f}_dice{dice_v:.3f}.png",
                ),
            )
            save_triplet(
                "Coronal",
                patch[:, y, :], gtm[:, y, :], prm[:, y, :],
                os.path.join(
                    OUT_DIR,
                    f"coronal_idx{vis_idx}_y{y}_thr{THRESH:.2f}_dice{dice_v:.3f}.png",
                ),
            )
            save_triplet(
                "Sagittal",
                patch[:, :, x], gtm[:, :, x], prm[:, :, x],
                os.path.join(
                    OUT_DIR,
                    f"sagittal_idx{vis_idx}_x{x}_thr{THRESH:.2f}_dice{dice_v:.3f}.png",
                ),
            )

print("Saved PNGs to:", OUT_DIR)


# ----------------- quick dataset dice (Top20) -----------------
topk = np.argsort(-fg)[:20]
ds = []
with torch.no_grad():
    for j in topk:
        xx = torch.tensor(images[j:j+1], dtype=torch.float32).to(DEVICE)
        gg = torch.tensor(masks[j:j+1], dtype=torch.float32).to(DEVICE)
        prob = torch.sigmoid(model(xx))
        if UNCERTAINTY_SUPPRESS:
            u = entropy_from_prob(prob)
            prob = suppress_prob(prob, u, u0, u1)
        pp = (prob > THRESH).float()   # 统一阈值！
        ds.append(dice_coeff(pp, gg).item())

print(f"Top20 mean Dice@{THRESH:.2f}:", float(np.mean(ds)))

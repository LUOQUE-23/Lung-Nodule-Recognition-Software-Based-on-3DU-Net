import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
WEIGHTS  = r"D:\desktop\3DUNET\unet3d_lite_v2.pth"
OUT_DIR  = r"D:\desktop\3DUNET\seg_outputs_thresh"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

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
        u  = self.up(e2)
        return self.dec(torch.cat([u, e1], dim=1))  # logits

def dice_coeff(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return (2*inter + eps) / (denom + eps)

def save_triplet(view_name, img2d, gt2d, pr2d, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img2d, cmap="gray"); axes[0].set_title(f"{view_name}: Patch")
    axes[1].imshow(img2d, cmap="gray"); axes[1].imshow(gt2d, alpha=0.35); axes[1].set_title(f"{view_name}: GT overlay")
    axes[2].imshow(img2d, cmap="gray"); axes[2].imshow(pr2d, alpha=0.35); axes[2].set_title(f"{view_name}: Pred overlay")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

# load data
images = np.load(os.path.join(DATA_DIR, "images.npy"))  # (N,1,64,64,64)
masks  = np.load(os.path.join(DATA_DIR, "masks.npy"))   # (N,1,64,64,64)

# choose top20 foreground samples
fg = masks.sum(axis=(1,2,3,4))
topk = np.argsort(-fg)[:20]
print("Top20 fg_vox mean:", float(fg[topk].mean()))

# load model
model = UNet3DLite().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
mean_dices = []

with torch.no_grad():
    # precompute probabilities for topk to speed up
    probs_list = []
    gts_list = []
    for j in topk:
        xx = torch.tensor(images[j:j+1], dtype=torch.float32).to(DEVICE)
        gg = torch.tensor(masks[j:j+1], dtype=torch.float32).to(DEVICE)
        prob = torch.sigmoid(model(xx))  # (1,1,64,64,64)
        probs_list.append(prob)
        gts_list.append(gg)

    for t in thresholds:
        ds = []
        for prob, gg in zip(probs_list, gts_list):
            pred = (prob > t).float()
            ds.append(dice_coeff(pred, gg).item())
        md = float(np.mean(ds))
        mean_dices.append(md)
        print(f"Top20 mean Dice @ thr={t:.1f}: {md:.4f}")

# plot curve
plt.figure(figsize=(6,4))
plt.plot(thresholds, mean_dices, marker="o")
plt.xlabel("Threshold")
plt.ylabel("Top20 mean Dice")
plt.title("Threshold Sensitivity (Top20)")
plt.grid(True, alpha=0.3)
curve_path = os.path.join(OUT_DIR, "threshold_curve.png")
plt.tight_layout()
plt.savefig(curve_path, dpi=200)
plt.close()
print("Saved curve:", curve_path)

# save overlays for one representative sample (foreground max)
idx = int(fg.argmax())
x = torch.tensor(images[idx:idx+1], dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    prob = torch.sigmoid(model(x))[0,0].cpu().numpy()  # (64,64,64)

patch = images[idx,0]
gtm = masks[idx,0].astype(np.uint8)
mid = patch.shape[0] // 2

for t in thresholds:
    prm = (prob > t).astype(np.uint8)
    save_triplet("Axial", patch[mid,:,:], gtm[mid,:,:], prm[mid,:,:],
                 os.path.join(OUT_DIR, f"axial_idx{idx}_thr{t:.1f}.png"))
    save_triplet("Coronal", patch[:,mid,:], gtm[:,mid,:], prm[:,mid,:],
                 os.path.join(OUT_DIR, f"coronal_idx{idx}_thr{t:.1f}.png"))
    save_triplet("Sagittal", patch[:,:,mid], gtm[:,:,mid], prm[:,:,mid],
                 os.path.join(OUT_DIR, f"sagittal_idx{idx}_thr{t:.1f}.png"))

print("Saved overlays to:", OUT_DIR)

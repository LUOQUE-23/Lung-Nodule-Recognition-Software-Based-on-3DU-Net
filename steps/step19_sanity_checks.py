import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------- config -----------------
DATA_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
WEIGHTS = r"D:\desktop\3DUNET\ablation_unet3d_lite_ch4.pth"
OUT_DIR = r"D:\desktop\3DUNET\seg_outputs_thresh"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ----------------- model (must match training) -----------------
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


# ----------------- check weights keys -----------------
if not os.path.exists(WEIGHTS):
    raise SystemExit(f"Weights not found: {WEIGHTS}")

model = UNet3DLite(base_ch=4).to(DEVICE)
state = torch.load(WEIGHTS, map_location=DEVICE)
missing, unexpected = model.load_state_dict(state, strict=False)
print("Weight key check:")
print("  Missing keys:", len(missing))
print("  Unexpected keys:", len(unexpected))
if missing:
    print("  Missing list (first 10):", missing[:10])
if unexpected:
    print("  Unexpected list (first 10):", unexpected[:10])

# ----------------- dataset foreground stats -----------------
images = np.load(os.path.join(DATA_DIR, "images.npy"))
masks = np.load(os.path.join(DATA_DIR, "masks.npy"))

fg = masks.sum(axis=(1, 2, 3, 4)).astype(np.float64)
fg_pos = fg[fg > 0]

print("Dataset foreground stats:")
print("  Total samples:", len(fg))
print("  FG>0 samples:", int((fg > 0).sum()))
print("  FG max:", float(fg.max()))
print("  FG mean (all):", float(fg.mean()))
print("  FG mean (fg>0):", float(fg_pos.mean()) if len(fg_pos) else 0.0)

os.makedirs(OUT_DIR, exist_ok=True)

plt.figure(figsize=(6, 4))
plt.hist(fg, bins=60, color="#4C78A8", alpha=0.85)
plt.title("Foreground Voxels per Sample")
plt.xlabel("Foreground voxel count")
plt.ylabel("Sample count")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
fg_hist_path = os.path.join(OUT_DIR, "fg_voxels_histogram.png")
plt.savefig(fg_hist_path, dpi=200)
plt.close()
print("Saved FG histogram:", fg_hist_path)

if len(fg_pos) > 0:
    plt.figure(figsize=(6, 4))
    plt.hist(fg_pos, bins=60, color="#F58518", alpha=0.85)
    plt.title("Foreground Voxels (FG>0 Samples)")
    plt.xlabel("Foreground voxel count")
    plt.ylabel("Sample count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fg_pos_hist_path = os.path.join(OUT_DIR, "fg_voxels_histogram_fgpos.png")
    plt.savefig(fg_pos_hist_path, dpi=200)
    plt.close()
    print("Saved FG>0 histogram:", fg_pos_hist_path)

# ----------------- simple inference sanity check -----------------
model.eval()
with torch.no_grad():
    idx = int(np.argmax(fg))
    x = torch.tensor(images[idx:idx+1], dtype=torch.float32).to(DEVICE)
    logits = model(x)
    prob = torch.sigmoid(logits).cpu().numpy().ravel()

print("Inference sanity check (max FG sample):")
print("  Logits mean/std:", float(logits.mean().item()), float(logits.std().item()))
print("  Prob mean/std:", float(prob.mean()), float(prob.std()))

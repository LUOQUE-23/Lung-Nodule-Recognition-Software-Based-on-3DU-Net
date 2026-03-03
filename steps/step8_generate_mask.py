import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

PATCH_SIZE = 64
HALF = PATCH_SIZE // 2

# ===== 路径 =====
LUNA_ROOT = r"D:\desktop\3DUNET\LUNA16"
CSV_PATH = os.path.join(LUNA_ROOT, "annotations.csv")
DATA_PATH = os.path.join(LUNA_ROOT, "data")

# ===== 1. 读取一条结节 =====
df = pd.read_csv(CSV_PATH)
row = df.iloc[0]

seriesuid = row["seriesuid"]
world = np.array([row["coordX"], row["coordY"], row["coordZ"]])
diam_mm = row["diameter_mm"]

# ===== 2. 找 CT =====
mhd_file = None
for subset in os.listdir(DATA_PATH):
    subset_dir = os.path.join(DATA_PATH, subset)
    cand = os.path.join(subset_dir, seriesuid + ".mhd")
    if os.path.exists(cand):
        mhd_file = cand
        break

img = sitk.ReadImage(mhd_file)
origin = np.array(img.GetOrigin())
spacing = np.array(img.GetSpacing())

# ===== 3. 世界坐标 → voxel =====
voxel = ((world - origin) / spacing).round().astype(int)
cx, cy, cz = voxel

# ===== 4. 裁 patch =====
vol = sitk.GetArrayFromImage(img)  # z,y,x
pad = HALF
vol_pad = np.pad(vol, ((pad,pad),(pad,pad),(pad,pad)), constant_values=-1024)

cz += pad; cy += pad; cx += pad

patch = vol_pad[
    cz-HALF:cz+HALF,
    cy-HALF:cy+HALF,
    cx-HALF:cx+HALF
]

# ===== 5. 生成 mask =====
# 半径（voxel）
radius = (diam_mm / 2) / spacing
rx, ry, rz = radius

mask = np.zeros_like(patch, dtype=np.uint8)

for z in range(PATCH_SIZE):
    for y in range(PATCH_SIZE):
        for x in range(PATCH_SIZE):
            dz = (z - HALF) / rz
            dy = (y - HALF) / ry
            dx = (x - HALF) / rx
            if dx*dx + dy*dy + dz*dz <= 1:
                mask[z, y, x] = 1

print("Mask voxels:", mask.sum())

# ===== 6. 可视化验证 =====
mid = HALF
fig, axes = plt.subplots(1, 2, figsize=(8,4))

axes[0].imshow(patch[mid], cmap="gray")
axes[0].set_title("Patch (Axial)")

axes[1].imshow(mask[mid], cmap="gray")
axes[1].set_title("Mask (Axial)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

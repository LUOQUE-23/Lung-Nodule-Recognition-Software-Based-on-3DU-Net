import matplotlib.pyplot as plt
import numpy as np

# 假设你已经从 step3 得到了 patch
# 这里我们直接加载 step3 中的方式，简单起见重新裁一次

import pandas as pd
import os
import SimpleITK as sitk

PATCH_SIZE = 64
HALF = PATCH_SIZE // 2

LUNA_ROOT = r"D:\desktop\3DUNET\LUNA16"
CSV_PATH = os.path.join(LUNA_ROOT, "annotations.csv")
DATA_PATH = os.path.join(LUNA_ROOT, "data")

df = pd.read_csv(CSV_PATH)
row = df.iloc[0]

seriesuid = row["seriesuid"]
world_coord = np.array([
    row["coordX"],
    row["coordY"],
    row["coordZ"]
])

# 找 CT
mhd_file = None
for subset in os.listdir(DATA_PATH):
    subset_dir = os.path.join(DATA_PATH, subset)
    if not os.path.isdir(subset_dir):
        continue
    candidate = os.path.join(subset_dir, seriesuid + ".mhd")
    if os.path.exists(candidate):
        mhd_file = candidate
        break

img = sitk.ReadImage(mhd_file)
origin = np.array(img.GetOrigin())
spacing = np.array(img.GetSpacing())

voxel = ((world_coord - origin) / spacing).round().astype(int)
cx, cy, cz = voxel

volume = sitk.GetArrayFromImage(img)

pad = HALF
volume_pad = np.pad(volume, ((pad, pad), (pad, pad), (pad, pad)), mode="constant")
cz += pad
cy += pad
cx += pad

patch = volume_pad[
    cz-HALF:cz+HALF,
    cy-HALF:cy+HALF,
    cx-HALF:cx+HALF
]

# ===== 可视化 =====
mid = PATCH_SIZE // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(patch[mid, :, :], cmap="gray")
axes[0].set_title("Axial")

axes[1].imshow(patch[:, mid, :], cmap="gray")
axes[1].set_title("Coronal")

axes[2].imshow(patch[:, :, mid], cmap="gray")
axes[2].set_title("Sagittal")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

import pandas as pd
import os
import SimpleITK as sitk
import numpy as np

# ===== 参数 =====
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
world_coord = np.array([
    row["coordX"],
    row["coordY"],
    row["coordZ"]
])

# ===== 2. 找 CT =====
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

# ===== 3. 世界坐标 → voxel =====
voxel = ((world_coord - origin) / spacing).round().astype(int)
cx, cy, cz = voxel

# ===== 4. 转为 numpy =====
volume = sitk.GetArrayFromImage(img)  # [z, y, x]

# ===== 5. padding（防止越界）=====
pad = HALF
volume_pad = np.pad(
    volume,
    pad_width=((pad, pad), (pad, pad), (pad, pad)),
    mode="constant",
    constant_values=0
)

# 修正中心点（因为 pad 了）
cz += pad
cy += pad
cx += pad

# ===== 6. 裁 patch =====
patch = volume_pad[
    cz-HALF:cz+HALF,
    cy-HALF:cy+HALF,
    cx-HALF:cx+HALF
]

print("Patch shape:", patch.shape)
print("Patch HU range:", patch.min(), patch.max())

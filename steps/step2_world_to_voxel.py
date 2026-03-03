import pandas as pd
import os
import SimpleITK as sitk
import numpy as np

# ===== 路径 =====
LUNA_ROOT = r"D:\desktop\3DUNET\LUNA16"
CSV_PATH = os.path.join(LUNA_ROOT, "annotations.csv")
DATA_PATH = os.path.join(LUNA_ROOT, "data")

# ===== 1. 读取 CSV，取一条结节 =====
df = pd.read_csv(CSV_PATH)
row = df.iloc[0]

seriesuid = row["seriesuid"]
world_coord = np.array([
    row["coordX"],
    row["coordY"],
    row["coordZ"]
])

print("World coord (mm):", world_coord)

# ===== 2. 找到对应 CT =====
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
size = np.array(img.GetSize())

print("Origin:", origin)
print("Spacing:", spacing)
print("Size:", size)

# ===== 3. 世界坐标 → 体素坐标 =====
voxel_coord = (world_coord - origin) / spacing

print("\nVoxel coord (float):", voxel_coord)
print("Voxel coord (int):", np.round(voxel_coord).astype(int))

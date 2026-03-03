import pandas as pd
import os
import SimpleITK as sitk

# ===== 路径设置 =====
LUNA_ROOT = r"D:\desktop\3DUNET\LUNA16"
CSV_PATH = os.path.join(LUNA_ROOT, "annotations.csv")
DATA_PATH = os.path.join(LUNA_ROOT, "data")

# ===== 1. 读取 CSV =====
df = pd.read_csv(CSV_PATH)
print("Total nodules:", len(df))
print(df.head())

# ===== 2. 取第一条结节 =====
row = df.iloc[0]

seriesuid = row["seriesuid"]
world_coord = (
    row["coordX"],
    row["coordY"],
    row["coordZ"]
)

print("\nSelected seriesuid:", seriesuid)
print("World coord (mm):", world_coord)

# ===== 3. 找到对应的 mhd 文件 =====
mhd_file = None
for subset in os.listdir(DATA_PATH):
    subset_dir = os.path.join(DATA_PATH, subset)
    if not os.path.isdir(subset_dir):
        continue
    candidate = os.path.join(subset_dir, seriesuid + ".mhd")
    if os.path.exists(candidate):
        mhd_file = candidate
        break

print("\nMatched mhd file:", mhd_file)

# ===== 4. 尝试读取该 CT =====
img = sitk.ReadImage(mhd_file)
print("CT size:", img.GetSize())
print("CT spacing:", img.GetSpacing())
print("CT origin:", img.GetOrigin())

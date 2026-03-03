import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import random

PATCH_SIZE = 64
HALF = PATCH_SIZE // 2
NUM_NEG = 200        # 与正样本数量一致
MIN_DIST = 2 * HALF # 与结节中心的最小距离（voxel）

LUNA_ROOT = r"D:\desktop\3DUNET\LUNA16"
CSV_PATH = os.path.join(LUNA_ROOT, "annotations.csv")
DATA_PATH = os.path.join(LUNA_ROOT, "data")

OUT_DIR = r"D:\desktop\3DUNET\dataset_luna"
os.makedirs(OUT_DIR, exist_ok=True)

def find_mhd(seriesuid: str) -> str:
    for subset in os.listdir(DATA_PATH):
        subset_dir = os.path.join(DATA_PATH, subset)
        if not os.path.isdir(subset_dir):
            continue
        cand = os.path.join(subset_dir, seriesuid + ".mhd")
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(seriesuid)

def world_to_voxel(img, world):
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    return ((world - origin) / spacing).round().astype(int)

def crop_patch(vol, center):
    cx, cy, cz = center
    pad = HALF
    vol_pad = np.pad(vol, ((pad,pad),(pad,pad),(pad,pad)), constant_values=-1024)
    cz += pad; cy += pad; cx += pad
    return vol_pad[
        cz-HALF:cz+HALF,
        cy-HALF:cy+HALF,
        cx-HALF:cx+HALF
    ]

# ===== 读取正样本元信息 =====
pos_meta = pd.read_csv(os.path.join(OUT_DIR, "meta_pos.csv"))

# 按 CT 分组，便于负样本采样
grouped = pos_meta.groupby("seriesuid")

neg_patches = []
neg_meta = []

cache_img = {}
cache_vol = {}

for seriesuid, group in grouped:
    if len(neg_patches) >= NUM_NEG:
        break

    # 读 CT
    mhd = find_mhd(seriesuid)
    img = sitk.ReadImage(mhd)
    vol = sitk.GetArrayFromImage(img)  # z,y,x

    # 所有结节中心（voxel）
    centers = []
    for _, row in group.iterrows():
        w = np.array([row["coordX"], row["coordY"], row["coordZ"]])
        centers.append(world_to_voxel(img, w))
    centers = np.array(centers)

    zmax, ymax, xmax = vol.shape

    trials = 0
    while len(neg_patches) < NUM_NEG and trials < 200:
        trials += 1

        cx = random.randint(HALF, xmax - HALF - 1)
        cy = random.randint(HALF, ymax - HALF - 1)
        cz = random.randint(HALF, zmax - HALF - 1)
        candidate = np.array([cx, cy, cz])

        # 距离最近结节是否足够远
        dists = np.linalg.norm(centers - candidate, axis=1)
        if np.min(dists) < MIN_DIST:
            continue

        patch = crop_patch(vol, candidate)

        if np.mean(patch == -1024) > 0.95:
            continue

        neg_patches.append(patch.astype(np.int16))
        neg_meta.append({
            "seriesuid": seriesuid,
            "cx": cx, "cy": cy, "cz": cz
        })

        if len(neg_patches) % 20 == 0:
            print(f"Built {len(neg_patches)} / {NUM_NEG}")

# 保存
neg_patches = np.stack(neg_patches, axis=0)
meta_neg = pd.DataFrame(neg_meta)

np.savez_compressed(
    os.path.join(OUT_DIR, "neg_patches.npz"),
    patches=neg_patches
)
meta_neg.to_csv(os.path.join(OUT_DIR, "meta_neg.csv"), index=False)

print("\nSaved neg_patches.npz shape:", neg_patches.shape)
print("Saved meta_neg.csv rows:", len(meta_neg))

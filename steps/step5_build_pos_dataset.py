import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

PATCH_SIZE = 64
HALF = PATCH_SIZE // 2
NUM_SAMPLES = 200  # 先做 200 个正样本验证流程，后面再扩大

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

def crop_patch(img: sitk.Image, world_coord_xyz: np.ndarray) -> np.ndarray:
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())

    voxel = ((world_coord_xyz - origin) / spacing).round().astype(int)
    cx, cy, cz = voxel

    vol = sitk.GetArrayFromImage(img)  # z,y,x

    pad = HALF
    vol_pad = np.pad(vol, ((pad, pad), (pad, pad), (pad, pad)), mode="constant", constant_values=-1024)

    cz += pad
    cy += pad
    cx += pad

    patch = vol_pad[
        cz-HALF:cz+HALF,
        cy-HALF:cy+HALF,
        cx-HALF:cx+HALF
    ]
    return patch

# ===== 读取标注 =====
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # 打乱

patches = []
meta_rows = []

cache = {}  # seriesuid -> sitk.Image，避免重复读盘

for i, row in df.iterrows():
    if len(patches) >= NUM_SAMPLES:
        break

    seriesuid = row["seriesuid"]
    world = np.array([row["coordX"], row["coordY"], row["coordZ"]], dtype=np.float32)
    diam = float(row["diameter_mm"])

    if seriesuid not in cache:
        mhd = find_mhd(seriesuid)
        cache[seriesuid] = sitk.ReadImage(mhd)

    img = cache[seriesuid]
    patch = crop_patch(img, world)

    # 基本质量检查：不应全是 padding
    if np.mean(patch == -1024) > 0.95:
        continue

    patches.append(patch.astype(np.int16))
    meta_rows.append({
        "seriesuid": seriesuid,
        "coordX": world[0],
        "coordY": world[1],
        "coordZ": world[2],
        "diameter_mm": diam
    })

    if len(patches) % 20 == 0:
        print(f"Built {len(patches)} / {NUM_SAMPLES}")

patches = np.stack(patches, axis=0)  # [N, 64,64,64]
meta = pd.DataFrame(meta_rows)

npz_path = os.path.join(OUT_DIR, "pos_patches.npz")
csv_out = os.path.join(OUT_DIR, "meta_pos.csv")

np.savez_compressed(npz_path, patches=patches)
meta.to_csv(csv_out, index=False)

print("\nSaved:", npz_path, "shape:", patches.shape, "dtype:", patches.dtype)
print("Saved:", csv_out, "rows:", len(meta))

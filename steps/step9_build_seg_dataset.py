import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

PATCH_SIZE = 64
HALF = PATCH_SIZE // 2
NUM_POS = 300
NEG_RATIO = 1.0
NUM_NEG = int(NUM_POS * NEG_RATIO)
MIN_DIST = 2 * HALF
MAX_NEG_TRIALS = 600

LUNA_ROOT = r"D:\desktop\3DUNET\LUNA16"
CSV_PATH = os.path.join(LUNA_ROOT, "annotations.csv")
DATA_PATH = os.path.join(LUNA_ROOT, "data")

OUT_DIR = r"D:\desktop\3DUNET\dataset_luna_seg"
os.makedirs(OUT_DIR, exist_ok=True)

def find_mhd(seriesuid):
    for subset in os.listdir(DATA_PATH):
        cand = os.path.join(DATA_PATH, subset, seriesuid + ".mhd")
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(seriesuid)

def world_to_voxel(img, world):
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    return ((world - origin) / spacing).round().astype(int), spacing

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

def make_mask(spacing, diameter_mm):
    rx, ry, rz = (diameter_mm / 2) / spacing
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    for z in range(PATCH_SIZE):
        for y in range(PATCH_SIZE):
            for x in range(PATCH_SIZE):
                dz = (z - HALF) / rz
                dy = (y - HALF) / ry
                dx = (x - HALF) / rx
                if dx*dx + dy*dy + dz*dz <= 1:
                    mask[z,y,x] = 1
    return mask

df = pd.read_csv(CSV_PATH).sample(frac=1.0, random_state=0)

images = []
masks = []

cache = {}

for _, row in df.iterrows():
    if len(images) >= NUM_POS:
        break

    seriesuid = row["seriesuid"]
    world = np.array([row["coordX"], row["coordY"], row["coordZ"]])
    diam = row["diameter_mm"]

    if seriesuid not in cache:
        img = sitk.ReadImage(find_mhd(seriesuid))
        cache[seriesuid] = img
    else:
        img = cache[seriesuid]

    voxel, spacing = world_to_voxel(img, world)
    vol = sitk.GetArrayFromImage(img)

    patch = crop_patch(vol, voxel)
    mask = make_mask(spacing, diam)

    if np.mean(mask) < 0.001:
        continue

    patch = np.clip(patch, -1024, 1024) / 1024.0

    images.append(patch[None, ...])  # (1,64,64,64)
    masks.append(mask[None, ...])

    if len(images) % 20 == 0:
        print(f"Built {len(images)} / {NUM_POS}")

# ----------------- build negative samples -----------------
rng = np.random.default_rng(0)
series_uids = df["seriesuid"].unique()
series_to_rows = df.groupby("seriesuid")

neg_images = []
neg_masks = []

for seriesuid in series_uids:
    if len(neg_images) >= NUM_NEG:
        break

    if seriesuid not in cache:
        img = sitk.ReadImage(find_mhd(seriesuid))
        cache[seriesuid] = img
    else:
        img = cache[seriesuid]

    vol = sitk.GetArrayFromImage(img)

    centers = []
    for _, row in series_to_rows.get_group(seriesuid).iterrows():
        world = np.array([row["coordX"], row["coordY"], row["coordZ"]])
        voxel, _ = world_to_voxel(img, world)
        centers.append(voxel)
    centers = np.array(centers)

    zmax, ymax, xmax = vol.shape
    trials = 0
    while len(neg_images) < NUM_NEG and trials < MAX_NEG_TRIALS:
        trials += 1
        cx = int(rng.integers(HALF, xmax - HALF))
        cy = int(rng.integers(HALF, ymax - HALF))
        cz = int(rng.integers(HALF, zmax - HALF))
        candidate = np.array([cx, cy, cz])

        dists = np.linalg.norm(centers - candidate, axis=1)
        if np.min(dists) < MIN_DIST:
            continue

        patch = crop_patch(vol, candidate)
        if np.mean(patch == -1024) > 0.95:
            continue

        patch = np.clip(patch, -1024, 1024) / 1024.0
        neg_images.append(patch[None, ...])
        neg_masks.append(np.zeros((1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8))

        if len(neg_images) % 20 == 0:
            print(f"Built {len(neg_images)} / {NUM_NEG} negatives")

if len(neg_images) > 0:
    images.extend(neg_images)
    masks.extend(neg_masks)

images = np.stack(images, axis=0)
masks = np.stack(masks, axis=0)

np.save(os.path.join(OUT_DIR, "images.npy"), images)
np.save(os.path.join(OUT_DIR, "masks.npy"), masks)

print("\nSaved images:", images.shape)
print("Saved masks :", masks.shape)

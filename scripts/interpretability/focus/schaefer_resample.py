import numpy as np
import json
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img, get_data, math_img
from pathlib import Path
import nibabel as nib

from nilearn.image import get_data
from nibabel.affines import apply_affine

def compute_parcel_centroids(img):
    """
    Compute MNI centroid coordinates for each parcel.
    Returns:
        dict[label_id] = [R, A, S]
    """
    data = get_data(img)
    affine = img.affine

    centroids = {}

    roi_ids = np.unique(data)
    roi_ids = roi_ids[roi_ids != 0]

    for roi in roi_ids:
        voxels = np.argwhere(data == roi)

        # voxel-space centroid
        centroid_voxel = voxels.mean(axis=0)

        # convert to MNI coordinates
        centroid_mni = apply_affine(affine, centroid_voxel)

        centroids[int(roi)] = centroid_mni.tolist()

    return centroids

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "exp_outputs"
    / "summary"
    / "networks"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Load Schaefer atlases (100 and 400 parcels)
# ------------------------------------------------------------
print("Loading Schaefer atlases...")

atlas_100 = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
atlas_400 = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1)

img_100 = nib.load(atlas_100.maps)
img_400 = nib.load(atlas_400.maps)

labels_100 = atlas_100.labels
labels_400 = atlas_400.labels

print("Computing centroids...")

centroids_100 = compute_parcel_centroids(img_100)
centroids_400 = compute_parcel_centroids(img_400)

# ------------------------------------------------------------
# 2. Resample 400 → 100 space (ensures alignment)
# ------------------------------------------------------------
print("Resampling atlases to same space...")

img_400_resampled = resample_to_img(img_400, img_100, interpolation="nearest")

data_100 = get_data(img_100)
data_400 = get_data(img_400_resampled)

# ------------------------------------------------------------
# 3. Prepare mapping dictionary
# ------------------------------------------------------------
mapping = {}

print("Computing parcel-wise overlaps...")

# ignore background label 0
roi_ids_100 = np.unique(data_100)
roi_ids_100 = roi_ids_100[roi_ids_100 != 0].astype(int)

roi_ids_400 = np.unique(data_400)
roi_ids_400 = roi_ids_400[roi_ids_400 != 0].astype(int)

for r400 in roi_ids_400:
    r400 = int(r400)
    mask_400 = (data_400 == r400)

    best_roi = None
    best_score = -1

    for r100 in roi_ids_100:
        r100 = int(r100)
        mask_100 = (data_100 == r100)

        # Dice coefficient
        intersection = np.logical_and(mask_400, mask_100).sum()
        size_sum = mask_400.sum() + mask_100.sum()

        dice = (2.0 * intersection) / size_sum if size_sum > 0 else 0

        if dice > best_score:
            best_score = dice
            best_roi = r100

    label_400 = labels_400[r400]
    label_100 = labels_100[best_roi]

    # mapping[label_400] = {
    #     "mapped_to_100_label": label_100,
    #     "mapped_to_100_roi": int(best_roi),
    #     "dice_score": float(best_score)
    # }
    mapping[label_400] = {
        "schaefer400_roi": int(r400),
        "schaefer400_label": label_400,
        "schaefer400_centroid_RAS": [
            round(v, 2) for v in centroids_400[r400]
        ],

        "mapped_to_100_roi": int(best_roi),
        "mapped_to_100_label": label_100,
        "mapped_to_100_centroid_RAS": [
            round(v, 2) for v in centroids_100[best_roi]
        ],

        "dice_score": float(best_score)
    }

# ------------------------------------------------------------
# 4. Save JSON output
# ------------------------------------------------------------
output_file = OUTPUT_DIR / "schaefer400_to_100_mapping.json"

with open(output_file, "w") as f:
    json.dump(mapping, f, indent=4)

print(f"Mapping saved to {output_file}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn import image
from nilearn import plotting
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "schaefer_networks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1. Load Schaefer atlas (100 parcels, 17 networks)
# ----------------------------
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=100,
    yeo_networks=17,
    resolution_mm=2
)

labels = atlas.labels  # parcel names
maps_img = atlas.maps   # NIfTI atlas image (path or image)
if isinstance(maps_img, str):
    maps_img = image.load_img(maps_img)

# ----------------------------
# 2. Parse network assignment from labels
# ----------------------------
# Schaefer labels look like:
# "17Networks_RH_VisPeri_ExStrInf_1", "17Networks_RH_TempPar_2", "Background", etc.

def _normalize_label(label):
    return label.decode("utf-8") if isinstance(label, bytes) else label


def extract_network(label):
    """Extract network name (e.g., 17Networks_VisPeri) from Schaefer label."""
    label = _normalize_label(label)
    if label == "Background":
        return "Background"

    parts = label.split("_")
    # Typical format:
    # 17Networks_<Hemisphere>_<Network>_<Region...>_<ParcelID>
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[2]}"
    return "Unknown"


def extract_region_name(label):
    """Extract region name without the '17Networks' prefix."""
    label = _normalize_label(label)
    if label == "Background":
        return "Background"
    parts = label.split("_")
    if len(parts) >= 2:
        return "_".join(parts[1:])
    return "Unknown"


label_str = [_normalize_label(l) for l in labels]
network_names = [extract_network(l) for l in labels]
region_names = [extract_region_name(l) for l in labels]

df = pd.DataFrame({
    "region_id": np.arange(len(labels)),
    "label": label_str,
    "region_name": region_names,
    "network": network_names
})

# # ----------------------------
# # 3. Count how many regions per network
# # ----------------------------
# network_counts = df["network"].value_counts().sort_values(ascending=False)

# print("\nParcel count per Yeo-17 network:\n")
# print(network_counts)

# # ----------------------------
# # 4. Bar plot: distribution of 100 parcels across 17 networks
# # ----------------------------
# plt.figure(figsize=(10, 5))
# network_counts.plot(kind="bar")
# plt.title("Schaefer-100 parcel distribution across Yeo-17 networks")
# plt.xlabel("Network")
# plt.ylabel("Number of parcels")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(OUT_DIR / "schaefer_network_distribution.png")
# plt.show()

# ----------------------------
# 5. Assign integer labels for visualization on brain
# ----------------------------
# Map each network to a color index
unique_networks = sorted(df["network"].unique())
networks_no_bg = [net for net in unique_networks if net != "Background"]
network_to_id = {net: i + 1 for i, net in enumerate(networks_no_bg)}
region_labels = np.array([network_to_id.get(n, 0) for n in df["network"]])

# ----------------------------
# 6. Plot brain surface (network-colored)
# ----------------------------
display = plotting.plot_roi(
    maps_img,
    title="Schaefer-100 colored by Yeo-17 networks",
    cmap="tab20",
    colorbar=True
)

display.savefig(OUT_DIR / "schaefer_networks_surface.png")
# plotting.show()

# ----------------------------
# 7. Plot network IDs (volume)
# ----------------------------
label_lookup = np.zeros(len(labels), dtype=int)
for idx, net in zip(df["region_id"], df["network"]):
    label_lookup[int(idx)] = network_to_id.get(net, 0)

mapped_data = label_lookup[maps_img.get_fdata().astype(int)]
network_img = image.new_img_like(maps_img, mapped_data)

display = plotting.plot_img(
    network_img,
    title="Schaefer-100 networks (ID map)",
    cmap="tab20",
    colorbar=True
)

display.savefig(OUT_DIR / "schaefer_networks_volume.png")

# # ----------------------------
# # 7. Optional: save mapping table
# # ----------------------------
# df.to_csv("schaefer100_to_yeo17_mapping.csv", index=False)
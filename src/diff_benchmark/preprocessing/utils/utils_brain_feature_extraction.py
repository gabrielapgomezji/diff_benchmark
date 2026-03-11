"""Parcellation and brain mask utilities.

This module provides helpers for reading FreeSurfer label files, extracting
parcellation labels from NIfTI headers, and building tissue/ventricle masks.

Metric computation and surface/skeleton projection have been moved to:
- :mod:`diff_benchmark.preprocessing.utils_dwi_metrics`
- :mod:`diff_benchmark.preprocessing.utils_surface_skeleton`

All public names are re-exported here for **full backward compatibility** —
existing code that imports from this module will continue to work unchanged.
"""
import json
from pathlib import Path
from xml import etree

import nibabel as nib
import numpy as np
from nilearn import image as nimage
from scipy import ndimage

from diff_benchmark.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Re-exports for backward compatibility
# ---------------------------------------------------------------------------
from diff_benchmark.preprocessing.utils.utils_dmri_metrics import (  # noqa: F401
    METRIC_COMPUTERS,
    compute_b0,
    compute_md,
    compute_mk,
    compute_rtop,
    compute_save_and_project_metric,
    compute_sh,
)
from diff_benchmark.preprocessing.utils.utils_surface_skeleton import (  # noqa: F401
    average_per_parcel,
    build_parcel_label_vector,
    classify_tract_hemisphere,
    download_fsl_skeleton,
    extract_region_data,
    extract_wm_tract_subset,
    get_jhu_tract_names,
    load_tbss_skeleton,
    load_template_surface,
    project_to_skeleton,
    project_to_surface,
    resample_schaefer_onto_fs_lr,
    resample_subject_to_template,
)

logger = setup_logger(__name__)


def read_label_file() -> dict:
    """Read the FreeSurfer colour LUT and return a mapping of label name → index.

    Returns:
        Dict with lowercase label names as keys and integer indices as values.
    """
    filepath = (
        Path(__file__).parent.parent.parent.parent
        / "aux_materials/FreeSurferColorLUT.txt"
    )
    label_dict = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip empty lines and header comments
            if not line or line.startswith("#"):
                continue

            # split by whitespace; expected format: index label_name R G B A
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                index = int(parts[0])
                label_name = parts[1].lower()
                label_dict[label_name] = index
            except (ValueError, IndexError):
                continue

    return label_dict


def extract_selected_labels(
    nifti_path: Path, labels_dict: dict | None = None, tissue_type: str = "gray"
) -> dict:
    """Extract parcellation labels from a NIfTI header extension.

    Args:
        nifti_path: Path to the NIfTI file with an embedded XML label table.
        labels_dict: Fallback label dict used when header parsing fails.
        tissue_type: ``"gray"`` keeps cortical and ventricular labels;
            ``"white"`` keeps white-matter and ventricular labels.

    Returns:
        Dict mapping label name → integer index.
    """
    try:
        header = nib.load(nifti_path).header
        labels = {
            n.text.lower(): int(n.get("Key"))
            for n in etree.ElementTree.fromstring(header.extensions[0].text).findall(
                ".//Label"
            )
        }
        if tissue_type == "gray":
            return {
                k: v
                for k, v in labels.items()
                if k.startswith("ctx") or "ventricle" in k
            }
        elif tissue_type == "white":
            return {
                k: v
                for k, v in labels_dict.items()
                if any(
                    [
                        ("white" in k and "matter" in k),
                        ("cerebral-white-matter" in k),
                        ("wm-" in k),
                        ("ventricle" in k),
                        (v in [2, 41]),
                    ]
                )
            }
    except Exception as e:
        logger.warning(f"Error extracting labels from given file: {e}")
        if labels_dict is not None:
            logger.info("Using provided labels_dict instead.")
            return labels_dict
        logger.info("Loading labels from fs_labels.json")
        fs_labels = (
            Path(__file__).parent.parent.parent.parent / "aux_materials/fs_labels.json"
        )
        labels_dict = json.load(fs_labels.open())
        return labels_dict


def create_masks(
    parcellation_img: nib.nifti1.Nifti1Image,
    labels: dict,
    selected_labels: list | None = None,
    tissue_type: str = "gray",
) -> tuple:
    """Build tissue and ventricle masks from a parcellation image.

    Args:
        parcellation_img: Parcellation NIfTI image.
        labels: Dict mapping label name → integer index.
        selected_labels: Subset of label names to include in the tissue mask.
            When ``None``, all cortical labels (``"ctx"`` prefix) are used.
        tissue_type: ``"gray"`` or ``"white"``.

    Returns:
        ``(tissue_mask, vent_mask)`` as NIfTI images.

    Raises:
        ValueError: If ``tissue_type`` is not ``"gray"`` or ``"white"``.
    """
    if tissue_type == "gray":
        # Original gray matter logic
        if selected_labels is not None:
            tissue_mask = nimage.math_img(
                " + ".join(
                    f"(x == {labels[k]})" for k in selected_labels if k in labels
                ),
                x=parcellation_img,
            )
        else:
            tissue_mask = nimage.math_img(
                " + ".join(f"(x == {v})" for k, v in labels.items() if "ctx" in k),
                x=parcellation_img,
            )

    elif tissue_type == "white":
        wm_matches = [
            (k, v)
            for k, v in labels.items()
            if any(
                [
                    ("white" in k.lower() and "matter" in k.lower()),
                    ("cerebral-white-matter" in k.lower()),
                    ("wm" in k.lower() and "cerebral" in k.lower()),
                    (v in [2, 41]),  # FreeSurfer left/right cerebral WM label IDs
                ]
            )
        ]
        wm_expr = " + ".join(f"(x == {v})" for k, v in wm_matches)
        tissue_mask = nimage.math_img(wm_expr, x=parcellation_img)

    else:
        raise ValueError(
            f"Unknown tissue_type: {tissue_type}. Must be 'gray', 'white', or 'both'"
        )

    # Erode ventricle mask to reduce partial-volume contamination
    vent_mask_raw = nimage.math_img(
        " + ".join(f"(x == {v})" for k, v in labels.items() if "vent" in k),
        x=parcellation_img,
    )
    vent_mask = nimage.new_img_like(
        parcellation_img, ndimage.binary_erosion(nimage.get_data(vent_mask_raw))
    )

    return tissue_mask, vent_mask


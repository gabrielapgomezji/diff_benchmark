import os
import json
from pathlib import Path
from xml import etree

import nibabel as nib
import nilearn as ni
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst import dti, dki
from dipy.reconst.mapmri import MapmriModel
from nilearn import image as nimage
from nilearn import maskers
from scipy import ndimage
from scipy.spatial import cKDTree
from templateflow import api as tflow
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


def read_label_file() -> dict:
    """
    Read a FreeSurfer-style label file and return a dictionary mapping
    lowercase label names to their indices.

    Args:
        filepath (str): Path to the .txt label file.

    Returns:
        dict: Dictionary with label names (lowercase) as keys and indices as values.
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
                label_name = parts[1].lower()  # convert to lowercase
                label_dict[label_name] = index
            except (ValueError, IndexError):
                continue

    return label_dict


def extract_selected_labels(nifti_path: Path, labels_dict: dict | None = None, tissue_type: str = "gray") -> dict:
    """Extract selected labels from a NIfTI file's header extensions.
    Args:
        nifti_path (Path): Path to the NIfTI file.
        labels_dict (dict | None): Optional dictionary of labels to use if extraction fails.
    Returns:
        dict: Dictionary of selected labels.
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
                k: v for k, v in labels.items() if k.startswith("ctx") or "ventricle" in k
            }
        elif tissue_type == "white":
            return {
                k: v for k, v in labels_dict.items()
                if any([
                    ("white" in k and "matter" in k),
                    ("cerebral-white-matter" in k),
                    ("wm-" in k),
                    ("ventricle" in k),
                    (v in [2, 41])
                ])
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
        # read_label_file()
        return labels_dict


def create_masks(
    parcellation_img: nib.nifti1.Nifti1Image,
    labels: dict,
    selected_labels: list | None = None,
    tissue_type: str = "gray",
) -> tuple:
    """Create context and ventricle masks from parcellation image.
    Args:
        parcellation_img (nib.Nifti1Image): Parcellation NIfTI image.
        labels (dict): Dictionary of label names to indices.
        selected_labels (list | None): Optional list of specific labels to include in context mask.
    Returns:
        tuple: Context mask and ventricle mask as NIfTI images.
    """
    if tissue_type == "gray":
        # Original gray matter logic
        if selected_labels is not None:
            tissue_mask = nimage.math_img(
                " + ".join(f"(x == {labels[k]})" for k in selected_labels if k in labels),
                x=parcellation_img,
            )
        else:
            tissue_mask = nimage.math_img(
                " + ".join(f"(x == {v})" for k, v in labels.items() if "ctx" in k),
                x=parcellation_img,
            )
    
    elif tissue_type == "white":
        # White matter mask (Left-Cerebral-White-Matter: 2, Right-Cerebral-White-Matter: 41)
        # tissue_mask = nimage.math_img(
        #     " + ".join(f"(x == {v})" for k, v in labels.items() 
        #               if "white" in k.lower() and "matter" in k.lower()),
        #     x=parcellation_img,
        # )
        wm_matches = [
            (k, v) for k, v in labels.items()
            if any([
                ("white" in k.lower() and "matter" in k.lower()),
                ("cerebral-white-matter" in k.lower()),
                ("wm" in k.lower() and "cerebral" in k.lower()),
                (v in [2, 41])  # Standard FreeSurfer left/right cerebral WM IDs
            ])
        ]
        wm_expr = " + ".join(f"(x == {v})" for k, v in wm_matches)
        tissue_mask = nimage.math_img(wm_expr, x=parcellation_img)
    
    # elif tissue_type == "both":
    #     # Combined gray + white matter
    #     ctx_expr = " + ".join(f"(x == {v})" for k, v in labels.items() if "ctx" in k)
    #     wm_expr = " + ".join(f"(x == {v})" for k, v in labels.items() 
    #                         if "white" in k.lower() and "matter" in k.lower())
    #     tissue_mask = nimage.math_img(
    #         f"({ctx_expr}) + ({wm_expr})",
    #         x=parcellation_img,
    #     )
    
    else:
        raise ValueError(f"Unknown tissue_type: {tissue_type}. Must be 'gray', 'white', or 'both'")
    
    # Ventricle mask (unchanged)
    vent_mask_raw = nimage.math_img(
        " + ".join(f"(x == {v})" for k, v in labels.items() if "vent" in k),
        x=parcellation_img,
    )
    vent_mask = nimage.new_img_like(
        parcellation_img, ndimage.binary_erosion(nimage.get_data(vent_mask_raw))
    )
    
    return tissue_mask, vent_mask


def compute_rtop(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
):
    """Compute RTOP (Radial Tensor Orientation Profile) from DWI data.
    Args:
        dwi_nib (nib.Nifti1Image): DWI NIfTI image.
        mask_img (nib.Nifti1Image): Brain mask NIfTI image.
        normalization_mask_img (nib.Nifti1Image): Normalization mask NIfTI image.
        bvals (np.ndarray): Array of b-values.
        bvecs (np.ndarray): Array of b-vectors.
        big_delta (float): Big delta value.
        small_delta (float): Small delta value.
        delta_per_bvalue (dict | None): Optional dictionary mapping b-values to delta values.
    Returns:
        nib.Nifti1Image: RTOP NIfTI image.
    """
    b0 = nimage.index_img(dwi_nib, 0)
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)

    if delta_per_bvalue is not None:
        selected_bvals = [0] + [
            k for k, v in delta_per_bvalue.items() if v == big_delta * 1000
        ]
        bvals_mask = np.any([bvals == s for s in selected_bvals], axis=0)
        dwi_data = dwi_data[bvals_mask, :]
    else:
        bvals_mask = np.ones_like(bvals, dtype=bool)

    gtab = gradient_table(
        bvals=bvals[bvals_mask],
        bvecs=bvecs[bvals_mask],
        small_delta=small_delta,
        big_delta=big_delta,
    )
    map_model = MapmriModel(
        gtab,
        radial_order=4,
        laplacian_regularization=True,
        laplacian_weighting=0.05,
        positivity_constraint=False,
    )
    rtop = map_model.fit(dwi_data.T).rtop()
    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit(b0)
        dwi_ventricles = norm_masker.transform(dwi_nib)
        if delta_per_bvalue is not None:
            dwi_ventricles = dwi_ventricles[bvals_mask, :]
        rtop_ventricles = map_model.fit(dwi_ventricles.T).rtop()

        nrtop = rtop / rtop_ventricles[~np.isnan(rtop_ventricles)].mean()
        nrtop = nrtop.clip(0, np.percentile(nrtop[~np.isnan(nrtop)], 99))
        nrtop_img = masker.inverse_transform(nrtop.T)
        return nrtop_img

    return masker.inverse_transform(rtop.T)


def compute_rtop_NEW(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
    min_directions: int = 20,
):
    """
    Compute RTOP (MAP-MRI) robustly across datasets.

    Strategy:
    1. Prefer shells with Δ == big_delta
    2. If too few directions survive, fall back to all shells
    3. Avoid silent MAP-MRI collapse and invalid normalization
    """

    # --------------------------------------------------
    # 1. Identify b0 robustly (do NOT assume index 0)
    # --------------------------------------------------
    b0_idx = np.where(bvals == 0)[0]
    if len(b0_idx) == 0:
        raise ValueError("No b0 volumes found")
    b0 = nimage.index_img(dwi_nib, int(b0_idx[0]))

    # --------------------------------------------------
    # 2. Mask DWI data
    # --------------------------------------------------
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)

    # --------------------------------------------------
    # 3. Preferred shell selection (Δ-aware)
    # --------------------------------------------------
    if delta_per_bvalue is not None:
        preferred_bvals = [
            b for b, d in delta_per_bvalue.items()
            if np.isclose(d, big_delta * 1000, atol=1)
        ]
        bvals_mask = np.isin(bvals, [0] + preferred_bvals)
        strategy = "preferred_delta"
    else:
        bvals_mask = bvals >= 0
        strategy = "all_shells"

    # Count usable diffusion directions
    n_dirs = np.sum((bvals_mask) & (bvals > 0))

    # --------------------------------------------------
    # 4. Fallback if MAP-MRI would be unstable
    # --------------------------------------------------
    if n_dirs < min_directions:
        logger.warning(
            f"RTOP fallback to all shells "
            f"(only {n_dirs} directions with preferred Δ)"
        )
        bvals_mask = bvals >= 0
        strategy = "fallback_all_shells"

    # Apply final mask
    dwi_data = dwi_data[bvals_mask, :]
    sel_bvals = bvals[bvals_mask]
    sel_bvecs = bvecs[bvals_mask]

    # --------------------------------------------------
    # 5. Gradient table
    # --------------------------------------------------
    gtab = gradient_table(
        bvals=sel_bvals,
        bvecs=sel_bvecs,
        small_delta=small_delta,
        big_delta=big_delta,
    )

    # --------------------------------------------------
    # 6. Stable MAP-MRI model
    # --------------------------------------------------
    map_model = MapmriModel(
        gtab,
        radial_order=4,                 # more stable than 6
        laplacian_regularization=True,
        laplacian_weighting=0.05,       # less aggressive
        positivity_constraint=True,     # critical for RTOP
    )

    rtop = map_model.fit(dwi_data.T).rtop()

    # --------------------------------------------------
    # 7. Optional ventricular normalization (safe)
    # --------------------------------------------------
    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit(b0)
        dwi_ventricles = norm_masker.transform(dwi_nib)[bvals_mask, :]

        rtop_vent = map_model.fit(dwi_ventricles.T).rtop()
        vent_mean = np.nanmean(rtop_vent)

        if vent_mean > 1e-6:
            rtop = rtop / vent_mean
        else:
            logger.warning("Skipping RTOP ventricular normalization")

    # --------------------------------------------------
    # 8. Clip pathological values
    # --------------------------------------------------
    valid = ~np.isnan(rtop)
    rtop[~valid] = 0
    rtop = np.clip(rtop, 0, np.percentile(rtop[valid], 99))

    logger.info(f"RTOP computed using strategy: {strategy}")

    return masker.inverse_transform(rtop.T)


def compute_md(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
):
    """Compute Mean Diffusivity (MD) from DWI data.
    Args:
        dwi_nib (nib.Nifti1Image): DWI NIfTI image.
        mask_img (nib.Nifti1Image): Brain mask NIfTI image.
        normalization_mask_img (nib.Nifti1Image): Normalization mask NIfTI image.
        bvals (np.ndarray): Array of b-values.
        bvecs (np.ndarray): Array of b-vectors.
        big_delta (float): Big delta value.
        small_delta (float): Small delta value.
        delta_per_bvalue (dict | None): Optional dictionary mapping b-values to delta values.
    Returns:
        nib.Nifti1Image: MD NIfTI image.
    """
    b0 = nimage.index_img(dwi_nib, 0)
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)
    if delta_per_bvalue is not None:
        selected_bvals = [0] + [
            k for k, v in delta_per_bvalue.items() if v == big_delta * 1000
        ]
        bvals_mask = np.any([bvals == s for s in selected_bvals], axis=0)
        dwi_data = dwi_data[bvals_mask, :]
    else:
        bvals_mask = np.ones_like(bvals, dtype=bool)

    gtab = gradient_table(
        bvals=bvals[bvals_mask],
        bvecs=bvecs[bvals_mask],
        small_delta=small_delta,
        big_delta=big_delta,
    )

    dti_model = dti.TensorModel(gtab)

    md = dti_model.fit(dwi_data.T).md

    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit(b0)
        dwi_ventricles = norm_masker.transform(dwi_nib)
        if delta_per_bvalue is not None:
            dwi_ventricles = dwi_ventricles[bvals_mask, :]
        md_ventricles = dti_model.fit(dwi_ventricles.T).md

        nmd = md / md_ventricles[~np.isnan(md_ventricles)].mean()
        nmd = nmd.clip(0, np.percentile(nmd[~np.isnan(nmd)], 99))
        nmd_img = masker.inverse_transform(nmd.T)
        return nmd_img
    logger.warning("Be careful, this is not normalized MD!")
    return masker.inverse_transform(md.T)


def compute_mk(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
):
    """Compute Mean Kurtosis (MK) from DWI data.
    Args:
        dwi_nib (nib.Nifti1Image): DWI NIfTI image.
        mask_img (nib.Nifti1Image): Brain mask NIfTI image.
        normalization_mask_img (nib.Nifti1Image): Normalization mask NIfTI image.
        bvals (np.ndarray): Array of b-values.
        bvecs (np.ndarray): Array of b-vectors.
        big_delta (float): Big delta value.
        small_delta (float): Small delta value.
        delta_per_bvalue (dict | None): Optional dictionary mapping b-values to delta values.
    Returns:
        nib.Nifti1Image: MK NIfTI image.
    """
    b0 = nimage.index_img(dwi_nib, 0)

    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)

    if delta_per_bvalue is not None:
        selected_bvals = [0] + [
            k for k, v in delta_per_bvalue.items() if v == big_delta * 1000
        ]
        # Count how many unique non-zero b-values are selected
        unique_nonzero = np.unique([b for b in selected_bvals if b > 0])
        if len(unique_nonzero) >= 2:
            # Apply filtering only if at least 2 shells are available
            bvals_mask = np.isin(bvals, selected_bvals)
            dwi_data = dwi_data[bvals_mask, :]
        else:
            # Do not filter
            bvals_mask = np.ones_like(bvals, dtype=bool)
            
    else:
        bvals_mask = np.ones_like(bvals, dtype=bool)

    gtab = gradient_table(
        bvals=bvals[bvals_mask],
        bvecs=bvecs[bvals_mask],
        small_delta=small_delta,
        big_delta=big_delta,
    )

    dki_model = dki.DiffusionKurtosisModel(gtab)
    dki_fit = dki_model.fit(dwi_data.T)

    mk = dki_fit.mk()

    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit(b0)

        dwi_ventricles = norm_masker.transform(dwi_nib)
        if delta_per_bvalue is not None:
            dwi_ventricles = dwi_ventricles[bvals_mask, :]

        mk_ventricles = dki_model.fit(dwi_ventricles.T).mk()

        nmk = mk / mk_ventricles[~np.isnan(mk_ventricles)].mean()
        nmk = nmk.clip(0, np.percentile(nmk[~np.isnan(nmk)], 99))
        return masker.inverse_transform(nmk.T)

    logger.warning("Be careful, this is not normalized MK!")
    return masker.inverse_transform(mk.T)


from dipy.reconst.shm import sf_to_sh, sh_to_rh
from dipy.core.sphere import Sphere
def compute_sh(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
    sh_order: int = 6,
):
    """
    Compute spherical harmonic representation of the raw diffusion signal.
    Returns power (L2 norm) of SH coefficients as a scalar metric.
    """
    b0 = nimage.index_img(dwi_nib, 0)
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)

    # b-value selection
    if delta_per_bvalue is not None:
        selected_bvals = [0] + [
            k for k, v in delta_per_bvalue.items() if v == big_delta * 1000
        ]
        bvals_mask = np.any([bvals == s for s in selected_bvals], axis=0)
        dwi_data = dwi_data[bvals_mask, :]
    else:
        bvals_mask = np.ones_like(bvals, dtype=bool)

    gtab = gradient_table(
        bvals=bvals[bvals_mask],
        bvecs=bvecs[bvals_mask],
        small_delta=small_delta,
        big_delta=big_delta,
    )

    # Get non-b0 gradient directions
    dwi_dirs = gtab.bvecs[gtab.bvals > 0]
    dwi_signal = dwi_data[gtab.bvals > 0, :]  # Exclude b0 from signal
    b0_data = dwi_data[0, :]
    
    # Normalize signal by b0
    dwi_normalized = dwi_signal / (b0_data + 1e-6)
    # Create Sphere object from gradient directions
    # Convert Cartesian (x,y,z) to spherical (theta, phi)
    sphere = Sphere(xyz=dwi_dirs)
    
    # Fit SH to the SIGNAL (not ODF)
    # This uses sf_to_sh which fits SH to signal values at gradient directions
    sh_coeffs = sf_to_sh(
        dwi_normalized.T,  # (n_voxels, n_gradients)
        sphere,  # Use Sphere object, not raw bvecs
        sh_order=sh_order,
        basis_type='descoteaux07'
    )
    
    # Compute power (L2 norm) - represents signal complexity/anisotropy
    scalar = np.linalg.norm(sh_coeffs, axis=1)
    scalar = np.nan_to_num(scalar)
    
    # Normalization with ventricles (questionable, but kept for consistency)
    # if normalization_mask_img is not None:
    #     norm_masker = maskers.NiftiMasker(normalization_mask_img)
    #     norm_masker.fit(b0)
    #     dwi_norm = norm_masker.transform(dwi_nib)
        
    #     if delta_per_bvalue is not None:
    #         dwi_norm = dwi_norm[bvals_mask, :]
        
    #     dwi_norm_signal = dwi_norm[gtab.bvals > 0, :]
    #     b0_norm = dwi_norm[0, :]
    #     dwi_norm_normalized = dwi_norm_signal / (b0_norm + 1e-6)
        
    #     sh_norm = sf_to_sh(
    #         dwi_norm_normalized.T,
    #         sphere,
    #         sh_order=sh_order,
    #         basis_type='descoteaux07'
    #     )
    #     scalar_norm = np.linalg.norm(sh_norm, axis=1)
        
    #     scalar = scalar / (scalar_norm.mean() + 1e-6)
    
    scalar = scalar.clip(0, np.percentile(scalar[~np.isnan(scalar)], 99))
    return masker.inverse_transform(scalar)

import requests

def download_fsl_skeleton(output_dir: Path = None) -> Path:
    """
    Download FSL's FMRIB58 FA skeleton if not available locally.
    
    Args:
        output_dir: Where to save the skeleton. Defaults to aux_materials/
    
    Returns:
        Path to the downloaded skeleton file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "aux_materials"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_file = output_dir / "FMRIB58_FA-skeleton_1mm.nii.gz"
    
    if skeleton_file.exists():
        logger.info(f"Skeleton already exists at {skeleton_file}")
        return skeleton_file
    
    # Try TemplateFlow first (most reliable)
    try:
        logger.info("Downloading white matter skeleton from TemplateFlow...")
        
        # Option 1: MNI152 skeleton (closest to FMRIB58)
        skeleton_path = tflow.get(
            'MNI152NLin2009cAsym',
            resolution=1,
            desc='brain',
            suffix='probseg',
            extension='nii.gz'
        )
        
        # Copy to our aux_materials directory
        import shutil
        shutil.copy(skeleton_path, skeleton_file)
        
        logger.info(f"Downloaded skeleton from TemplateFlow to {skeleton_file}")
        return skeleton_file
    
    except Exception as e:
        logger.warning(f"TemplateFlow download failed: {e}")
    
    # Final fallback: print manual instructions
    raise FileNotFoundError(
        f"\n{'='*80}\n"
        f"MANUAL DOWNLOAD REQUIRED\n"
        f"{'='*80}\n\n"
        f"Could not automatically download FSL skeleton.\n"
        f"Please download manually using ONE of these methods:\n\n"
        f"Method 3: Install TemplateFlow\n"
        f"  pip install templateflow\n"
        f"  python -c \"from templateflow import api as tflow; tflow.get('MNI152NLin2009cAsym', resolution=1)\"\n\n"
        f"Then place the file at: {skeleton_file}\n"
        f"{'='*80}\n"
    )
    
def get_jhu_tract_names() -> list[str]:
    """
    Return the standard JHU ICBM-DTI-81 white matter tract names.
    
    These correspond to label IDs 1-48 in the JHU-ICBM-labels-1mm.nii.gz atlas.
    Source: FSL JHU atlas documentation
    https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/blob/FinalFive/JHU-labels.xml
    
    Returns:
        list[str]: 48 tract names in order (label 1-48)
    """
    return [
        "Middle cerebellar peduncle",
        "Pontine crossing tract (a part of MCP)",
        "Genu of corpus callosum",
        "Body of corpus callosum",
        "Splenium of corpus callosum",
        "Fornix (column and body of fornix)",
        "Corticospinal tract R",
        "Corticospinal tract L",
        "Medial lemniscus R",
        "Medial lemniscus L",
        "Inferior cerebellar peduncle R",
        "Inferior cerebellar peduncle L",
        "Superior cerebellar peduncle R",
        "Superior cerebellar peduncle L",
        "Cerebral peduncle R",
        "Cerebral peduncle L",
        "Anterior limb of internal capsule R",
        "Anterior limb of internal capsule L",
        "Posterior limb of internal capsule R",
        "Posterior limb of internal capsule L",
        "Retrolenticular part of internal capsule R",
        "Retrolenticular part of internal capsule L",
        "Anterior corona radiata R",
        "Anterior corona radiata L",
        "Superior corona radiata R",
        "Superior corona radiata L",
        "Posterior corona radiata R",
        "Posterior corona radiata L",
        "Posterior thalamic radiation (include optic radiation) R",
        "Posterior thalamic radiation (include optic radiation) L",
        "Sagittal stratum (include inferior longitidinal fasciculus and inferior fronto-occipital fasciculus) R",
        "Sagittal stratum (include inferior longitidinal fasciculus and inferior fronto-occipital fasciculus) L",
        "External capsule R",
        "External capsule L",
        "Cingulum (cingulate gyrus) R",
        "Cingulum (cingulate gyrus) L",
        "Cingulum (hippocampus) R",
        "Cingulum (hippocampus) L",
        "Fornix (cres) / Stria terminalis (can not be resolved with current resolution) R",
        "Fornix (cres) / Stria terminalis (can not be resolved with current resolution) L",
        "Superior longitudinal fasciculus R",
        "Superior longitudinal fasciculus L",
        "Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) R",
        "Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) L",
        "Uncinate fasciculus R",
        "Uncinate fasciculus L",
        "Tapetum R",
        "Tapetum L",
    ]
    
def load_tbss_skeleton(
    skeleton_path: Path | None = None,
    template: str = "fmrib",
) -> tuple:
    """
    Load standard TBSS skeleton and tract atlas.
    
    Args:
        skeleton_path: Path to existing skeleton file. If None, will try to find/download.
        template (str): Which skeleton template to use:
            - "fmrib": FSL's FMRIB58_FA standard space (most common)
            - "mni": MNI152 space (alternative)
        atlas (str): Which white matter atlas to use:
            - "jhu": JHU ICBM-DTI-81 (48 tracts, most common)
            - "jhu-labels": JHU white matter tractography atlas (20 tracts)
            - "aal": AAL atlas white matter regions
    
    Returns:
        tuple: (skeleton_mask, tract_labels, tract_names)
    """
    from nilearn import datasets
    
    # Step 1: Load or download skeleton
    if skeleton_path is None:
        # Check FSL installation first
        fsl_dir = Path(__file__).parent.parent.parent.parent / "aux_materials"
        skeleton_path = fsl_dir / "FMRIB58_FA-skeleton_1mm.nii.gz"
        
        if not skeleton_path.exists():
            # Download to aux_materials
            logger.warning("FSL skeleton not found locally, downloading...")
            skeleton_path = download_fsl_skeleton()
    
    skeleton_mask = nib.load(skeleton_path)
    logger.info(f"Loaded TBSS skeleton from {skeleton_path}")
    
    # Step 2: Load white matter atlas
    # JHU ICBM-DTI-81 atlas (48 tracts)
    jhu_labels_file = fsl_dir / "JHU-ICBM-labels-1mm.nii.gz"
    tract_labels_img = nib.load(jhu_labels_file)
    # jhu_data = datasets.fetch_atlas_jhu()
    # tract_labels_img = nib.load(jhu_data['maps'])
    tract_names = get_jhu_tract_names()
    logger.info(f"Loaded JHU ICBM-DTI-81 atlas with {len(tract_names)} tracts")
    
    # Step 3: Resample atlas to skeleton space
    tract_labels_resampled = nimage.resample_to_img(
        tract_labels_img, 
        skeleton_mask, 
        interpolation='nearest'
    )
    
    # Step 4: Mask tract labels to only skeleton voxels
    skeleton_data = skeleton_mask.get_fdata() > 0.2  # Threshold for skeleton
    tract_data = tract_labels_resampled.get_fdata()
    tract_labels_on_skeleton = tract_data * skeleton_data
    
    # Convert to NIfTI image
    tract_labels_skeleton_img = nimage.new_img_like(
        skeleton_mask,
        tract_labels_on_skeleton
    )
    
    return skeleton_mask, tract_labels_skeleton_img, tract_names

def classify_tract_hemisphere(tract_name: str) -> str:
    """
    Classify JHU tract as left ('L'), right ('R'), or midline ('M').
    
    Args:
        tract_name: Name from JHU atlas (e.g., "Anterior thalamic radiation L")
    
    Returns:
        str: 'L', 'R', or 'M'
    """
    tract_lower = tract_name.lower()
    
    # Explicit hemisphere indicators
    if tract_lower.endswith(' l') or 'left' in tract_lower:
        return 'L'
    elif tract_lower.endswith(' r') or 'right' in tract_lower:
        return 'R'
    
    # Midline structures
    midline_keywords = [
        'corpus callosum',
        'genu of corpus callosum',
        'body of corpus callosum',
        'splenium of corpus callosum',
        'fornix',
        'middle cerebellar peduncle',
    ]
    
    if any(keyword in tract_lower for keyword in midline_keywords):
        return 'M'
    
    # Default: treat as midline if unclear
    logger.warning(f"Could not classify hemisphere for tract: {tract_name}, treating as midline")
    return 'M'
        
def project_to_skeleton(
    metric_img: nib.nifti1.Nifti1Image,
    output_dir: Path,
    subject_id: str,
    metric_name: str,
) -> tuple[nib.nifti1.Nifti1Image, np.ndarray]:
    """
    Project volumetric metric onto TBSS skeleton and extract tract-level scalars.
    Saves hemisphere-split .scalar.gii files matching gray matter format.
    
    Args:
        metric_img: Subject's metric (RTOP, MD, etc.) in subject's native space
        skeleton_mask: Standard TBSS skeleton (in MNI space)
        tract_labels: JHU tract labels on skeleton
        tract_names: List of tract names from JHU atlas
        output_dir: Where to save outputs
        subject_id: Subject ID
        metric_name: Metric name (rtop, md, etc.)
        register_to_standard: If True, resample metric to MNI space
        save_skeleton_image: If True, save full skeleton NIfTI
        save_as_gifti: If True, save tract scalars as .scalar.gii
        split_hemispheres: If True, split by L/R/M hemispheres
    
    Returns:
        tuple: (skeleton_img, tract_scalars_all)
    """
    logger.info(f"[{subject_id}] Projecting {metric_name} onto TBSS skeleton")
    skeleton_mask, tract_labels_skeleton_img, tract_names = load_tbss_skeleton()
    
    # Step 1: Resample metric from subject space to MNI/skeleton space
    logger.info(f"[{subject_id}] Transforming skeleton from MNI to subject space")
        
    # Resample skeleton mask to subject's metric space
    skeleton_mask_subject = nimage.resample_to_img(
        skeleton_mask,
        metric_img,
        interpolation='linear'  # For mask values
    )
    
    # Resample tract labels to subject's metric space
    tract_labels_subject = nimage.resample_to_img(
        tract_labels_skeleton_img,
        metric_img,
        interpolation='nearest'  # IMPORTANT: nearest for labels!
    )
    
    # Use subject-space data directly
    metric_data = metric_img.get_fdata()
    skeleton_data = skeleton_mask_subject.get_fdata() > 0.2
    tract_data = tract_labels_subject.get_fdata()
    
    # Step 2: Apply skeleton mask
    metric_on_skeleton = metric_data * skeleton_data

    left_scalars = []
    right_scalars = []
    midline_scalars = []
    left_names = []
    right_names = []
    midline_names = []
    
    for tract_id, tract_name in enumerate(tract_names, start=1):
        tract_mask = (tract_data == tract_id) & skeleton_data
        
        if tract_mask.sum() > 0:
            value = np.nanmean(metric_on_skeleton[tract_mask])
        else:
            value = np.nan
        
        # Classify by hemisphere
        hemi = classify_tract_hemisphere(tract_name)
        
        if hemi == 'L':
            left_scalars.append(value)
            left_names.append(tract_name)
        elif hemi == 'R':
            right_scalars.append(value)
            right_names.append(tract_name)
        else:  # 'M'
            midline_scalars.append(value)
            midline_names.append(tract_name)
    
    # Convert to numpy arrays
    left_scalars = np.array(left_scalars, dtype=np.float32)
    right_scalars = np.array(right_scalars, dtype=np.float32)
    midline_scalars = np.array(midline_scalars, dtype=np.float32)

    # Left hemisphere
    if len(left_scalars) > 0:
        left_gii = nib.gifti.gifti.GiftiImage()
        left_gii.add_gifti_data_array(
            nib.gifti.gifti.GiftiDataArray(
                left_scalars,
                intent="NIFTI_INTENT_DIMLESS",
            )
        )
        left_file = output_dir / f"sub-{subject_id}_hemi-L_param-{metric_name}_tissue-white.scalar.gii"
        nib.save(left_gii, left_file)
        logger.info(f"[{subject_id}] Saved {len(left_scalars)} LEFT WM tracts to {left_file}")
    
    # Right hemisphere
    if len(right_scalars) > 0:
        right_gii = nib.gifti.gifti.GiftiImage()
        right_gii.add_gifti_data_array(
            nib.gifti.gifti.GiftiDataArray(
                right_scalars,
                intent="NIFTI_INTENT_DIMLESS",
            )
        )
        right_file = output_dir / f"sub-{subject_id}_hemi-R_param-{metric_name}_tissue-white.scalar.gii"
        nib.save(right_gii, right_file)
        logger.info(f"[{subject_id}] Saved {len(right_scalars)} RIGHT WM tracts to {right_file}")
    
    # Midline tracts
    if len(midline_scalars) > 0:
        midline_gii = nib.gifti.gifti.GiftiImage()
        midline_gii.add_gifti_data_array(
            nib.gifti.gifti.GiftiDataArray(
                midline_scalars,
                intent="NIFTI_INTENT_DIMLESS",
            )
        )
        midline_file = output_dir / f"sub-{subject_id}_hemi-M_param-{metric_name}_tissue-white.scalar.gii"
        nib.save(midline_gii, midline_file)
        logger.info(f"[{subject_id}] Saved {len(midline_scalars)} MIDLINE WM tracts to {midline_file}")

    # Concatenate all for return
    tract_scalars_all = np.concatenate([left_scalars, right_scalars, midline_scalars])

    # skeleton_img = nimage.new_img_like(metric_img, metric_on_skeleton)
    
    return None

def extract_wm_tract_subset(
    left_tracts: np.ndarray,
    right_tracts: np.ndarray,
    midline_tracts: np.ndarray,
    tract_names: list[str] | None = None,
    target_tracts: list[str] | None = None,
) -> np.ndarray:
    """
    Extract a subset of white matter tracts by name pattern.
    
    This function allows regional analysis of white matter by selecting specific
    tract groups (e.g., all "corona radiata", all "internal capsule", etc.).
    
    Args:
        left_tracts (np.ndarray): Values for left hemisphere tracts
        right_tracts (np.ndarray): Values for right hemisphere tracts  
        midline_tracts (np.ndarray): Values for midline tracts
        tract_names (list[str] | None): Full JHU tract names. If None, loads default.
        target_tracts (list[str] | None): List of tract name substrings to include.
            Examples: ["corona radiata", "internal capsule", "corticospinal"]
            If None, returns all tracts.
    
    Returns:
        np.ndarray: Concatenated values for selected tracts only
    """
    if tract_names is None:
        tract_names = get_jhu_tract_names()
    
    # Create tract indices by hemisphere
    left_indices = []
    right_indices = []
    midline_indices = []
    
    for i, name in enumerate(tract_names):
        hemi = classify_tract_hemisphere(name)
        if hemi == 'L':
            left_indices.append(len(left_indices))
        elif hemi == 'R':
            right_indices.append(len(right_indices))
        else:
            midline_indices.append(len(midline_indices))
    
    # If no target specified, return all
    if target_tracts is None:
        return np.concatenate([left_tracts, right_tracts, midline_tracts])
    
    # Filter by target tract names
    selected_left = []
    selected_right = []
    selected_midline = []
    
    left_counter = 0
    right_counter = 0
    midline_counter = 0
    
    for i, name in enumerate(tract_names):
        # Check if any target substring matches this tract
        matches = any(target.lower() in name.lower() for target in target_tracts)
        
        if matches:
            hemi = classify_tract_hemisphere(name)
            if hemi == 'L':
                selected_left.append(left_tracts[left_counter])
                left_counter += 1
            elif hemi == 'R':
                selected_right.append(right_tracts[right_counter])
                right_counter += 1
            else:
                selected_midline.append(midline_tracts[midline_counter])
                midline_counter += 1
        else:
            # Still need to increment counters
            hemi = classify_tract_hemisphere(name)
            if hemi == 'L':
                left_counter += 1
            elif hemi == 'R':
                right_counter += 1
            else:
                midline_counter += 1
    
    # Concatenate selected tracts
    all_selected = []
    if selected_left:
        all_selected.append(np.array(selected_left))
    if selected_right:
        all_selected.append(np.array(selected_right))
    if selected_midline:
        all_selected.append(np.array(selected_midline))
    
    if not all_selected:
        logger.warning(f"No tracts matched target patterns: {target_tracts}")
        return np.array([])
    
    return np.concatenate(all_selected)
       
def project_to_surface(
    micr_img: nib.nifti1.Nifti1Image,
    ctx_mask: nib.nifti1.Nifti1Image,
    surfaces: dict,
    output_dir: Path,
    subject_id: str,
    micr_metric: str,
    layouts: list = None,
    target_space: str = "fslr_32k",
    data_reading: str = "hcp",
    tissue_type: str = "gray",
):
    """
    Project image onto surface meshes and save as GIFTI files.
    For BIDS datasets, automatically resamples from native space to template space.
    
    Args:
        micr_img (nib.Nifti1Image): NIfTI image with microstructure values.
        ctx_mask (nib.Nifti1Image): Context mask NIfTI image.
        surfaces (dict): Dictionary with keys 'L.pial', 'L.white', 'R.pial', 'R.white' and corresponding surface file paths.
        output_dir (Path): Directory to save output GIFTI files.
        subject_id (str): Subject identifier for naming output files.
        micr_metric (str): Metric name for naming output files.
        layouts (list): List of BIDS layouts (needed for BIDS datasets to find sphere files).
        target_space (str): Target surface space for resampling (default: "fslr_32k").
        data_reading (str): Dataset format ("hcp", "bids", "multicenter-bids").
        tissue_type (str): Type of tissue ("gray" or "white").
    Returns:
        None
    """
    if tissue_type == "white":
        logger.info(f"[{subject_id}] Skipping surface projection for white matter")
        
        # Project white matter onto skeleton
        logger.info(f"[{subject_id}] Projecting white matter to skeleton")
        project_to_skeleton(
            micr_img,
            output_dir,
            subject_id,
            micr_metric,
        )
        return None
    
    # First, project to surfaces (native space for BIDS, template space for HCP)
    left_data = None
    right_data = None
    
    for h in ("L", "R"):
        insula_surf = ni.surface.vol_to_surf(
            micr_img,
            surfaces[f"{h}.pial"],
            mask_img=ctx_mask,
            inner_mesh=surfaces[f"{h}.white"],
            depth=[0.1, 0.5, 0.9],
        )
        
        if h == "L":
            left_data = insula_surf
        else:
            right_data = insula_surf
    
    # For BIDS datasets, resample from native space to template space
    if "bids" in data_reading and layouts is not None:
        try:
            logger.info(f"[{subject_id}] Resampling surface data from native to {target_space} space")
            left_data, right_data = resample_subject_to_template(
                subject_id=subject_id,
                left_data=left_data,
                right_data=right_data,
                layouts=layouts,
                target_space=target_space,
            )
        except Exception as e:
            logger.warning(f"[{subject_id}] Resampling failed, saving native space data: {e}")
    
    # Save the data (resampled for BIDS, native/template for HCP)
    for h, data in [("L", left_data), ("R", right_data)]:
        img = nib.gifti.gifti.GiftiImage()
        img.add_gifti_data_array(
            nib.gifti.gifti.GiftiDataArray(
                data.astype(np.float32),
                intent="NIFTI_INTENT_DIMLESS",
            )
        )
        nib.save(
            img,
            output_dir / f"sub-{subject_id}_hemi-{h}_param-{micr_metric}_tissue-{tissue_type}.scalar.gii",
        )
    
    # Return None for gray matter (no additional image to save)
    return None


def resample_subject_to_template(
    subject_id: str,
    left_data: np.ndarray,
    right_data: np.ndarray,
    layouts: list,
    target_space: str = "fslr_32k",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample subject's native surface data to a template space using sphere mapping.
    
    Args:
        subject_id (str): Subject identifier
        left_data (np.ndarray): Data on left hemisphere in subject's native space
        right_data (np.ndarray): Data on right hemisphere in subject's native space
        layouts (list): List of BIDS layouts to search for subject data
        target_space (str): Target template space ("fslr_32k" or "fsaverage")
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Resampled left and right hemisphere data
    """
    # Find the layout containing this subject
    layout = None
    for lay in layouts:
        if subject_id in lay.get_subjects():
            layout = lay
            break
    
    if layout is None:
        raise ValueError(f"Subject {subject_id} not found in any layout")
    
    resampled_data = {}
    
    for hemi, data in [("L", left_data), ("R", right_data)]:
        # Get subject's native sphere
        subject_sphere_files = layout.get(
            subject=subject_id,
            suffix='sphere',
            hemi=hemi,
            extension=".surf.gii",
            space='fsLR',
            return_type='files'
        )
        
        if not subject_sphere_files:
            raise FileNotFoundError(
                f"No sphere surface found for subject {subject_id}, hemisphere {hemi}"
            )
        
        subject_sphere = nib.load(subject_sphere_files[0])
        
        # Get template sphere based on target_space
        if target_space == "fslr_32k":
            template_sphere_fn = tflow.get(
                "fsLR", hemi=hemi, density="32k", suffix="sphere", desc=None, space=None
            )
        elif target_space == "fsaverage":
            template_sphere_fn = tflow.get(
                "fsaverage", hemi=hemi, density="164k", suffix="sphere", desc=None
            )
        else:
            raise ValueError(f"Unknown target_space: {target_space}")
        
        template_sphere = nib.load(template_sphere_fn)
        
        # Build KD-trees for nearest-neighbor mapping
        kdtree_subject = cKDTree(subject_sphere.darrays[0].data)
        kdtree_template = cKDTree(template_sphere.darrays[0].data)
        
        # Find nearest neighbors from subject to template
        subject_to_template = kdtree_subject.query(template_sphere.darrays[0].data, k=1)
        
        # Resample data: for each template vertex, use nearest subject vertex
        resampled_data[hemi] = data[subject_to_template[1]]
    
    return resampled_data["L"], resampled_data["R"]


def resample_schaefer_onto_fs_lr(scale: int = 1000, target_space: str = "fslr_32k") -> dict:
    """Resample Schaefer 2018 parcellation onto fsLR or fsaverage space.
    Args:
        scale (int): Scale of Schaefer parcellation (e.g., 1000 for 1000 parcels).
        target_space (str): Target surface space. Options:
            - "fslr_32k": fsLR 32k space (HCP default)
            - "fsaverage": Native fsaverage space (FreeSurfer/CamCAN)
    Returns:
        dict: Dictionary with keys 'left.data', 'left.labels', 'left.sulc',
              'right.data', 'right.labels', 'right.sulc'.
    """
    fsaverage_left_schaefer_fn = tflow.get(
        "fsaverage",
        hemi="L",
        density="164k",
        atlas="Schaefer2018",
        segmentation="17n",
        scale=str(scale),
        extension="label.gii",
    )
    fsaverage_left_schaefer = nib.load(fsaverage_left_schaefer_fn)

    fsaverage_right_schaefer_fn = tflow.get(
        "fsaverage",
        hemi="R",
        density="164k",
        atlas="Schaefer2018",
        segmentation="17n",
        scale=str(scale),
        extension="label.gii",
    )
    fsaverage_right_schaefer = nib.load(fsaverage_right_schaefer_fn)

    labels = pd.read_csv(
        tflow.get(
            "fsaverage",
            atlas="Schaefer2018",
            segmentation="17n",
            scale=str(scale),
            extension="tsv",
        ),
        sep="\t",
    )

    labels_left = labels[labels["hemi"] == "L"]
    labels_right = labels[labels["hemi"] == "R"]

    # If target is fsaverage, return the atlas directly without resampling
    if target_space == "fsaverage":
        fslr_left_sulc = nib.load(
            tflow.get("fsaverage", hemi="L", density="164k", suffix="sulc", desc=None)
        ).darrays[0].data
        
        fslr_right_sulc = nib.load(
            tflow.get("fsaverage", hemi="R", density="164k", suffix="sulc", desc=None)
        ).darrays[0].data
        
        return {
            "left.data": fsaverage_left_schaefer.darrays[0].data,
            "left.labels": labels_left,
            "left.sulc": fslr_left_sulc,
            "right.data": fsaverage_right_schaefer.darrays[0].data,
            "right.labels": labels_right,
            "right.sulc": fslr_right_sulc,
        }

    # Otherwise, resample to fsLR 32k (original HCP behavior)
    fslr_2_fsaverage_left_sphere_fn = tflow.get(
        "fsLR", hemi="L", density="32k", space="fsaverage"
    )
    fsaverage_sphere_left_fn = tflow.get(
        "fsaverage", hemi="L", density="164k", suffix="sphere", desc=None
    )
    fslr_2_fsaverage_left_sphere = nib.load(fslr_2_fsaverage_left_sphere_fn)
    fsaverage_sphere_left = nib.load(fsaverage_sphere_left_fn)
    kdtree_fsaverage_left = cKDTree(fsaverage_sphere_left.darrays[0].data)
    kdtree_fslr_left = cKDTree(fslr_2_fsaverage_left_sphere.darrays[0].data)
    fslr_to_fsaverage_left_sphere = kdtree_fslr_left.query(
        fsaverage_sphere_left.darrays[0].data, k=1
    )
    fsaverage_to_fslr_left_sphere = kdtree_fsaverage_left.query(
        fslr_2_fsaverage_left_sphere.darrays[0].data, k=1
    )

    fslr_2_fsaverage_right_sphere_fn = tflow.get(
        "fsLR", hemi="R", density="32k", space="fsaverage"
    )
    fsaverage_sphere_right_fn = tflow.get(
        "fsaverage", hemi="R", density="164k", suffix="sphere", desc=None
    )
    fslr_2_fsaverage_right_sphere = nib.load(fslr_2_fsaverage_right_sphere_fn)
    fsaverage_sphere_right = nib.load(fsaverage_sphere_right_fn)
    kdtree_fsaverage_right = cKDTree(fsaverage_sphere_right.darrays[0].data)
    kdtree_fslr_right = cKDTree(fslr_2_fsaverage_right_sphere.darrays[0].data)
    fslr_to_fsaverage_right_sphere = kdtree_fslr_right.query(
        fsaverage_sphere_right.darrays[0].data, k=1
    )
    fsaverage_to_fslr_right_sphere = kdtree_fsaverage_right.query(
        fslr_2_fsaverage_right_sphere.darrays[0].data, k=1
    )

    fslr_left_schaefer = np.zeros(len(fsaverage_to_fslr_left_sphere[1]))
    fslr_left_schaefer[fslr_to_fsaverage_left_sphere[1]] = (
        fsaverage_left_schaefer.darrays[0].data
    )

    fslr_right_schaefer = np.zeros(len(fsaverage_to_fslr_right_sphere[1]))
    fslr_right_schaefer[fslr_to_fsaverage_right_sphere[1]] = (
        fsaverage_right_schaefer.darrays[0].data
    )

    fslr_left_sulc = np.zeros(len(fsaverage_to_fslr_left_sphere[1]))
    fslr_left_sulc[fslr_to_fsaverage_left_sphere[1]] = (
        nib.load(
            tflow.get("fsaverage", hemi="L", density="164k", suffix="sulc", desc=None)
        )
        .darrays[0]
        .data
    )

    fslr_right_sulc = np.zeros(len(fsaverage_to_fslr_right_sphere[1]))
    fslr_right_sulc[fslr_to_fsaverage_right_sphere[1]] = (
        nib.load(
            tflow.get("fsaverage", hemi="R", density="164k", suffix="sulc", desc=None)
        )
        .darrays[0]
        .data
    )

    return {
        "left.data": fslr_left_schaefer,
        "left.labels": labels_left,
        "left.sulc": fslr_left_sulc,
        "right.data": fslr_right_schaefer,
        "right.labels": labels_right,
        "right.sulc": fslr_right_sulc,
    }

def average_per_parcel(
    hem_left: np.ndarray, hem_right: np.ndarray, schaefer_resampled: dict
) -> np.ndarray:
    """
    Average RTOP values across parcels in both hemispheres.
    hem_left: RTOP/MD/microstructure values for left hemisphere
    hem_right: RTOP/MD/microstructure values for right hemisphere
    schaefer_resampled: dict with 'left.data' and 'right.data' arrays
    Returns:
        np.ndarray: Mean RTOP values per parcel (concatenated left and right).
    """
    parcellation_left = schaefer_resampled["left.data"]
    parcellation_right = schaefer_resampled["right.data"]
    # breakpoint()
    parcels_left = np.unique(parcellation_left)
    parcels_right = np.unique(parcellation_right)

    # Initialize array to store mean values per parcel
    rtop_avg = np.zeros(len(parcels_left) + len(parcels_right))

    for i, parcel in enumerate(sorted(parcels_left)):
        mask = parcellation_left == parcel
        rtop_avg[i] = hem_left[mask].mean()
    for i, parcel in enumerate(sorted(parcels_right), start=len(parcels_left)):
        mask = parcellation_right == parcel
        rtop_avg[i] = hem_right[mask].mean()
    return rtop_avg


def extract_region_data(
    hem_left: np.ndarray,
    hem_right: np.ndarray,
    schaefer_resampled: dict,
    target_substring: str | None = None,
    average: bool = False,
):
    """
    Average microstructure values across selected parcels in both hemispheres.

    Args:
        hem_left (ndarray): Microstructure values for left hemisphere.
        hem_right (ndarray): Microstructure values for right hemisphere.
        schaefer_resampled (dict): Contains 'left.data' and 'right.data' arrays.
        region_ids (list[int] or None): Optional list of parcel IDs to include.
            If None, all parcels will be included.
        average (bool): If True, return the overall average across selected regions.
    Returns:
        np.ndarray: Mean values per selected region (in order of selection if provided).
    """
    parc_left = schaefer_resampled["left.data"]
    parc_right = schaefer_resampled["right.data"]
    labels_left = schaefer_resampled["left.labels"].copy()
    labels_right = schaefer_resampled["right.labels"].copy()

    unique_left = np.unique(parc_left)
    unique_right = np.unique(parc_right)

    labels_left["array_index"] = unique_left[: len(labels_left)]
    labels_right["array_index"] = unique_right[: len(labels_right)]
    labels_left["hemi"] = "L"
    labels_right["hemi"] = "R"
    all_labels = pd.concat([labels_left, labels_right], ignore_index=True)

    if target_substring:
        matched_labels = all_labels[
            all_labels["name"].str.contains(target_substring, case=False, na=False)
        ]
    else:
        matched_labels = all_labels

    region_data = {}
    region_values = []

    # --- Iterate through matched regions (each has unique hemi + name)
    for _, row in matched_labels.iterrows():
        hemi = row["hemi"]
        region_id = row["array_index"]
        region_name = row["name"]
        if hemi == "L":
            mask = parc_left == region_id
            vals = hem_left[mask]
            region_values.append(vals)
        elif hemi == "R":
            mask = parc_right == region_id
            vals = hem_right[mask]
            region_values.append(vals)
        else:
            continue  # unexpected hemi label

        if vals.size == 0:
            continue  # skip empty

        region_data[region_name] = np.nanmean(vals) if average else vals

    region_values = np.concatenate([np.atleast_1d(v) for v in region_values])
    # return region_data # returns dict and csv is a column per region with the corresponding arrays
    return region_values


def compute_b0(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
    b0_threshold: float = 50.0,
) -> nib.nifti1.Nifti1Image:
    """Compute a T2-weighted baseline image by averaging all b0 (low-b) volumes.

    This provides a baseline "b0 metric" that captures T2-weighted tissue
    contrast without any diffusion weighting. It can be used as a lower bound
    in the benchmark to assess how much information is contained in the raw
    diffusion signal beyond what is already encoded in T2.

    The function signature deliberately mirrors all other ``compute_*``
    functions so it plugs into ``compute_save_and_project_metric`` and
    ``METRIC_COMPUTERS`` without any pipeline changes.

    Args:
        dwi_nib (nib.Nifti1Image): Full DWI 4-D NIfTI image.
        mask_img (nib.Nifti1Image): Brain / tissue mask NIfTI image.
        normalization_mask_img (nib.Nifti1Image): Ventricular mask used for
            intensity normalization (same convention as MD/RTOP). Pass ``None``
            to skip normalization.
        bvals (np.ndarray): 1-D array of b-values (one per volume).
        bvecs (np.ndarray): 2-D array of b-vectors ``(N, 3)``; not used here
            but kept for API consistency.
        big_delta (float): Not used; kept for API consistency.
        small_delta (float): Not used; kept for API consistency.
        delta_per_bvalue (dict | None): Not used; kept for API consistency.
        b0_threshold (float): Volumes with ``bval <= b0_threshold`` are treated
            as b0. Defaults to 50 s/mm², which safely captures nominally-zero
            b-values that are stored as 5 or similar small numbers.

    Returns:
        nib.Nifti1Image: 3-D NIfTI image of the averaged (and optionally
            ventricular-normalised) b0 signal, masked to ``mask_img``.

    Raises:
        ValueError: If no b0 volumes are found within ``b0_threshold``.
    """
    # ------------------------------------------------------------------ #
    # 1. Identify b0 indices                                              #
    # ------------------------------------------------------------------ #
    b0_indices = np.where(bvals <= b0_threshold)[0]
    if len(b0_indices) == 0:
        raise ValueError(
            f"No b0 volumes found with bval <= {b0_threshold}. "
            f"Unique bvals: {np.unique(np.round(bvals)).tolist()}"
        )

    logger.info(
        f"compute_b0: found {len(b0_indices)} b0 volume(s) "
        f"(bval <= {b0_threshold}): indices {b0_indices.tolist()}"
    )

    # Use the first b0 as the reference for masking (consistent with other funcs)
    ref_b0 = nimage.index_img(dwi_nib, int(b0_indices[0]))

    # ------------------------------------------------------------------ #
    # 2. Apply brain mask                                                 #
    # ------------------------------------------------------------------ #
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(ref_b0)

    # Extract only b0 volumes → shape (n_b0, n_voxels)
    b0_imgs = [nimage.index_img(dwi_nib, int(i)) for i in b0_indices]
    b0_data = np.stack(
        [masker.transform(img).squeeze() for img in b0_imgs], axis=0
    )  # (n_b0, n_voxels)

    # ------------------------------------------------------------------ #
    # 3. Average across b0 volumes                                        #
    # ------------------------------------------------------------------ #
    mean_b0 = np.mean(b0_data, axis=0)  # (n_voxels,)
    mean_b0 = np.nan_to_num(mean_b0, nan=0.0)

    # ------------------------------------------------------------------ #
    # 4. Optional ventricular normalisation (same convention as MD/RTOP) #
    # ------------------------------------------------------------------ #
    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit(ref_b0)

        vent_b0_data = np.stack(
            [norm_masker.transform(img).squeeze() for img in b0_imgs], axis=0
        )
        mean_vent_b0 = np.mean(vent_b0_data, axis=0)
        vent_mean = np.nanmean(mean_vent_b0)

        if vent_mean > 1e-6:
            mean_b0 = mean_b0 / vent_mean
            logger.info(f"compute_b0: ventricular normalisation applied (vent_mean={vent_mean:.4f})")
        else:
            logger.warning("compute_b0: skipping ventricular normalisation (vent_mean ~ 0)")
    else:
        logger.warning(
            "compute_b0: no ventricular mask provided, returning un-normalised b0 signal."
        )

    # ------------------------------------------------------------------ #
    # 5. Clip to 99th percentile to remove outlier voxels                #
    # ------------------------------------------------------------------ #
    valid = ~np.isnan(mean_b0)
    if valid.any():
        mean_b0 = mean_b0.clip(0, np.percentile(mean_b0[valid], 99))

    return masker.inverse_transform(mean_b0)


METRIC_COMPUTERS = {
    "rtop": compute_rtop,
    "md": compute_md,
    "mk": compute_mk,
    "sh": compute_sh,
    "b0": compute_b0,
}


def compute_save_and_project_metric(
    *,
    metric: str,
    dwi_nib: nib.nifti1.Nifti1Image,
    ctx_mask: nib.nifti1.Nifti1Image,
    vent_mask: nib.nifti1.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    big_delta_per_bvalue: float,
    surfaces: dict,
    derivatives_dir: Path,
    subject_id: str,
    layouts: list = None,
    target_space: str = "fslr_32k",
    data_reading: str = "hcp",
    tissue_type: str = "gray",
) -> nib.nifti1.Nifti1Image:
    """
    Computes a specified diffusion metric, saves the resulting image to disk,
    and projects the metric onto cortical surfaces.
    
    For BIDS datasets, automatically resamples surface data from native space 
    to template space during projection.
    
    Parameters:
        metric (str): The name of the diffusion metric to compute. Must be a key in `METRIC_COMPUTERS`.
        dwi_nib (nib.nifti1.Nifti1Image): The diffusion-weighted imaging (DWI) data as a NIfTI image.
        ctx_mask (nib.nifti1.Nifti1Image): The cortical mask as a NIfTI image.
        vent_mask (nib.nifti1.Nifti1Image): The ventricular mask as a NIfTI image.
        bvals (np.ndarray): Array of b-values corresponding to the DWI data.
        bvecs (np.ndarray): Array of b-vectors corresponding to the DWI data.
        big_delta (float): The big delta parameter for the diffusion metric computation.
        small_delta (float): The small delta parameter for the diffusion metric computation.
        big_delta_per_bvalue (float): The big delta per b-value for the diffusion metric computation.
        surfaces (dict): A dictionary containing cortical surface data for projection.
        derivatives_dir (Path): Directory where the computed metric image will be saved.
        subject_id (str): Identifier for the subject being processed.
        layouts (list): List of BIDS layouts (needed for BIDS datasets to find sphere files).
        target_space (str): Target surface space for resampling (default: "fslr_32k").
        data_reading (str): Dataset format ("hcp", "bids", "multicenter-bids").
    Returns:
        nib.nifti1.Nifti1Image: The computed diffusion metric as a NIfTI image.
    Raises:
        ValueError: If the specified metric is not found in `METRIC_COMPUTERS`.
    Notes:
        - The computed metric image is saved to the `derivatives_dir` with a filename
          formatted as `sub-{subject_id}_param-{metric}_dwimap.nii.gz`.
        - For BIDS datasets, the metric is projected to surfaces in native space, 
          then automatically resampled to template space before saving.
        - For HCP datasets, no resampling is needed (data already in template space).
    """
    if metric not in METRIC_COMPUTERS:
        raise ValueError(f"Unknown metric: {metric}")

    compute_fn = METRIC_COMPUTERS[metric]

    metric_img = compute_fn(
        dwi_nib,
        ctx_mask,
        vent_mask,
        bvals,
        bvecs,
        big_delta,
        small_delta,
        big_delta_per_bvalue,
    )

    out_file = derivatives_dir / f"sub-{subject_id}_param-{metric}_tissue-{tissue_type}_dwimap.nii.gz"
    nib.save(metric_img, out_file)
    logger.info(f"[{subject_id}] Saved raw {metric} image to {out_file}")

    # Step 3: Project to surface/skeleton and save hemisphere scalars
    _ = project_to_surface(
        metric_img,
        ctx_mask,
        surfaces,
        derivatives_dir,
        subject_id,
        metric,
        layouts=layouts,
        target_space=target_space,
        data_reading=data_reading,
        tissue_type=tissue_type,
    )

    return metric_img

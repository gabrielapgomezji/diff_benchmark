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


def extract_selected_labels(nifti_path: Path, labels_dict: dict | None = None) -> dict:
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
        return {
            k: v for k, v in labels.items() if k.startswith("ctx") or "ventricle" in k
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
) -> tuple:
    """Create context and ventricle masks from parcellation image.
    Args:
        parcellation_img (nib.Nifti1Image): Parcellation NIfTI image.
        labels (dict): Dictionary of label names to indices.
        selected_labels (list | None): Optional list of specific labels to include in context mask.
    Returns:
        tuple: Context mask and ventricle mask as NIfTI images.
    """
    if selected_labels is not None:
        ctx_mask = nimage.math_img(
            " + ".join(f"(x == {labels[k]})" for k in selected_labels if k in labels),
            x=parcellation_img,
        )
    else:
        ctx_mask = nimage.math_img(
            " + ".join(f"(x == {v})" for k, v in labels.items() if "ctx" in k),
            x=parcellation_img,
        )
    vent_mask_raw = nimage.math_img(
        " + ".join(f"(x == {v})" for k, v in labels.items() if "vent" in k),
        x=parcellation_img,
    )
    vent_mask = nimage.new_img_like(
        parcellation_img, ndimage.binary_erosion(nimage.get_data(vent_mask_raw))
    )
    return ctx_mask, vent_mask


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
        radial_order=6,
        laplacian_regularization=True,
        laplacian_weighting=0.2,
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

    dki_model = dki.DiffusionKurtosisModel(gtab)
    dki_fit = dki_model.fit(dwi_data.T)

    mk = dki_fit.mk

    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit(b0)

        dwi_ventricles = norm_masker.transform(dwi_nib)
        if delta_per_bvalue is not None:
            dwi_ventricles = dwi_ventricles[bvals_mask, :]

        mk_ventricles = dki_model.fit(dwi_ventricles.T).mk

        nmk = mk / mk_ventricles[~np.isnan(mk_ventricles)].mean()
        nmk = nmk.clip(0, np.percentile(nmk[~np.isnan(nmk)], 99))
        return masker.inverse_transform(nmk.T)

    logger.warning("Be careful, this is not normalized MK!")
    return masker.inverse_transform(mk.T)

from dipy.reconst.shm import CsaOdfModel
def compute_sh(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,  # unused but kept for API consistency
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
    sh_order: int = 6,
    sh_stat: str = "power",  # "power" | "l0" | "l2"
):
    """
    Compute a scalar SH-derived metric from DWI data.
    Returns a NIfTI image compatible with the rest of the pipeline.
    """
    breakpoint()
    b0 = nimage.index_img(dwi_nib, 0)
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)

    # --- b-value selection (same logic as MD / MK)
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

    # --- SH model
    sh_model = CsaOdfModel(gtab, sh_order=sh_order)
    sh_fit = sh_model.fit(dwi_data.T)

    sh_coeffs = sh_fit.shm_coeff  # (n_voxels, n_coeffs)

    # --- Reduce SH → scalar
    if sh_stat == "power":
        scalar = np.linalg.norm(sh_coeffs, axis=1)
    elif sh_stat == "l0":
        scalar = sh_coeffs[:, 0]
    elif sh_stat == "l2":
        scalar = np.linalg.norm(sh_coeffs[:, 1:6], axis=1)
    else:
        raise ValueError(f"Unknown SH stat: {sh_stat}")

    scalar = np.nan_to_num(scalar)
    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit(b0)
        dwi_ventricles = norm_masker.transform(dwi_nib)
        if delta_per_bvalue is not None:
            dwi_ventricles = dwi_ventricles[bvals_mask, :]
        sh_ventricles = sh_model.fit(dwi_ventricles.T)
        sh_vent_coeffs = sh_ventricles.shm_coeff
        if sh_stat == "power":
            scalar_vent = np.linalg.norm(sh_vent_coeffs, axis=1)
        elif sh_stat == "l0":
            scalar_vent = sh_vent_coeffs[:, 0]
        elif sh_stat == "l2":
            scalar_vent = np.linalg.norm(sh_vent_coeffs[:, 1:6], axis=1)
        else:
            raise ValueError(f"Unknown SH stat: {sh_stat}")
        nsh = scalar / scalar_vent.mean()
        scalar = nsh
        return masker.inverse_transform(scalar)
            
    scalar = scalar.clip(0, np.percentile(scalar, 99))

    return masker.inverse_transform(scalar)


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
    Returns:
        None
    """
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
            output_dir / f"sub-{subject_id}_hemi-{h}_param-{micr_metric}.scalar.gii",
        )


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


def load_rtop_data(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load RTOP scalar data from left and right .scalar.gii files.
    Assumes filenames follow format *_rtop_cortex.L/R*.scalar.gii
    Args:
        config (dict): Configuration dictionary with data paths and metric to compute.
    Returns:
        tuple[np.ndarray, np.ndarray]: RTOP data for left and right hemispheres.
    """
    subject_id = "100206"  # Test subject
    subject_dir = Path(config["data_paths"]["hcp_results"]) / subject_id / "processed"
    rtop_left = (
        nib.load(subject_dir / f"{config['metric_to_compute']}.L.scalar.gii")
        .darrays[0]
        .data
    )
    rtop_right = (
        nib.load(subject_dir / f"{config['metric_to_compute']}.R.scalar.gii")
        .darrays[0]
        .data
    )
    # Clean up (clip and replace NaNs)
    rtop_left = np.nan_to_num(rtop_left).clip(0, 7)
    rtop_right = np.nan_to_num(rtop_right).clip(0, 7)
    # breakpoint()
    return rtop_left, rtop_right


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


METRIC_COMPUTERS = {
    "rtop": compute_rtop,
    "md": compute_md,
    "mk": compute_mk,
    "sh": compute_sh,
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

    out_file = derivatives_dir / f"sub-{subject_id}_param-{metric}_dwimap.nii.gz"
    nib.save(metric_img, out_file)

    project_to_surface(
        metric_img,
        ctx_mask,
        surfaces,
        derivatives_dir,
        subject_id,
        metric,
        layouts=layouts,
        target_space=target_space,
        data_reading=data_reading,
    )

    return metric_img

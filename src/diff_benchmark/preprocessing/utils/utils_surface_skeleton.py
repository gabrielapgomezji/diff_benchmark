"""Surface and skeleton projection utilities for diffusion MRI.

This module handles:
- Projecting volumetric metrics onto cortical surfaces (gray matter).
- Projecting volumetric metrics onto the TBSS white matter skeleton.
- Resampling surface data between template spaces (fsLR ↔ fsaverage).
- Schaefer parcellation resampling and region-level feature extraction.
- JHU white matter atlas utilities.

All public function signatures are unchanged from the original
``utils_brain_feature_extraction`` module.
"""
import shutil
from pathlib import Path

import nibabel as nib
import nilearn as ni
import numpy as np
import pandas as pd
from nilearn import image as nimage
from scipy.spatial import cKDTree
from templateflow import api as tflow

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# FSL / JHU skeleton helpers
# ---------------------------------------------------------------------------


def download_fsl_skeleton(output_dir: Path = None) -> Path:
    """Download FSL's FMRIB58 FA skeleton if not available locally.

    Args:
        output_dir: Where to save the skeleton. Defaults to ``aux_materials/``.

    Returns:
        Path to the downloaded skeleton file.

    Raises:
        FileNotFoundError: When automatic download fails and manual steps are
            required.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "aux_materials"

    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_file = output_dir / "FMRIB58_FA-skeleton_1mm.nii.gz"

    if skeleton_file.exists():
        logger.info(f"Skeleton already exists at {skeleton_file}")
        return skeleton_file

    try:
        logger.info("Downloading white matter skeleton from TemplateFlow...")
        skeleton_path = tflow.get(
            "MNI152NLin2009cAsym",
            resolution=1,
            desc="brain",
            suffix="probseg",
            extension="nii.gz",
        )
        shutil.copy(skeleton_path, skeleton_file)
        logger.info(f"Downloaded skeleton from TemplateFlow to {skeleton_file}")
        return skeleton_file

    except Exception as e:
        logger.warning(f"TemplateFlow download failed: {e}")

    raise FileNotFoundError(
        f"\n{'='*80}\n"
        f"MANUAL DOWNLOAD REQUIRED\n"
        f"{'='*80}\n\n"
        f"Could not automatically download FSL skeleton.\n"
        f"Please download manually using ONE of these methods:\n\n"
        f"Method 3: Install TemplateFlow\n"
        f"  pip install templateflow\n"
        f"  python -c \"from templateflow import api as tflow; "
        f"tflow.get('MNI152NLin2009cAsym', resolution=1)\"\n\n"
        f"Then place the file at: {skeleton_file}\n"
        f"{'='*80}\n"
    )


def get_jhu_tract_names() -> list[str]:
    """Return the standard JHU ICBM-DTI-81 white matter tract names.

    These correspond to label IDs 1–48 in ``JHU-ICBM-labels-1mm.nii.gz``.

    Source: FSL JHU atlas documentation
    https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/blob/FinalFive/JHU-labels.xml

    Returns:
        48 tract names in order (label 1–48).
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
    template: str = "fmrib",  # noqa: ARG001 – kept for API parity; only "fmrib" is implemented
) -> tuple:
    """Load standard TBSS skeleton and JHU tract atlas.

    Args:
        skeleton_path: Path to an existing skeleton file. If ``None``, the
            function looks for it in ``aux_materials/`` and downloads it if
            absent.
        template: Which skeleton template to use (currently only ``"fmrib"``
            is implemented; kept for API compatibility).

    Returns:
        ``(skeleton_mask, tract_labels_skeleton_img, tract_names)`` where:
        - *skeleton_mask* – ``nib.Nifti1Image`` of the FA skeleton.
        - *tract_labels_skeleton_img* – JHU tract labels resampled and masked
          to skeleton voxels.
        - *tract_names* – list of 48 JHU tract name strings.
    """
    aux_dir = Path(__file__).parent.parent.parent.parent / "aux_materials"

    # Step 1: Load or locate/download skeleton
    if skeleton_path is None:
        skeleton_path = aux_dir / "FMRIB58_FA-skeleton_1mm.nii.gz"
        if not skeleton_path.exists():
            logger.warning("FSL skeleton not found locally, downloading...")
            skeleton_path = download_fsl_skeleton()

    skeleton_mask = nib.load(skeleton_path)
    logger.info(f"Loaded TBSS skeleton from {skeleton_path}")

    # Load JHU ICBM-DTI-81 atlas (48 tracts)
    jhu_labels_file = aux_dir / "JHU-ICBM-labels-1mm.nii.gz"
    tract_labels_img = nib.load(jhu_labels_file)
    tract_names = get_jhu_tract_names()
    logger.info(f"Loaded JHU ICBM-DTI-81 atlas with {len(tract_names)} tracts")

    # Resample atlas to skeleton space
    tract_labels_resampled = nimage.resample_to_img(
        tract_labels_img, skeleton_mask, interpolation="nearest"
    )

    # Restrict tract labels to skeleton voxels (FA > 0.2 threshold)
    skeleton_data = skeleton_mask.get_fdata() > 0.2
    tract_labels_on_skeleton = tract_labels_resampled.get_fdata() * skeleton_data
    tract_labels_skeleton_img = nimage.new_img_like(
        skeleton_mask, tract_labels_on_skeleton
    )

    return skeleton_mask, tract_labels_skeleton_img, tract_names


def classify_tract_hemisphere(tract_name: str) -> str:
    """Classify a JHU tract as left (``'L'``), right (``'R'``), or midline (``'M'``).

    Args:
        tract_name: Tract name from the JHU atlas (e.g.,
            ``"Anterior thalamic radiation L"``).

    Returns:
        ``'L'``, ``'R'``, or ``'M'``.
    """
    tract_lower = tract_name.lower()

    if tract_lower.endswith(" l") or "left" in tract_lower:
        return "L"
    if tract_lower.endswith(" r") or "right" in tract_lower:
        return "R"

    midline_keywords = [
        "corpus callosum",
        "genu of corpus callosum",
        "body of corpus callosum",
        "splenium of corpus callosum",
        "fornix",
        "middle cerebellar peduncle",
    ]
    if any(keyword in tract_lower for keyword in midline_keywords):
        return "M"

    logger.warning(
        f"Could not classify hemisphere for tract: {tract_name}, treating as midline"
    )
    return "M"


# ---------------------------------------------------------------------------
# White matter skeleton projection
# ---------------------------------------------------------------------------


def _save_hemisphere_gifti(
    scalars: np.ndarray,
    hemi: str,
    subject_id: str,
    metric_name: str,
    output_dir: Path,
) -> None:
    """Save a 1-D array of tract scalars as a ``.scalar.gii`` GIFTI file.

    Args:
        scalars: Tract scalar values (float32).
        hemi: Hemisphere label (``"L"``, ``"R"``, or ``"M"``).
        subject_id: Subject identifier.
        metric_name: Metric name (e.g., ``"rtop"``, ``"md"``).
        output_dir: Directory where the file is saved.
    """
    gii = nib.gifti.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.gifti.GiftiDataArray(scalars, intent="NIFTI_INTENT_DIMLESS")
    )
    out_file = (
        output_dir
        / f"sub-{subject_id}_hemi-{hemi}_param-{metric_name}_tissue-white.scalar.gii"
    )
    nib.save(gii, out_file)
    logger.info(
        f"[{subject_id}] Saved {len(scalars)} {hemi} WM tracts to {out_file}"
    )


def project_to_skeleton(
    metric_img: nib.nifti1.Nifti1Image,
    output_dir: Path,
    subject_id: str,
    metric_name: str,
) -> None:
    """Project a volumetric metric onto the TBSS skeleton and save GIFTI files.

    Resamples the TBSS skeleton into the subject's native metric space, extracts
    per-tract mean values, splits them by hemisphere (L/R/M), and saves
    hemisphere-split ``.scalar.gii`` files that mirror the gray matter format.

    Args:
        metric_img: Subject's metric image (RTOP, MD, etc.) in any space.
        output_dir: Where to save outputs.
        subject_id: Subject ID (used for filenames).
        metric_name: Metric name (used for filenames).

    Returns:
        ``None`` (results are written to disk).
    """
    logger.info(f"[{subject_id}] Projecting {metric_name} onto TBSS skeleton")
    skeleton_mask, tract_labels_skeleton_img, tract_names = load_tbss_skeleton()

    # Resample skeleton and tract atlas to subject metric space
    skeleton_mask_subject = nimage.resample_to_img(
        skeleton_mask, metric_img, interpolation="linear"
    )
    tract_labels_subject = nimage.resample_to_img(
        tract_labels_skeleton_img, metric_img, interpolation="nearest"
    )

    metric_data = metric_img.get_fdata()
    skeleton_data = skeleton_mask_subject.get_fdata() > 0.2
    tract_data = tract_labels_subject.get_fdata()
    metric_on_skeleton = metric_data * skeleton_data

    left_scalars, right_scalars, midline_scalars = [], [], []

    for tract_id, tract_name in enumerate(tract_names, start=1):
        tract_mask = (tract_data == tract_id) & skeleton_data
        value = np.nanmean(metric_on_skeleton[tract_mask]) if tract_mask.sum() > 0 else np.nan

        hemi = classify_tract_hemisphere(tract_name)
        if hemi == "L":
            left_scalars.append(value)
        elif hemi == "R":
            right_scalars.append(value)
        else:
            midline_scalars.append(value)

    left_arr = np.array(left_scalars, dtype=np.float32)
    right_arr = np.array(right_scalars, dtype=np.float32)
    midline_arr = np.array(midline_scalars, dtype=np.float32)

    for arr, hemi in [(left_arr, "L"), (right_arr, "R"), (midline_arr, "M")]:
        if len(arr) > 0:
            _save_hemisphere_gifti(arr, hemi, subject_id, metric_name, output_dir)

    return None


def extract_wm_tract_subset(
    left_tracts: np.ndarray,
    right_tracts: np.ndarray,
    midline_tracts: np.ndarray,
    tract_names: list[str] | None = None,
    target_tracts: list[str] | None = None,
) -> np.ndarray:
    """Extract a subset of white matter tracts by name pattern.

    Allows regional analysis by selecting specific tract groups (e.g., all
    "corona radiata", all "internal capsule", etc.).

    Args:
        left_tracts: Values for left hemisphere tracts.
        right_tracts: Values for right hemisphere tracts.
        midline_tracts: Values for midline tracts.
        tract_names: Full JHU tract names. If ``None``, loads the default list.
        target_tracts: List of tract name substrings to include, e.g.
            ``["corona radiata", "internal capsule"]``. If ``None``, returns
            all tracts.

    Returns:
        Concatenated values for the selected tracts only.
    """
    if tract_names is None:
        tract_names = get_jhu_tract_names()

    # If no target specified, return all
    if target_tracts is None:
        return np.concatenate([left_tracts, right_tracts, midline_tracts])

    selected_left: list = []
    selected_right: list = []
    selected_midline: list = []

    left_counter = right_counter = midline_counter = 0

    for name in tract_names:
        matches = any(t.lower() in name.lower() for t in target_tracts)
        hemi = classify_tract_hemisphere(name)

        if matches:
            if hemi == "L":
                selected_left.append(left_tracts[left_counter])
            elif hemi == "R":
                selected_right.append(right_tracts[right_counter])
            else:
                selected_midline.append(midline_tracts[midline_counter])

        # Always increment the relevant counter
        if hemi == "L":
            left_counter += 1
        elif hemi == "R":
            right_counter += 1
        else:
            midline_counter += 1

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


# ---------------------------------------------------------------------------
# Gray matter / cortical surface projection
# ---------------------------------------------------------------------------


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
) -> None:
    """Project a metric image onto cortical surfaces and save GIFTI files.

    For white matter, delegates to :func:`project_to_skeleton`.
    For BIDS datasets, automatically resamples from native space to template
    space using sphere-based nearest-neighbour mapping.

    Args:
        micr_img: NIfTI image with microstructure values.
        ctx_mask: Context (tissue) mask NIfTI image.
        surfaces: Dict with keys ``"L.pial"``, ``"L.white"``, ``"R.pial"``,
            ``"R.white"`` pointing to surface file paths.
        output_dir: Directory to save output GIFTI files.
        subject_id: Subject identifier for naming output files.
        micr_metric: Metric name for naming output files.
        layouts: List of BIDS layouts (needed for BIDS datasets to find sphere
            files).
        target_space: Target surface space for resampling (default
            ``"fslr_32k"``).
        data_reading: Dataset format (``"hcp"``, ``"bids"``,
            ``"multicenter-bids"``).
        tissue_type: Tissue type (``"gray"`` or ``"white"``).

    Returns:
        ``None`` (results are written to disk).
    """
    if tissue_type == "white":
        logger.info(f"[{subject_id}] Projecting white matter to skeleton")
        project_to_skeleton(micr_img, output_dir, subject_id, micr_metric)
        return None

    # Project to surface (native space for BIDS, template space for HCP)
    left_data = right_data = None
    for h in ("L", "R"):
        surf_data = ni.surface.vol_to_surf(
            micr_img,
            surfaces[f"{h}.pial"],
            mask_img=ctx_mask,
            inner_mesh=surfaces[f"{h}.white"],
            depth=[0.1, 0.5, 0.9],
        )
        if h == "L":
            left_data = surf_data
        else:
            right_data = surf_data

    # For BIDS datasets, resample native → template space
    if "bids" in data_reading and layouts is not None:
        try:
            logger.info(
                f"[{subject_id}] Resampling surface data from native to "
                f"{target_space} space"
            )
            left_data, right_data = resample_subject_to_template(
                subject_id=subject_id,
                left_data=left_data,
                right_data=right_data,
                layouts=layouts,
                target_space=target_space,
            )
        except Exception as e:
            logger.warning(
                f"[{subject_id}] Resampling failed, saving native space data: {e}"
            )

    # Save hemisphere GIFTI files
    for h, data in [("L", left_data), ("R", right_data)]:
        img = nib.gifti.gifti.GiftiImage()
        img.add_gifti_data_array(
            nib.gifti.gifti.GiftiDataArray(
                data.astype(np.float32), intent="NIFTI_INTENT_DIMLESS"
            )
        )
        nib.save(
            img,
            output_dir
            / f"sub-{subject_id}_hemi-{h}_param-{micr_metric}_tissue-{tissue_type}.scalar.gii",
        )

    return None


# ---------------------------------------------------------------------------
# Template-space resampling
# ---------------------------------------------------------------------------


def resample_subject_to_template(
    subject_id: str,
    left_data: np.ndarray,
    right_data: np.ndarray,
    layouts: list,
    target_space: str = "fslr_32k",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample subject-native surface data to a template space.

    Uses sphere-based nearest-neighbour mapping via
    :class:`scipy.spatial.cKDTree`.

    Args:
        subject_id: Subject identifier.
        left_data: Data on the left hemisphere in subject native space.
        right_data: Data on the right hemisphere in subject native space.
        layouts: List of BIDS layouts to search for subject data.
        target_space: Target template space (``"fslr_32k"`` or
            ``"fsaverage"``).

    Returns:
        ``(left_resampled, right_resampled)`` — data mapped to the template
        space.

    Raises:
        ValueError: If the subject is not found in any layout, or if
            ``target_space`` is unknown.
        FileNotFoundError: If sphere surface files are missing for the subject.
    """
    # Find the layout containing this subject
    layout = None
    for lay in layouts:
        if subject_id in lay.get_subjects():
            layout = lay
            break

    if layout is None:
        raise ValueError(f"Subject {subject_id} not found in any layout")

    resampled_data: dict[str, np.ndarray] = {}

    for hemi, data in [("L", left_data), ("R", right_data)]:
        subject_sphere_files = layout.get(
            subject=subject_id,
            suffix="sphere",
            hemi=hemi,
            extension=".surf.gii",
            space="fsLR",
            return_type="files",
        )

        if not subject_sphere_files:
            raise FileNotFoundError(
                f"No sphere surface found for subject {subject_id}, "
                f"hemisphere {hemi}"
            )

        subject_sphere = nib.load(subject_sphere_files[0])

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

        # Nearest-neighbour mapping: for each template vertex → nearest subject vertex
        kdtree_subject = cKDTree(subject_sphere.darrays[0].data)
        subject_to_template = kdtree_subject.query(
            template_sphere.darrays[0].data, k=1
        )
        resampled_data[hemi] = data[subject_to_template[1]]

    return resampled_data["L"], resampled_data["R"]


# ---------------------------------------------------------------------------
# Schaefer parcellation resampling
# ---------------------------------------------------------------------------


def resample_schaefer_onto_fs_lr(
    scale: int = 1000, target_space: str = "fslr_32k"
) -> dict:
    """Resample the Schaefer 2018 parcellation onto fsLR or fsaverage space.

    Args:
        scale: Schaefer parcellation scale (e.g., ``1000`` for 1000 parcels).
        target_space: Target surface space.

            - ``"fslr_32k"``: fsLR 32k space (HCP default).
            - ``"fsaverage"``: Native fsaverage space (FreeSurfer/CamCAN).

    Returns:
        Dictionary with keys ``"left.data"``, ``"left.labels"``,
        ``"left.sulc"``, ``"right.data"``, ``"right.labels"``,
        ``"right.sulc"``.
    """
    fsaverage_left_schaefer = nib.load(
        tflow.get(
            "fsaverage",
            hemi="L",
            density="164k",
            atlas="Schaefer2018",
            segmentation="17n",
            scale=str(scale),
            extension="label.gii",
        )
    )
    fsaverage_right_schaefer = nib.load(
        tflow.get(
            "fsaverage",
            hemi="R",
            density="164k",
            atlas="Schaefer2018",
            segmentation="17n",
            scale=str(scale),
            extension="label.gii",
        )
    )

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

    if target_space == "fsaverage":
        fslr_left_sulc = (
            nib.load(
                tflow.get(
                    "fsaverage", hemi="L", density="164k", suffix="sulc", desc=None
                )
            )
            .darrays[0]
            .data
        )
        fslr_right_sulc = (
            nib.load(
                tflow.get(
                    "fsaverage", hemi="R", density="164k", suffix="sulc", desc=None
                )
            )
            .darrays[0]
            .data
        )
        return {
            "left.data": fsaverage_left_schaefer.darrays[0].data,
            "left.labels": labels_left,
            "left.sulc": fslr_left_sulc,
            "right.data": fsaverage_right_schaefer.darrays[0].data,
            "right.labels": labels_right,
            "right.sulc": fslr_right_sulc,
        }

    # Resample to fsLR 32k via KD-tree nearest-neighbour mapping
    def _load_sphere(template: str, hemi: str, density: str, **kwargs):
        return nib.load(tflow.get(template, hemi=hemi, density=density, **kwargs))

    def _build_mapping(src_sphere, tgt_sphere):
        """Return (src→tgt, tgt→src) nearest-neighbour index arrays."""
        kdtree_src = cKDTree(src_sphere.darrays[0].data)
        kdtree_tgt = cKDTree(tgt_sphere.darrays[0].data)
        src_to_tgt = kdtree_src.query(tgt_sphere.darrays[0].data, k=1)
        tgt_to_src = kdtree_tgt.query(src_sphere.darrays[0].data, k=1)
        return src_to_tgt, tgt_to_src

    results = {}
    for hemi, fsaverage_schaefer in [
        ("L", fsaverage_left_schaefer),
        ("R", fsaverage_right_schaefer),
    ]:
        fslr_sphere = _load_sphere(
            "fsLR", hemi=hemi, density="32k", space="fsaverage"
        )
        fsaverage_sphere = _load_sphere(
            "fsaverage", hemi=hemi, density="164k", suffix="sphere", desc=None
        )
        fslr_to_fsaverage, fsaverage_to_fslr = _build_mapping(
            fslr_sphere, fsaverage_sphere
        )

        fslr_schaefer = np.zeros(len(fsaverage_to_fslr[1]))
        fslr_schaefer[fslr_to_fsaverage[1]] = fsaverage_schaefer.darrays[0].data

        sulc_data = (
            nib.load(
                tflow.get(
                    "fsaverage",
                    hemi=hemi,
                    density="164k",
                    suffix="sulc",
                    desc=None,
                )
            )
            .darrays[0]
            .data
        )
        fslr_sulc = np.zeros(len(fsaverage_to_fslr[1]))
        fslr_sulc[fslr_to_fsaverage[1]] = sulc_data

        h_key = "left" if hemi == "L" else "right"
        lbl = labels_left if hemi == "L" else labels_right
        results[f"{h_key}.data"] = fslr_schaefer
        results[f"{h_key}.labels"] = lbl
        results[f"{h_key}.sulc"] = fslr_sulc

    return results


# ---------------------------------------------------------------------------
# Parcel-level feature extraction
# ---------------------------------------------------------------------------


def average_per_parcel(
    hem_left: np.ndarray, hem_right: np.ndarray, schaefer_resampled: dict
) -> np.ndarray:
    """Average microstructure values across Schaefer parcels in both hemispheres.

    Args:
        hem_left: Microstructure values for the left hemisphere.
        hem_right: Microstructure values for the right hemisphere.
        schaefer_resampled: Dict from :func:`resample_schaefer_onto_fs_lr`
            containing ``"left.data"`` and ``"right.data"`` arrays.

    Returns:
        Mean values per parcel, concatenated left then right.
    """
    parcellation_left = schaefer_resampled["left.data"]
    parcellation_right = schaefer_resampled["right.data"]
    parcels_left = np.unique(parcellation_left)
    parcels_right = np.unique(parcellation_right)

    rtop_avg = np.zeros(len(parcels_left) + len(parcels_right))
    for i, parcel in enumerate(sorted(parcels_left)):
        rtop_avg[i] = hem_left[parcellation_left == parcel].mean()
    for i, parcel in enumerate(sorted(parcels_right), start=len(parcels_left)):
        rtop_avg[i] = hem_right[parcellation_right == parcel].mean()
    return rtop_avg


# ---------------------------------------------------------------------------
# Surface-mesh helpers (used by MeshPipeline)
# ---------------------------------------------------------------------------


def load_template_surface(
    hemi: str,
    space: str = "fslr_32k",
    surf_type: str = "midthickness",
) -> tuple[np.ndarray, np.ndarray]:
    """Load template-space surface vertices and faces from TemplateFlow.

    Downloads the surface file if not already cached locally.

    Args:
        hemi: Hemisphere identifier — ``"L"`` or ``"R"``.
        space: Template space.  Currently only ``"fslr_32k"`` is supported
            (uses the HCP fsLR 32k surface from TemplateFlow).
        surf_type: Surface type (``"midthickness"``, ``"inflated"``,
            ``"pial"``, ``"white"``).

    Returns:
        ``(vertices, faces)`` as ``(N, 3)`` float32 and ``(M, 3)`` int32 arrays.

    Raises:
        ValueError: If *space* is not supported.
        FileNotFoundError: If TemplateFlow cannot fetch the file.
    """
    if space != "fslr_32k":
        raise ValueError(
            f"load_template_surface: only 'fslr_32k' is currently supported, "
            f"got '{space}'"
        )

    surf_path = tflow.get(
        "fsLR",
        hemi=hemi,
        density="32k",
        suffix=surf_type,
        extension=".surf.gii",
    )
    if not surf_path:
        raise FileNotFoundError(
            f"TemplateFlow could not find {surf_type} surface for fsLR 32k "
            f"hemisphere {hemi}"
        )

    img = nib.load(str(surf_path))
    vertices = img.darrays[0].data.astype(np.float32)  # (N, 3)
    faces = img.darrays[1].data.astype(np.int32)       # (M, 3)
    logger.debug(
        "Loaded template surface %s %s %s: %d vertices, %d faces",
        space, hemi, surf_type, vertices.shape[0], faces.shape[0],
    )
    return vertices, faces


def build_parcel_label_vector(
    schaefer_resampled: dict,
    n_left: int | None = None,
    n_right: int | None = None,
) -> np.ndarray:
    """Build a combined vertex-wise parcel label vector for L+R hemispheres.

    Concatenates the left and right parcel ID arrays from *schaefer_resampled*
    into a single ``(N_L + N_R,)`` int32 vector.  Vertices on the medial wall
    (parcel ID == 0) are left as 0.

    Args:
        schaefer_resampled: Dict returned by :func:`resample_schaefer_onto_fs_lr`.
        n_left: Expected number of left-hemisphere vertices.  If given and the
            label array length differs, a warning is emitted.
        n_right: Expected number of right-hemisphere vertices.

    Returns:
        ``(N_L + N_R,)`` int32 array of parcel IDs.
    """
    left_labels = schaefer_resampled["left.data"].astype(np.int32)
    right_labels = schaefer_resampled["right.data"].astype(np.int32)

    if n_left is not None and len(left_labels) != n_left:
        logger.warning(
            "build_parcel_label_vector: left label count %d != expected %d",
            len(left_labels), n_left,
        )
    if n_right is not None and len(right_labels) != n_right:
        logger.warning(
            "build_parcel_label_vector: right label count %d != expected %d",
            len(right_labels), n_right,
        )

    return np.concatenate([left_labels, right_labels])


def extract_region_data(
    hem_left: np.ndarray,
    hem_right: np.ndarray,
    schaefer_resampled: dict,
    target_substring: str | None = None,
    average: bool = False,
) -> np.ndarray:
    """Extract microstructure values for selected Schaefer parcels.

    Args:
        hem_left: Microstructure values for the left hemisphere.
        hem_right: Microstructure values for the right hemisphere.
        schaefer_resampled: Dict from :func:`resample_schaefer_onto_fs_lr`.
        target_substring: Optional string to filter parcel names (case-
            insensitive). If ``None``, all parcels are returned.
        average: If ``True``, return the per-region mean instead of all
            vertex values.

    Returns:
        Concatenated vertex (or mean) values for the selected regions.
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

    region_values = []
    for _, row in matched_labels.iterrows():
        hemi = row["hemi"]
        region_id = row["array_index"]
        if hemi == "L":
            vals = hem_left[parc_left == region_id]
        elif hemi == "R":
            vals = hem_right[parc_right == region_id]
        else:
            continue  # unexpected hemi label

        if vals.size == 0:
            continue

        region_values.append(np.nanmean(vals) if average else vals)

    return np.concatenate([np.atleast_1d(v) for v in region_values])

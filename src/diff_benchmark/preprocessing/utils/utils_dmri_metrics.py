"""DWI metric computation utilities.

This module contains functions for computing diffusion MRI scalar metrics
(RTOP, MD, MK, SH, b0) from raw DWI data, plus the orchestrating function
that saves and projects a metric through the pipeline.

All public function signatures are unchanged from the original
``utils_brain_feature_extraction`` module.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.reconst import dki, dti
from dipy.reconst.mapmri import MapmriModel
from dipy.reconst.shm import sf_to_sh
from nilearn import image as nimage
from nilearn import maskers

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _select_bvals_mask(
    bvals: np.ndarray,
    big_delta: float,
    delta_per_bvalue: dict | None,
) -> np.ndarray:
    """Return a boolean mask selecting the b-values relevant for a given Δ.

    If ``delta_per_bvalue`` is provided, only b0 volumes and shells whose
    stored Δ matches ``big_delta * 1000`` (ms → µs unit) are selected.
    Otherwise all volumes are selected (mask of all True).

    Args:
        bvals: 1-D array of b-values, one per DWI volume.
        big_delta: Big delta value (in seconds).
        delta_per_bvalue: Optional mapping ``{bvalue: delta_in_ms}``.

    Returns:
        Boolean array of shape ``(len(bvals),)``.
    """
    if delta_per_bvalue is not None:
        selected_bvals = [0] + [
            k for k, v in delta_per_bvalue.items() if v == big_delta * 1000
        ]
        return np.any([bvals == s for s in selected_bvals], axis=0)
    return np.ones_like(bvals, dtype=bool)


def _normalize_by_ventricles(
    metric: np.ndarray,
    model,
    dwi_nib: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,
    ref_b0: nib.nifti1.Nifti1Image,
    bvals_mask: np.ndarray,
    fit_method: str = "rtop",
    clip: bool = True,
) -> np.ndarray:
    """Normalise a 1-D metric array by the ventricular mean.

    Args:
        metric: Scalar values in the brain mask (1-D, shape ``(n_voxels,)``).
        model: Fitted dipy model supporting ``fit(data).{fit_method}()``.
        dwi_nib: Full DWI 4-D image (used to extract ventricular signal).
        normalization_mask_img: Ventricular mask.
        ref_b0: Reference b0 image (used to initialise the masker).
        bvals_mask: Boolean mask selecting the relevant volumes.
        fit_method: Which attribute of the fit result to call (``"rtop"``,
            ``"md"``, ``"mk"``).
        clip: Whether to clip the result to the 99th percentile.

    Returns:
        Normalised metric array (same shape as ``metric``).
    """
    norm_masker = maskers.NiftiMasker(normalization_mask_img)
    norm_masker.fit(ref_b0)
    dwi_ventricles = norm_masker.transform(dwi_nib)[bvals_mask, :]

    fit = model.fit(dwi_ventricles.T)
    metric_ventricles = getattr(fit, fit_method)()

    valid_vent = ~np.isnan(metric_ventricles)
    vent_mean = metric_ventricles[valid_vent].mean()

    normalised = metric / vent_mean
    if clip:
        valid = ~np.isnan(normalised)
        normalised = normalised.clip(0, np.percentile(normalised[valid], 99))
    return normalised


# ---------------------------------------------------------------------------
# Public metric computers
# ---------------------------------------------------------------------------


def compute_rtop(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
) -> nib.nifti1.Nifti1Image:
    """Compute RTOP (Return-to-Origin Probability) from DWI data via MAP-MRI.

    Args:
        dwi_nib: DWI NIfTI image.
        mask_img: Brain mask NIfTI image.
        normalization_mask_img: Ventricular mask for intensity normalisation.
        bvals: Array of b-values.
        bvecs: Array of b-vectors.
        big_delta: Big delta value (seconds).
        small_delta: Small delta value (seconds).
        delta_per_bvalue: Optional mapping ``{bvalue: delta_in_ms}``.

    Returns:
        RTOP NIfTI image (ventricular-normalised when mask is provided).
    """
    b0 = nimage.index_img(dwi_nib, 0)
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)

    bvals_mask = _select_bvals_mask(bvals, big_delta, delta_per_bvalue)
    dwi_data = dwi_data[bvals_mask, :]

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
        # Percentile clipping intentionally omitted to preserve raw normalised values.
        return masker.inverse_transform(nrtop.T)

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
) -> nib.nifti1.Nifti1Image:
    """Compute Mean Diffusivity (MD) from DWI data via DTI.

    Args:
        dwi_nib: DWI NIfTI image.
        mask_img: Brain mask NIfTI image.
        normalization_mask_img: Ventricular mask for intensity normalisation.
        bvals: Array of b-values.
        bvecs: Array of b-vectors.
        big_delta: Big delta value (seconds).
        small_delta: Small delta value (seconds).
        delta_per_bvalue: Optional mapping ``{bvalue: delta_in_ms}``.

    Returns:
        MD NIfTI image (ventricular-normalised when mask is provided).
    """
    b0 = nimage.index_img(dwi_nib, 0)
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)

    bvals_mask = _select_bvals_mask(bvals, big_delta, delta_per_bvalue)
    dwi_data = dwi_data[bvals_mask, :]

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
        return masker.inverse_transform(nmd.T)

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
) -> nib.nifti1.Nifti1Image:
    """Compute Mean Kurtosis (MK) from DWI data via DKI.

    Args:
        dwi_nib: DWI NIfTI image.
        mask_img: Brain mask NIfTI image.
        normalization_mask_img: Ventricular mask for intensity normalisation.
        bvals: Array of b-values.
        bvecs: Array of b-vectors.
        big_delta: Big delta value (seconds).
        small_delta: Small delta value (seconds).
        delta_per_bvalue: Optional mapping ``{bvalue: delta_in_ms}``.

    Returns:
        MK NIfTI image (ventricular-normalised when mask is provided).
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
            # Do not filter — DKI requires multi-shell data
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


def compute_sh(
    dwi_nib: nib.nifti1.Nifti1Image,
    mask_img: nib.nifti1.Nifti1Image,
    normalization_mask_img: nib.nifti1.Nifti1Image,  # noqa: ARG001 – kept for API parity
    bvals: np.ndarray,
    bvecs: np.ndarray,
    big_delta: float,
    small_delta: float,
    delta_per_bvalue: dict | None = None,
    sh_order: int = 6,
) -> nib.nifti1.Nifti1Image:
    """Compute the L2-norm of spherical harmonic coefficients fitted to the DWI signal.

    Ventricular normalisation is intentionally skipped: the SH-power scale
    differs from RTOP/MD, making normalisation by ventricular mean unreliable.

    Args:
        dwi_nib: DWI NIfTI image.
        mask_img: Brain mask NIfTI image.
        normalization_mask_img: Ventricular mask (unused; kept for API parity).
        bvals: Array of b-values.
        bvecs: Array of b-vectors.
        big_delta: Big delta value (seconds).
        small_delta: Small delta value (seconds).
        delta_per_bvalue: Optional mapping ``{bvalue: delta_in_ms}``.
        sh_order: Spherical harmonic order (default 6).

    Returns:
        SH-power NIfTI image (un-normalised).
    """
    b0 = nimage.index_img(dwi_nib, 0)
    masker = maskers.NiftiMasker(mask_img)
    masker.fit(b0)
    dwi_data = masker.transform(dwi_nib)

    bvals_mask = _select_bvals_mask(bvals, big_delta, delta_per_bvalue)
    dwi_data = dwi_data[bvals_mask, :]

    gtab = gradient_table(
        bvals=bvals[bvals_mask],
        bvecs=bvecs[bvals_mask],
        small_delta=small_delta,
        big_delta=big_delta,
    )

    # Non-b0 gradient directions and signal
    dwi_dirs = gtab.bvecs[gtab.bvals > 0]
    dwi_signal = dwi_data[gtab.bvals > 0, :]
    b0_data = dwi_data[0, :]

    dwi_normalized = dwi_signal / (b0_data + 1e-6)

    sphere = Sphere(xyz=dwi_dirs)
    sh_coeffs = sf_to_sh(
        dwi_normalized.T,  # (n_voxels, n_gradients)
        sphere,
        sh_order=sh_order,
        basis_type="descoteaux07",
    )

    # Power (L2 norm) over SH coefficients
    scalar = np.linalg.norm(sh_coeffs, axis=1)
    scalar = np.nan_to_num(scalar)

    scalar = scalar.clip(0, np.percentile(scalar[~np.isnan(scalar)], 99))
    return masker.inverse_transform(scalar)


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
    """Average all b0 volumes to produce a T2-weighted baseline image.

    Signature mirrors other ``compute_*`` functions for drop-in compatibility
    with :func:`compute_save_and_project_metric` and ``METRIC_COMPUTERS``.

    Args:
        dwi_nib: Full DWI 4-D NIfTI image.
        mask_img: Brain / tissue mask.
        normalization_mask_img: Ventricular mask for intensity normalisation.
            Pass ``None`` to skip normalisation.
        bvals: 1-D array of b-values.
        bvecs: Array of b-vectors (unused; kept for API consistency).
        big_delta: Unused; kept for API consistency.
        small_delta: Unused; kept for API consistency.
        delta_per_bvalue: Unused; kept for API consistency.
        b0_threshold: Volumes with ``bval <= b0_threshold`` are treated as b0
            (default 50 s/mm²).

    Returns:
        3-D NIfTI image of averaged (and optionally normalised) b0 signal.

    Raises:
        ValueError: If no b0 volumes are found within ``b0_threshold``.
    """
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

    ref_b0 = nimage.index_img(dwi_nib, int(b0_indices[0]))

    masker = maskers.NiftiMasker(mask_img)
    masker.fit(ref_b0)

    b0_imgs = [nimage.index_img(dwi_nib, int(i)) for i in b0_indices]
    b0_data = np.stack(
        [masker.transform(img).squeeze() for img in b0_imgs], axis=0
    )  # (n_b0, n_voxels)

    mean_b0 = np.mean(b0_data, axis=0)
    mean_b0 = np.nan_to_num(mean_b0, nan=0.0)

    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit(ref_b0)

        vent_b0_data = np.stack(
            [norm_masker.transform(img).squeeze() for img in b0_imgs], axis=0
        )
        vent_mean = np.nanmean(np.mean(vent_b0_data, axis=0))

        if vent_mean > 1e-6:
            mean_b0 = mean_b0 / vent_mean
            logger.info(
                f"compute_b0: ventricular normalisation applied "
                f"(vent_mean={vent_mean:.4f})"
            )
        else:
            logger.warning(
                "compute_b0: skipping ventricular normalisation (vent_mean ~ 0)"
            )
    else:
        logger.warning(
            "compute_b0: no ventricular mask provided, "
            "returning un-normalised b0 signal."
        )

    # Clip to 99th percentile to suppress outlier voxels
    valid = ~np.isnan(mean_b0)
    if valid.any():
        mean_b0 = mean_b0.clip(0, np.percentile(mean_b0[valid], 99))

    return masker.inverse_transform(mean_b0)


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRIC_COMPUTERS: dict = {
    "rtop": compute_rtop,
    "md": compute_md,
    "mk": compute_mk,
    "sh": compute_sh,
    "b0": compute_b0,
}


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


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
    """Compute a diffusion metric, save to disk, and project to surface/skeleton.

    For BIDS datasets, surface data is automatically resampled from native
    to template space during projection.

    Args:
        metric: Metric name; must be a key in ``METRIC_COMPUTERS``.
        dwi_nib: DWI 4-D NIfTI image.
        ctx_mask: Cortical/tissue mask NIfTI image.
        vent_mask: Ventricular mask NIfTI image.
        bvals: Array of b-values.
        bvecs: Array of b-vectors.
        big_delta: Big delta (seconds).
        small_delta: Small delta (seconds).
        big_delta_per_bvalue: Per-b-value big delta mapping.
        surfaces: Dict of cortical surface paths for projection.
        derivatives_dir: Output directory; file is saved as
            ``sub-{subject_id}_param-{metric}_tissue-{tissue_type}_dwimap.nii.gz``.
        subject_id: Subject identifier.
        layouts: BIDS layouts required for sphere-based resampling.
        target_space: Target surface space (default ``"fslr_32k"``).
        data_reading: Dataset format (``"hcp"``, ``"bids"``, ``"multicenter-bids"``).
        tissue_type: Tissue type embedded in output filenames.

    Returns:
        Computed metric as a NIfTI image.

    Raises:
        ValueError: If ``metric`` is not in ``METRIC_COMPUTERS``.
    """
    # Lazy import to avoid circular dependency (surface utils import from here)
    from diff_benchmark.preprocessing.utils.utils_surface_skeleton import project_to_surface

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

    out_file = (
        derivatives_dir
        / f"sub-{subject_id}_param-{metric}_tissue-{tissue_type}_dwimap.nii.gz"
    )
    nib.save(metric_img, out_file)
    logger.info(f"[{subject_id}] Saved raw {metric} image to {out_file}")

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

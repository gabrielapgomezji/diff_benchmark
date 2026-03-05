from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from diff_benchmark.data.surface_mesh import SurfaceMeshData
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.preparation_pipeline import (
    BrainDataPreparationPipeline,
)
from diff_benchmark.preprocessing.utils.utils_brain_feature_extraction import (
    build_parcel_label_vector,
    extract_region_data,
    load_template_surface,
)
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


class DefaultPipeline(BrainDataPreparationPipeline):
    """Surface-based pipeline that stores per-subject metrics as ``.scalar.gii`` files."""

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        self.results_root = Path(dataset_config.results_dir) / "default"

    def verify_subject_files(
        self, subject_id: str, metric: str, tissue_type: str
    ) -> bool:
        """Return True if both hemispheres' ``.scalar.gii`` files exist for the given subject.

        Args:
            subject_id: Subject identifier.
            metric: Microstructure metric (e.g. ``"rtop"``, ``"md"``).
            tissue_type: Tissue type (``"gray"`` or ``"white"``).

        Returns:
            True if both left and right scalar files are present.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        left_file = (
            derivatives_dir
            / f"sub-{subject_id}_hemi-L_param-{metric}_tissue-{tissue_type}.scalar.gii"
        )
        right_file = (
            derivatives_dir
            / f"sub-{subject_id}_hemi-R_param-{metric}_tissue-{tissue_type}.scalar.gii"
        )

        return left_file.exists() and right_file.exists()

    def verify_resampling(self, subject_id: str) -> bool:
        """Return True if scalar data is in fsLR 32k template space.

        For non-BIDS datasets, always returns True (data is already in template
        space). For BIDS datasets, checks that each hemisphere's vertex count
        matches the expected fsLR 32k size (32,492 vertices).

        Args:
            subject_id: Subject identifier.

        Returns:
            True if data is properly resampled (or resampling is not required).
        """
        # HCP data is already in template space, no resampling needed
        if "bids" not in self.data_reading:
            return True

        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        left_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-L_param-{self.metric}.scalar.gii"
        )
        right_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-R_param-{self.metric}.scalar.gii"
        )

        if not (left_file.exists() and right_file.exists()):
            return False

        try:
            EXPECTED_VERTICES = 32492  # fsLR 32k template vertex count

            left_data = nib.load(left_file).darrays[0].data
            right_data = nib.load(right_file).darrays[0].data

            left_vertices = left_data.shape[0]
            right_vertices = right_data.shape[0]

            # Check if data has the expected number of vertices for template space
            is_resampled = (
                left_vertices == EXPECTED_VERTICES
                and right_vertices == EXPECTED_VERTICES
            )

            if not is_resampled:
                logger.debug(
                    f"[{subject_id}] Data in native space: L={left_vertices}, R={right_vertices} vertices"
                )

            return is_resampled

        except Exception as e:
            logger.warning(f"[{subject_id}] Error checking resampling status: {e}")
            return False

    def resample_data(self, subject_id: str):
        """Resample existing ``.scalar.gii`` files from native to template space.

        Only applies to BIDS datasets. Intended for data preprocessed before
        automatic resampling was introduced.

        Args:
            subject_id: Subject identifier.
        """
        if "bids" not in self.data_reading:
            logger.info(f"[{subject_id}] Not a BIDS dataset, no resampling needed")
            return

        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        left_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-L_param-{self.metric}.scalar.gii"
        )
        right_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-R_param-{self.metric}.scalar.gii"
        )

        if not (left_file.exists() and right_file.exists()):
            logger.warning(
                f"[{subject_id}] Scalar files not found, skipping resampling"
            )
            return

        try:
            logger.info(f"[{subject_id}] Resampling existing data to template space")

            # Clip extreme outliers to ensure stable interpolation during resampling.
            left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(0, 7)
            right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(0, 7)

            # Resample to template space
            from diff_benchmark.preprocessing.utils.utils_brain_feature_extraction import (
                resample_subject_to_template,
            )

            left_resampled, right_resampled = resample_subject_to_template(
                subject_id=subject_id,
                left_data=left_data,
                right_data=right_data,
                layouts=self.layouts,
                target_space=self.surface_space,
            )

            # Overwrite native-space files with template-space data
            left_gii = nib.gifti.GiftiImage()
            left_gii.add_gifti_data_array(
                nib.gifti.GiftiDataArray(data=left_resampled.astype(np.float32))
            )
            nib.save(left_gii, left_file)

            right_gii = nib.gifti.GiftiImage()
            right_gii.add_gifti_data_array(
                nib.gifti.GiftiDataArray(data=right_resampled.astype(np.float32))
            )
            nib.save(right_gii, right_file)

            logger.info(f"[{subject_id}] Resampled data saved successfully")

        except Exception as e:
            logger.error(f"[{subject_id}] Error during resampling: {e}")

    def run_analysis(self):
        """Fill ``self.results`` with existing ``.scalar.gii`` derivative files.

        Loads left/right hemisphere scalar files, extracts parcel-level or
        tract-level features, and stores them keyed by subject ID.
        """
        tissue_type = self.dataset_config.tissue_type

        scalar_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}_tissue-{tissue_type}.scalar.gii"
            )
        )
        for left_file in tqdm(scalar_files, desc="Running analysis"):
            try:
                subject_id = left_file.stem.split("_")[0].replace("sub-", "")
                right_file = left_file.with_name(
                    left_file.name.replace("hemi-L", "hemi-R")
                )

                # Load data (already in template space if BIDS dataset was properly preprocessed)
                # Clip extreme outliers to ensure stable interpolation during resampling.
                left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
                    0, 7
                )
                right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
                    0, 7
                )
                target = self.dataset_config.region

                if tissue_type == "white":
                    # Load midline data for white matter
                    midline_file = left_file.with_name(
                        left_file.name.replace("hemi-L", "hemi-M")
                    )
                    if midline_file.exists():
                        midline_data = np.nan_to_num(
                            nib.load(midline_file).darrays[0].data
                        ).clip(0, 7)
                        # Concatenate L, R, M for white matter
                        if target is not None:
                            # Extract a subset of tracts matching the target pattern
                            from diff_benchmark.preprocessing.utils.utils_brain_feature_extraction import (
                                extract_wm_tract_subset,
                            )

                            tract_data = extract_wm_tract_subset(
                                left_data,
                                right_data,
                                midline_data,
                                target_tracts=(
                                    [target] if isinstance(target, str) else target
                                ),
                            )
                            logger.info(
                                f"[{subject_id}] Extracted {len(tract_data)} tracts matching '{target}'"
                            )
                        else:
                            tract_data = np.concatenate(
                                [left_data, right_data, midline_data]
                            )

                        self.results[subject_id] = tract_data
                    else:
                        tract_data = np.concatenate([left_data, right_data])
                        self.results[subject_id] = tract_data
                else:
                    avg_data = extract_region_data(
                        left_data,
                        right_data,
                        self.schaefer_resampled,
                        target_substring=target,
                        average=False,
                    )
                    self.results[subject_id] = avg_data
            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during analysis: {e}")
                logger.warning(f"[{subject_id}] Expected error during analysis: {e}")


class ImagePipeline(BrainDataPreparationPipeline):
    """Volumetric pipeline that stores per-subject metrics as ``.nii.gz`` files."""

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        self.results_root = Path(dataset_config.results_dir) / "default"

    def verify_subject_files(
        self, subject_id: str, metric: str, tissue_type: str
    ) -> bool:
        """Return True if the whole-brain ``.nii.gz`` metric file exists.

        Args:
            subject_id: Subject identifier.
            metric: Microstructure metric (e.g. ``"rtop"``, ``"md"``).
            tissue_type: Tissue type (``"gray"`` or ``"white"``).

        Returns:
            True if the expected NIfTI file is present.
        """
        tissue_type = self.dataset_config.tissue_type
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        file = (
            derivatives_dir
            / f"sub-{subject_id}_param-{metric}_tissue-{tissue_type}_dwimap.nii.gz"
        )

        return file.exists()

    def run_analysis(self):
        """Fill ``self.results`` with paths to existing ``.nii.gz`` derivative files."""
        tissue_type = self.dataset_config.tissue_type
        img_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_param-{self.metric}_tissue-{tissue_type}_dwimap.nii.gz"
            )
        )
        for file in tqdm(img_files, desc="Running analysis"):
            try:
                subject_id = file.stem.split("_")[0].replace("sub-", "")
                self.results[subject_id] = file
            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during analysis: {e}")

    def run_analysis_region(self):
        """Fill ``self.results`` with paths to existing ``.nii.gz`` derivative files.

        Identical to :meth:`run_analysis` but additionally logs warnings on failure.
        """
        tissue_type = self.dataset_config.tissue_type
        img_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_param-{self.metric}_tissue-{tissue_type}_dwimap.nii.gz"
            )
        )

        for file in tqdm(img_files, desc="Running analysis"):
            try:
                subject_id = file.stem.split("_")[0].replace("sub-", "")
                self.results[subject_id] = file
            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during analysis: {e}")
                logger.warning(f"[{subject_id}] Expected error during analysis: {e}")


class MeshPipeline(DefaultPipeline):
    """Surface-mesh pipeline that stores per-subject :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` objects.

    Each result value is a :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData`
    instance that bundles:

    - The template-space cortical mesh (vertices + faces from TemplateFlow).
    - Per-vertex microstructural features loaded from ``.scalar.gii`` derivatives.
    - A vertex-wise parcellation label vector built from the Schaefer atlas
      already held in ``self.schaefer_resampled``.

    This pipeline is a **drop-in extension** of :class:`DefaultPipeline`: it
    reuses identical file discovery, verification, and resampling logic —
    only :meth:`run_analysis` is overridden to assemble
    :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` objects instead
    of flat arrays.

    The resulting objects are compatible with PyTorch Geometric (call
    :meth:`~diff_benchmark.data.surface_mesh.SurfaceMeshData.to_pyg`) and
    can be used directly for graph-based pooling using ``parcel_labels``.

    Args:
        dataset_config: Dataset configuration.  The ``tissue_type`` field must
            be ``"gray"`` (only cortical surface is supported).
        surface_type: Which surface geometry to load from TemplateFlow
            (``"midthickness"``, ``"inflated"``, ``"pial"``, ``"white"``).
            Defaults to ``"midthickness"`` as it is the standard choice for
            signal projection.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        surface_type: str = "midthickness",
    ):
        super().__init__(dataset_config)
        self.surface_type = surface_type
        self._template_mesh: dict | None = None  # lazily loaded

    # ------------------------------------------------------------------
    # Template mesh (shared across all subjects)
    # ------------------------------------------------------------------

    def _get_template_mesh(self) -> dict:
        """Load (and cache) the template-space surface geometry.

        Returns:
            Dict with keys ``"left_vertices"``, ``"left_faces"``,
            ``"right_vertices"``, ``"right_faces"`` (all numpy arrays).
        """
        if self._template_mesh is not None:
            return self._template_mesh

        logger.info(
            "Loading template surface (%s, %s)", self.surface_space, self.surface_type
        )
        lv, lf = load_template_surface(
            hemi="L", space=self.surface_space, surf_type=self.surface_type
        )
        rv, rf = load_template_surface(
            hemi="R", space=self.surface_space, surf_type=self.surface_type
        )
        self._template_mesh = {
            "left_vertices": lv,
            "left_faces": lf,
            "right_vertices": rv,
            "right_faces": rf,
        }
        return self._template_mesh

    # ------------------------------------------------------------------
    # Parcel label vector (shared across all subjects)
    # ------------------------------------------------------------------

    def _get_parcel_labels(self, n_left: int, n_right: int) -> np.ndarray:
        """Return a vertex-wise parcel label vector for the combined mesh.

        Args:
            n_left: Number of left-hemisphere vertices.
            n_right: Number of right-hemisphere vertices.

        Returns:
            ``(N_L + N_R,)`` int32 array of Schaefer parcel IDs.
        """
        return build_parcel_label_vector(
            self.schaefer_resampled, n_left=n_left, n_right=n_right
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def run_analysis(self) -> None:
        """Fill ``self.results`` with :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` objects.

        Iterates over existing ``.scalar.gii`` derivative files (same glob
        pattern as :class:`DefaultPipeline`), assembles a combined L+R mesh
        with per-vertex features and parcel labels, and stores the result
        keyed by subject ID.

        Only gray-matter (cortical) data is supported.  If ``tissue_type``
        is not ``"gray"``, a ``ValueError`` is raised.

        Raises:
            ValueError: If ``tissue_type != "gray"``.
        """
        tissue_type = self.dataset_config.tissue_type
        if tissue_type != "gray":
            raise ValueError(
                "MeshPipeline only supports tissue_type='gray' "
                f"(got '{tissue_type}').  Use DefaultPipeline for white matter."
            )

        # Load template geometry once
        tmesh = self._get_template_mesh()
        lv = tmesh["left_vertices"]
        lf = tmesh["left_faces"]
        rv = tmesh["right_vertices"]
        rf = tmesh["right_faces"]

        # Pre-compute parcel labels (same for all subjects in template space)
        parcel_labels = self._get_parcel_labels(
            n_left=lv.shape[0], n_right=rv.shape[0]
        )

        # Offset right-hemisphere faces
        n_left_verts = lv.shape[0]
        rf_offset = rf + n_left_verts

        # Combined template mesh arrays
        all_vertices = np.concatenate([lv, rv], axis=0)
        all_faces = np.concatenate([lf, rf_offset], axis=0)

        scalar_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}_tissue-{tissue_type}.scalar.gii"
            )
        )

        for left_file in tqdm(scalar_files, desc="Building mesh dataset"):
            try:
                subject_id = left_file.stem.split("_")[0].replace("sub-", "")
                right_file = left_file.with_name(
                    left_file.name.replace("hemi-L", "hemi-R")
                )

                if not right_file.exists():
                    logger.warning("[%s] Right scalar file missing, skipping", subject_id)
                    continue

                # Load and clip scalar data (same as DefaultPipeline)
                left_data = np.nan_to_num(
                    nib.load(left_file).darrays[0].data
                ).clip(0, 7).astype(np.float32)
                right_data = np.nan_to_num(
                    nib.load(right_file).darrays[0].data
                ).clip(0, 7).astype(np.float32)

                # Stack features: shape (N_L + N_R, 1)
                combined_features = np.concatenate(
                    [left_data[:, np.newaxis], right_data[:, np.newaxis]], axis=0
                )

                mesh = SurfaceMeshData(
                    vertices=all_vertices.copy(),
                    faces=all_faces.copy(),
                    features=combined_features,
                    parcel_labels=parcel_labels.copy(),
                    subject_id=subject_id,
                    metric=self.metric,
                    hemisphere="LR",
                )
                self.results[subject_id] = mesh

            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during mesh analysis: {e}")
                logger.warning(
                    "[%s] Expected error during mesh analysis: %s", subject_id, e
                )

    # ------------------------------------------------------------------
    # Export override — DataFrames don't hold meshes; return a plain index DF
    # ------------------------------------------------------------------

    def export_to_csv(self) -> pd.DataFrame:
        """Return a single-column DataFrame with ``subject_id`` as the index.

        The actual :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData`
        objects remain in ``self.results`` and are accessed directly by the
        dataset class via :meth:`get_mesh_results`.

        Returns:
            DataFrame with ``subject_id`` index and a dummy ``"mesh"`` column
            so the standard :meth:`load_features` / ``reset_index()`` call
            chain in :class:`~diff_benchmark.data.prepare_data.DatasetPreparation`
            still works.
        """
        if not self.results:
            raise ValueError("No results to save.")

        df = pd.DataFrame(
            {"mesh": list(self.results.keys())},
            index=pd.Index(list(self.results.keys()), name="subject_id"),
        )
        return df

    def get_mesh_results(self) -> dict:
        """Return the raw ``{subject_id: SurfaceMeshData}`` dictionary.

        Call this after :meth:`load_features` to retrieve the full mesh objects.

        Returns:
            Dict mapping subject ID strings to
            :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` instances.
        """
        return dict(self.results)

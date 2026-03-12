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
from diff_benchmark.preprocessing.utils.utils_graph_export import (
    build_mesh_stem,
    export_mesh_graph,
    mesh_parquet_paths,
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


class MeshPipeline(BrainDataPreparationPipeline):
    """Surface-mesh pipeline that stores per-subject :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` objects.

    Each result value is a :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData`
    instance that bundles:

    - The template-space cortical mesh (vertices + faces from TemplateFlow).
    - Per-vertex microstructural features loaded from ``.scalar.gii`` derivatives.
    - A vertex-wise parcellation label vector built from the Schaefer atlas
      already held in ``self.schaefer_resampled``.

    Unlike :class:`DefaultPipeline` or :class:`ImagePipeline`, this pipeline
    inherits directly from :class:`~diff_benchmark.preprocessing.preparation_pipeline.BrainDataPreparationPipeline`
    and stores results under a dedicated ``mesh/`` sub-directory so that mesh
    Parquet files never collide with flat-array derivatives.

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

    def _init_bids_layouts(self) -> None:
        """No-op override: MeshPipeline reads pre-computed Parquet/NPZ files
        and never accesses raw BIDS data, so no layout index is needed."""
        self.layouts = []

    def __init__(
        self,
        dataset_config: DatasetConfig,
        surface_type: str = "midthickness",
    ):
        super().__init__(dataset_config)
        breakpoint()
        self.surface_type = surface_type
        self._template_mesh: dict | None = None  # lazily loaded
        # Atlas identifier — always "schaefer" for the current pipeline;
        # n_parcels comes from dataset_config.scale (e.g. 1000).
        self.atlas_name: str = "schaefer"
        # Mesh pipeline keeps its own results root separate from "default/"
        self.results_root = Path(dataset_config.results_dir) / "mesh"
        # Root where microstructure (.scalar.gii / .nii.gz) files are stored
        # by DefaultPipeline / compute_microstructure — never written into mesh/
        self._default_root = Path(dataset_config.results_dir) / "default"

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _mesh_dwi_dir(self, subject_id: str) -> Path:
        """Return the mesh derivatives directory for *subject_id*.

        Output layout: ``<results_dir>/mesh/derivatives/sub-<id>/dwi/``
        """
        return self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"

    def _default_dwi_dir(self, subject_id: str) -> Path:
        """Return the default derivatives directory for *subject_id*.

        Microstructure files (``.scalar.gii``, ``.nii.gz``) are stored here.
        Layout: ``<results_dir>/default/derivatives/sub-<id>/dwi/``
        """
        return self._default_root / "derivatives" / f"sub-{subject_id}" / "dwi"

    # ------------------------------------------------------------------
    # Cache-check helpers
    # ------------------------------------------------------------------

    def _mesh_outputs_exist(self, subject_id: str) -> bool:
        """Return True if all three mesh output files already exist.

        Checks for BIDS-named files:

        - ``sub-<id>_param-<metric>_tissue-<tissue_type>_atlas-<prefix><n_parcels>_nodes.parquet``
        - ``sub-<id>_param-<metric>_tissue-<tissue_type>_atlas-<prefix><n_parcels>_edges.parquet``
        - ``sub-<id>_param-<metric>_tissue-<tissue_type>_atlas-<prefix><n_parcels>_mesh.npz``

        all inside ``mesh/derivatives/sub-<id>/dwi/``.
        """
        d = self._mesh_dwi_dir(subject_id)
        nodes_path, edges_path = mesh_parquet_paths(
            subject_id, d, self.metric, self.tissue_type,
            self.atlas_name, self.scale,
        )
        stem = build_mesh_stem(
            subject_id, self.metric, self.tissue_type, self.atlas_name, self.scale
        )
        return (
            nodes_path.exists()
            and edges_path.exists()
            and (d / f"{stem}_mesh.npz").exists()
        )

    def _microstructure_outputs_exist(self, subject_id: str) -> bool:
        """Return True if the scalar.gii microstructure files already exist.

        Looks inside ``default/derivatives/sub-<id>/dwi/`` for the left and
        right ``.scalar.gii`` files produced by
        :meth:`~diff_benchmark.preprocessing.preparation_pipeline.BrainDataPreparationPipeline.compute_microstructure`.
        """
        d = self._default_dwi_dir(subject_id)
        left = d / (
            f"sub-{subject_id}_hemi-L_param-{self.metric}"
            f"_tissue-{self.tissue_type}.scalar.gii"
        )
        right = d / (
            f"sub-{subject_id}_hemi-R_param-{self.metric}"
            f"_tissue-{self.tissue_type}.scalar.gii"
        )
        return left.exists() and right.exists()

    # ------------------------------------------------------------------
    # Abstract-method implementation — delegates to mesh outputs
    # ------------------------------------------------------------------

    def verify_subject_files(
        self, subject_id: str, metric: str, tissue_type: str
    ) -> bool:
        """Return True if all mesh output files exist for *subject_id*.

        The mesh pipeline considers a subject "done" when the Parquet graph
        files **and** the debug NPZ are all present under
        ``mesh/derivatives/sub-<id>/dwi/``.  This is what
        :meth:`~diff_benchmark.preprocessing.preparation_pipeline.BrainDataPreparationPipeline._process_subject`
        consults to decide whether to skip computation.

        Args:
            subject_id: Subject identifier.
            metric: Unused (kept for signature compatibility).
            tissue_type: Unused (kept for signature compatibility).

        Returns:
            True when nodes.parquet, edges.parquet and sub-<id>_mesh.npz all exist.
        """
        return self._mesh_outputs_exist(subject_id)

    # ------------------------------------------------------------------
    # Mesh validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_mesh(mesh: "SurfaceMeshData", subject_id: str) -> None:
        """Assert shape consistency of a :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData`.

        Args:
            mesh: Mesh object to validate.
            subject_id: Used in error messages.

        Raises:
            ValueError: On vertices/features/labels count mismatch or out-of-
                range face indices.
        """
        n = mesh.vertices.shape[0]
        if n != mesh.features.shape[0]:
            raise ValueError(
                f"[{subject_id}] vertices ({n}) != features ({mesh.features.shape[0]})"
            )
        if n != mesh.parcel_labels.shape[0]:
            raise ValueError(
                f"[{subject_id}] vertices ({n}) != parcel_labels "
                f"({mesh.parcel_labels.shape[0]})"
            )
        if mesh.faces.shape[0] > 0 and int(mesh.faces.max()) >= n:
            raise ValueError(
                f"[{subject_id}] face index {int(mesh.faces.max())} "
                f">= n_vertices {n}"
            )


    # ------------------------------------------------------------------
    # Debug NPZ export
    # ------------------------------------------------------------------

    @staticmethod
    def _export_debug_npz(
        mesh: "SurfaceMeshData",
        subject_id: str,
        output_dir: Path,
        metric: str,
        tissue_type: str,
        atlas_name: str = "schaefer",
        n_parcels: int = 1000,
    ) -> Path:
        """Save mesh arrays to a compressed ``.npz`` for offline inspection.

        Writes ``sub-{subject_id}_param-{metric}_tissue-{tissue_type}_atlas-<prefix>{n_parcels}_mesh.npz``
        containing arrays: ``vertices``, ``faces``, ``features``, ``parcel_labels``.

        Args:
            mesh: Source mesh object.
            subject_id: Used for file naming.
            output_dir: Directory to write into (created if absent).
            metric: Microstructure metric name (e.g. ``"ndi"``).
            tissue_type: Tissue type (e.g. ``"white"`` or ``"gray"``).
            atlas_name: Atlas name (e.g. ``"schaefer"``).
            n_parcels: Number of parcels (e.g. ``1000``).

        Returns:
            Path to the written file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = build_mesh_stem(subject_id, metric, tissue_type, atlas_name, n_parcels)
        npz_path = output_dir / f"{stem}_mesh.npz"
        np.savez_compressed(
            npz_path,
            vertices=mesh.vertices,
            faces=mesh.faces,
            features=mesh.features,
            parcel_labels=mesh.parcel_labels,
        )
        return npz_path

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _save_visualizations(
        self, mesh: "SurfaceMeshData", subject_id: str, mesh_cfg
    ) -> None:
        """Render and save visualisation plots for *subject_id*.

        Outputs go to ``./exp_outputs/mesh/``.

        Args:
            mesh: The mesh to visualise.
            subject_id: Subject identifier (used for file naming).
            mesh_cfg: Hydra ``mesh_pipeline`` config node exposing
                ``visualization_method`` (``"plotly"`` or ``"nilearn"``).
        """
        from diff_benchmark.preprocessing.utils.utils_mesh_visualization import (
            plot_mesh_nilearn,
            plot_mesh_plotly,
        )

        viz_dir = Path("./exp_outputs/mesh")
        viz_dir.mkdir(parents=True, exist_ok=True)

        graph_dir = self._mesh_dwi_dir(subject_id)
        method = getattr(mesh_cfg, "visualization_method", "plotly")

        if method == "plotly":
            html_path = viz_dir / f"sub-{subject_id}_plotly.html"
            plot_mesh_plotly(
                subject_id=subject_id,
                graph_dir=graph_dir,
                show_edges=False,
                output_html=html_path,
            )
        elif method == "nilearn":
            png_path = viz_dir / f"sub-{subject_id}_nilearn.png"
            plot_mesh_nilearn(
                mesh,
                mode="feature",
                hemi="left",
                view="lateral",
                output_file=png_path,
            )
        else:
            logger.warning(
                "[MeshPipeline] [%s] Unknown visualization_method '%s' — skipping",
                subject_id, method,
            )

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

    def run_analysis(
        self,
        subject_filter: str | None = None,
        mesh_cfg=None,
    ) -> None:
        """Fill ``self.results`` with :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` objects.

        Two-tier cache logic per subject
        ---------------------------------
        1. **All mesh outputs present** (nodes.parquet + edges.parquet + mesh.npz)
           → load the mesh from Parquet and skip all computation.
        2. **Microstructure outputs present** (.scalar.gii in default/) but mesh
           outputs missing → build the mesh from cached scalar files, then
           export graph + NPZ.
        3. **Nothing present** → this should not happen if called after
           :meth:`~diff_benchmark.preprocessing.preparation_pipeline.BrainDataPreparationPipeline.run_pipeline`
           which calls :meth:`~diff_benchmark.preprocessing.preparation_pipeline.BrainDataPreparationPipeline.compute_microstructure`
           first.  A warning is logged and the subject is skipped.

        Microstructure files (``.scalar.gii``) are **always** read from
        ``<results_dir>/default/derivatives/sub-<id>/dwi/`` and mesh outputs
        are **always** written to
        ``<results_dir>/mesh/derivatives/sub-<id>/dwi/``.

        Args:
            subject_filter: When provided, process only this subject ID.
            mesh_cfg: Hydra ``mesh_pipeline`` config node.  Controls
                ``export_graph``, ``export_debug_mesh``, and
                ``run_visualization`` flags.  When ``None`` all export/viz
                steps are performed with their defaults (export on, viz off).

        Raises:
            ValueError: If ``tissue_type != "gray"``.
        """
        from diff_benchmark.preprocessing.utils.utils_graph_export import (
            load_graph_from_parquet,
        )

        tissue_type = self.dataset_config.tissue_type
        if tissue_type != "gray":
            raise ValueError(
                "MeshPipeline only supports tissue_type='gray' "
                f"(got '{tissue_type}').  Use DefaultPipeline for white matter."
            )

        # Resolve config flags (safe defaults when mesh_cfg is None)
        do_export_graph = True if mesh_cfg is None else bool(mesh_cfg.export_graph)
        do_export_npz = True if mesh_cfg is None else bool(mesh_cfg.export_debug_mesh)
        do_viz = False if mesh_cfg is None else bool(mesh_cfg.run_visualization)

        # Load template geometry once (shared across all subjects)
        tmesh = self._get_template_mesh()
        lv = tmesh["left_vertices"]
        lf = tmesh["left_faces"]
        rv = tmesh["right_vertices"]
        rf = tmesh["right_faces"]

        parcel_labels = self._get_parcel_labels(
            n_left=lv.shape[0], n_right=rv.shape[0]
        )
        n_left_verts = lv.shape[0]
        rf_offset = rf + n_left_verts
        all_vertices = np.concatenate([lv, rv], axis=0)
        all_faces = np.concatenate([lf, rf_offset], axis=0)

        # ------------------------------------------------------------------
        # Determine which subjects to process
        # ------------------------------------------------------------------
        if subject_filter is not None:
            subject_ids = [subject_filter]
        else:
            # Preferred: discover subjects from existing mesh output directories.
            # This allows mesh-only execution (e.g. on a cluster) where the
            # default/ tree is absent but parquet/npz files are already present.
            mesh_subject_dirs = sorted(
                self.results_root.glob("derivatives/sub-*")
            )
            if mesh_subject_dirs:
                subject_ids = [
                    d.name.replace("sub-", "") for d in mesh_subject_dirs if d.is_dir()
                ]
            else:
                # Fallback: discover subjects from default derivatives (scalar.gii glob)
                scalar_glob = sorted(
                    self._default_root.glob(
                        f"derivatives/sub-*/dwi/"
                        f"*_hemi-L_param-{self.metric}_tissue-{tissue_type}.scalar.gii"
                    )
                )
                subject_ids = [
                    p.stem.split("_")[0].replace("sub-", "") for p in scalar_glob
                ]

        # ------------------------------------------------------------------
        # Per-subject loop
        # ------------------------------------------------------------------
        for subject_id in tqdm(subject_ids, desc="Building mesh dataset"):
            mesh_dir = self._mesh_dwi_dir(subject_id)

            # ---- Tier 1: all mesh outputs already present ---------------
            if self._mesh_outputs_exist(subject_id):
                # Load from Parquet so self.results is populated
                try:
                    mesh = load_graph_from_parquet(
                        subject_id=subject_id,
                        graph_dir=mesh_dir,
                        metric=self.metric,
                        tissue_type=tissue_type,
                        atlas_name=self.atlas_name,
                        n_parcels=self.scale,
                    )
                    self.results[subject_id] = mesh
                except Exception as load_err:
                    logger.warning(
                        "[MeshPipeline] [%s] Failed to load cached mesh: %s",
                        subject_id, load_err,
                    )
                if do_viz:
                    try:
                        self._save_visualizations(
                            self.results[subject_id], subject_id, mesh_cfg
                        )
                    except Exception as viz_err:
                        logger.warning(
                            "[MeshPipeline] [%s] Visualization failed: %s",
                            subject_id, viz_err,
                        )
                continue

            # ---- Tier 2: microstructure cached, mesh missing ------------
            default_dir = self._default_dwi_dir(subject_id)
            left_file = default_dir / (
                f"sub-{subject_id}_hemi-L_param-{self.metric}"
                f"_tissue-{tissue_type}.scalar.gii"
            )
            right_file = default_dir / (
                f"sub-{subject_id}_hemi-R_param-{self.metric}"
                f"_tissue-{tissue_type}.scalar.gii"
            )

            if self._microstructure_outputs_exist(subject_id):
                logger.info(
                    "[MeshPipeline] Found cached microstructure files for sub-%s"
                    " — skipping recomputation",
                    subject_id,
                )
                print(
                    f"[MeshPipeline] Found cached microstructure files for sub-{subject_id}"
                    " — skipping recomputation"
                )
            else:
                # ---- Tier 3: nothing cached — should have been computed by
                #              run_pipeline() first; log and skip gracefully
                logger.warning(
                    "[MeshPipeline] Missing mesh outputs — generating mesh for sub-%s"
                    " (no cached microstructure found; run run_pipeline() first)",
                    subject_id,
                )
                print(
                    f"[MeshPipeline] Missing mesh outputs — generating mesh for sub-{subject_id}"
                    " (microstructure files not found in default/derivatives)"
                )
                if not left_file.exists():
                    logger.warning(
                        "[MeshPipeline] [%s] Scalar files missing — skipping subject",
                        subject_id,
                    )
                    continue

            # ---- Build SurfaceMeshData from scalar.gii files ------------
            try:
                if not right_file.exists():
                    logger.warning(
                        "[MeshPipeline] [%s] Right scalar file missing, skipping",
                        subject_id,
                    )
                    continue

                logger.info(
                    "[MeshPipeline] [%s] Loading scalar data from %s",
                    subject_id, default_dir,
                )
                left_data = np.nan_to_num(
                    nib.load(left_file).darrays[0].data
                ).clip(0, 7).astype(np.float32)
                right_data = np.nan_to_num(
                    nib.load(right_file).darrays[0].data
                ).clip(0, 7).astype(np.float32)

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
                    n_left_vertices=lv.shape[0],
                )

                # Validate before exporting
                self._validate_mesh(mesh, subject_id)

                self.results[subject_id] = mesh

                # ---- Export graph (Parquet) --------------------------------
                if do_export_graph:
                    logger.info(
                        "[MeshPipeline] [%s] Exporting graph parquet", subject_id
                    )
                    print(f"[MeshPipeline] [{subject_id}] Exporting graph parquet")
                    try:
                        export_mesh_graph(
                            mesh=mesh,
                            subject_id=subject_id,
                            output_dir=mesh_dir,
                            metric=self.metric,
                            tissue_type=tissue_type,
                            atlas_name=self.atlas_name,
                            n_parcels=self.scale,
                            overwrite=False,
                        )
                    except Exception as export_err:
                        logger.warning(
                            "[MeshPipeline] [%s] Graph export failed: %s",
                            subject_id, export_err,
                        )

                # ---- Export debug NPZ -------------------------------------
                if do_export_npz:
                    try:
                        self._export_debug_npz(
                            mesh, subject_id, mesh_dir,
                            metric=self.metric,
                            tissue_type=tissue_type,
                            atlas_name=self.atlas_name,
                            n_parcels=self.scale,
                        )
                    except Exception as npz_err:
                        logger.warning(
                            "[MeshPipeline] [%s] NPZ export failed: %s",
                            subject_id, npz_err,
                        )

                # ---- Visualization ----------------------------------------
                if do_viz:
                    try:
                        self._save_visualizations(mesh, subject_id, mesh_cfg)
                    except Exception as viz_err:
                        logger.warning(
                            "[MeshPipeline] [%s] Visualization failed: %s",
                            subject_id, viz_err,
                        )

            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during mesh analysis: {e}")
                logger.warning(
                    "[MeshPipeline] [%s] Expected error during mesh analysis: %s",
                    subject_id, e,
                )

    # ------------------------------------------------------------------
    # Export override — DataFrames don't hold meshes; return a plain index DF
    # ------------------------------------------------------------------

    def export_to_csv(self) -> pd.DataFrame:
        """Return a single-column DataFrame with ``subject_id`` as the index.

        The actual :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData`
        objects remain in ``self.results``; parquet file paths are exposed via
        :meth:`get_mesh_parquet_paths` and consumed by
        :class:`~diff_benchmark.data.generate_dataset.CustomDataset`.

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

    def get_mesh_parquet_paths(self) -> dict:
        """Return per-subject paths to the BIDS-named nodes/edges Parquet files.

        Returns:
            Dict mapping subject ID strings to a sub-dict::

                {
                    "nodes": Path(..._nodes.parquet),
                    "edges": Path(..._edges.parquet),
                }

        Only subjects whose output files exist on disk are included.
        """
        paths: dict = {}
        for subject_id in self.results:
            mesh_dir = self._mesh_dwi_dir(subject_id)
            nodes_path, edges_path = mesh_parquet_paths(
                subject_id, mesh_dir, self.metric, self.tissue_type,
                self.atlas_name, self.scale,
            )
            if nodes_path.exists() and edges_path.exists():
                paths[subject_id] = {
                    "nodes": nodes_path,
                    "edges": edges_path,
                }
        return paths

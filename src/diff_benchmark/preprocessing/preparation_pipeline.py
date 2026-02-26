"""Brain data preparation pipeline — core abstract base class.

The concrete classes for demographics loading (``DemographicsPreparationPipeline``,
``CachedBIDSFile``, ``CachedBIDSLayout``) and I/O dataclasses
(``DiffusionInputs``, ``ProcessingResult``, ``process_subject_wrapper``) have
been extracted to separate modules for clarity.  They are re-exported here for
backward compatibility so that existing import sites continue to work unchanged.

Modules:
    :mod:`diff_benchmark.preprocessing.pipeline_io` – I/O dataclasses and the
        joblib/SLURM subject-wrapper function.
    :mod:`diff_benchmark.preprocessing.demographics_pipeline` – demographics
        loading, normalisation, and BIDS-cache helpers.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import bids
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.filebasedimages import ImageFileError
from nilearn import image as nimage

from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.demographics_pipeline import (  # noqa: F401
    COLUMN_ALIASES,
    CachedBIDSFile,
    CachedBIDSLayout,
    DemographicsPreparationPipeline,
)
from diff_benchmark.preprocessing.pipeline_io import (  # noqa: F401
    DiffusionInputs,
    ProcessingResult,
    process_subject_wrapper,
)
from diff_benchmark.preprocessing.utils.utils_brain_feature_extraction import (
    compute_save_and_project_metric,
    create_masks,
    extract_selected_labels,
    resample_schaefer_onto_fs_lr,
)
from diff_benchmark.utils.job_manager import run_jobs
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_subject_ids(dataset: DatasetConfig) -> list[str]:
    """Derive a sorted list of subject IDs by globbing the dataset base dir.

    Args:
        dataset: The dataset configuration object.

    Returns:
        Sorted list of subject ID strings (without ``sub-`` prefix for BIDS).

    Raises:
        ValueError: If ``dataset.data_reading`` is not a recognised key.
    """
    base = Path(dataset.base_dir)

    glob_patterns: dict[str, str] = {
        "multicenter-bids": "*/sub-*",
        "bids": "sub-*",
        "hcp": "*",
    }

    try:
        pattern = glob_patterns[dataset.data_reading]
    except KeyError:
        raise ValueError(f"Unknown data_reading: {dataset.data_reading}")

    subjects = []
    for p in base.glob(pattern):
        name = p.name
        sid = name if dataset.data_reading == "hcp" else name[4:]
        subjects.append(sid)

    return sorted(subjects)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BrainDataPreparationPipeline(ABC):
    """Abstract base class for brain-data preparation and microstructure analysis.

    Subclasses must implement:
    - :meth:`verify_subject_files` – check whether derived files exist.
    - :meth:`run_analysis` – load derived files and populate ``self.results``.

    Optionally override:
    - :meth:`verify_resampling` / :meth:`resample_data` – surface-space
      resampling hooks (default implementations are no-ops).
    """

    def __init__(self, dataset_config: DatasetConfig):
        self.tissue_type = dataset_config.tissue_type
        self.dataset_config = dataset_config
        self.data_reading = dataset_config.data_reading
        self.base_dir = Path(dataset_config.base_dir)
        self.in_derivatives = self.base_dir / "derivatives"
        self.results_dir = Path(dataset_config.results_dir)
        self.metric = dataset_config.metric_to_compute
        self.scale = dataset_config.scale
        self.surface_space = dataset_config.surface_space
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(
            self.scale, target_space=self.surface_space
        )
        self.big_delta = dataset_config.big_delta
        self.small_delta = dataset_config.small_delta
        self.big_delta_per_bvalue = dataset_config.big_delta_per_bvalue
        self.results: dict = {}

        self.dwi_desc = dataset_config.dwi_desc
        self.bvec_extensions = dataset_config.bvec_extensions
        self.bval_extensions = dataset_config.bval_extensions
        self.nodif_mask_extension = dataset_config.nodif_mask_extension
        self.aparcaseg_extension = dataset_config.aparcaseg_extension

        if "bids" in self.data_reading:
            self._init_bids_layouts()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_bids_layouts(self) -> None:
        """Build (or reload from cache) one :class:`bids.BIDSLayout` per centre.

        For unicentre datasets the base dir is used directly; for multicentre
        datasets each subdirectory is treated as a separate centre.
        """
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True, exist_ok=True)

        if (self.base_dir / "sub-").exists() or any(
            p.name.startswith("sub-") for p in self.base_dir.iterdir()
        ):
            center_dirs = [self.base_dir]
        else:
            center_dirs = [p for p in self.base_dir.iterdir() if p.is_dir()]

        self.layouts: list[bids.BIDSLayout] = []
        for center in center_dirs:
            if len(center_dirs) > 1:
                db_name = f"bids_layout_{self.dataset_config.name}_{center.name}.db"
            else:
                db_name = f"bids_layout_{self.dataset_config.name}.db"

            database_path = self.results_dir / db_name
            logger.info("Using BIDS layout database: %s", database_path)

            self.layouts.append(
                bids.BIDSLayout(
                    str(center),
                    derivatives=center / "derivatives",
                    validate=False,
                    database_path=database_path,
                )
            )

    def get_subjects(self) -> list[str]:
        """
        Retrieve a sorted list of unique subject identifiers from the layouts.
        This method aggregates subject identifiers from all layouts associated
        with the instance, removes duplicates, and returns them in sorted order.
        Returns:
            list[str]: A sorted list of unique subject identifiers.
        """

        return sorted(
            {subject for layout in self.layouts for subject in layout.get_subjects()}
        )

    def get_layout_for_subject(self, subject_id: str) -> bids.BIDSLayout:
        """Return the layout that contains *subject_id*.

        Args:
            subject_id: BIDS subject identifier (without ``sub-`` prefix).

        Raises:
            ValueError: If the subject is not found in any registered layout.
        """
        for layout in self.layouts:
            if subject_id in layout.get_subjects():
                return layout
        raise ValueError(f"Subject {subject_id} not found in any center")

    # ------------------------------------------------------------------
    # Raw file discovery
    # ------------------------------------------------------------------

    def _get_required_raw_files(self, subject_id: str) -> DiffusionInputs | None:
        """Return a :class:`DiffusionInputs` for *subject_id*, or ``None`` on failure.

        Dispatches to :meth:`_get_hcp_raw_files` or
        :meth:`_get_bids_raw_files` based on ``self.data_reading``.
        """
        try:
            if self.data_reading == "hcp":
                return self._get_hcp_raw_files(subject_id)
            if "bids" in self.data_reading:
                return self._get_bids_raw_files(subject_id)
        except (IndexError, KeyError):
            return None
        return None

    def _get_hcp_raw_files(self, subject_id: str) -> DiffusionInputs:
        """Build :class:`DiffusionInputs` for an HCP-structured subject directory.

        Args:
            subject_id: Subject folder name inside ``self.base_dir``.

        Returns:
            Populated :class:`DiffusionInputs`.
        """
        subject_dir = self.base_dir / subject_id
        diffusion_dir = subject_dir / "T1w" / "Diffusion"

        surfaces = {
            f"{h}.{s}": subject_dir
            / "T1w"
            / "fsaverage_LR32k"
            / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
            for s in ("white", "pial")
            for h in ("L", "R")
        }

        return DiffusionInputs(
            dwi_data=diffusion_dir / self.dwi_desc,
            bvals=diffusion_dir / self.bval_extensions,
            bvecs=diffusion_dir / self.bvec_extensions,
            nodif_mask=diffusion_dir / self.nodif_mask_extension,
            aparc_aseg=subject_dir / self.aparcaseg_extension,
            surfaces=surfaces,
        )

    def _get_bids_raw_files(self, subject_id: str) -> DiffusionInputs:
        """Build :class:`DiffusionInputs` by querying the BIDS layout.

        Args:
            subject_id: BIDS subject ID (without ``sub-`` prefix).

        Returns:
            Populated :class:`DiffusionInputs`.
        """
        layout = self.get_layout_for_subject(subject_id)

        aparcaseg = layout.get(
            subject=subject_id,
            desc="aparcaseg",
            suffix="dseg",
            return_type="file",
        )[0]

        dwi_bids = layout.get(
            subject=subject_id,
            suffix="dwi",
            extension=".nii.gz",
            desc=self.dwi_desc,
        )[0]

        entities = dwi_bids.get_entities()

        bvals = layout.get(
            subject=entities["subject"],
            extension=self.bval_extensions,
            return_type="file",
        )[0]

        bvecs = layout.get(
            subject=entities["subject"],
            extension=self.bvec_extensions,
            return_type="file",
        )[0]

        surfaces = {
            f"{h}.{s}": Path(
                layout.get(
                    subject=subject_id,
                    suffix=s,
                    hemi=h,
                    space=None,
                    extension=".surf.gii",
                    return_type="files",  # , density='32k'
                )[0]
            )
            for s in ("white", "pial", "inflated")
            for h in ("L", "R")
        }

        return DiffusionInputs(
            dwi_data=Path(dwi_bids.path),
            bvals=Path(bvals),
            bvecs=Path(bvecs),
            aparc_aseg=Path(aparcaseg),
            nodif_mask=None,
            surfaces=surfaces,
        )

    def verify_raw_files(self, subject_id: str) -> bool:
        """Return ``True`` if every file in :meth:`_get_required_raw_files` exists and is non-empty.

        Args:
            subject_id: Subject identifier.

        Returns:
            ``False`` when any required file is missing, empty, or the method
            returns ``None``; ``True`` otherwise.
        """
        required_files = self._get_required_raw_files(subject_id)
        if required_files is None:
            logger.warning("[SKIP] %s: required raw files not found", subject_id)
            return False

        missing_or_empty = []
        for name, path in required_files.iter_paths().items():
            if not path.exists():
                missing_or_empty.append(f"{name} (missing)")
            elif path.is_file() and path.stat().st_size == 0:
                missing_or_empty.append(f"{name} (empty)")

        if missing_or_empty:
            logger.warning(
                "[WARNING] Missing or empty files for subject %s: %s",
                subject_id,
                ", ".join(missing_or_empty),
            )
            return False

        return True

    @abstractmethod
    def verify_subject_files(
        self, subject_id: str, metric: str, tissue_type: str
    ) -> bool:
        """Return ``True`` if all derived (post-microstructure) files exist for *subject_id*.

        Args:
            subject_id: Subject identifier.
            metric: Microstructure metric being checked (e.g. ``"rtop"``).
            tissue_type: Tissue type (``"gray"`` or ``"white"``).

        Returns:
            Boolean indicating whether all expected derivative files are present.
        """

    # ------------------------------------------------------------------
    # Microstructure computation
    # ------------------------------------------------------------------

    def _prepare_derivatives_dir(self, subject_id: str) -> Path:
        """Create and return the per-subject derivatives directory.

        Args:
            subject_id: Subject identifier.

        Returns:
            :class:`Path` to the created directory.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        return derivatives_dir

    def _resample_parcellation_to_dwi(
        self,
        files: DiffusionInputs,
        dwi_nib: nib.Nifti1Image,
    ) -> nib.Nifti1Image:
        """Resample the aparc+aseg parcellation into DWI voxel space.

        HCP data uses the *nodif_mask* as resampling reference; BIDS data uses
        the DWI image affine/shape directly.

        Args:
            files: :class:`DiffusionInputs` for the subject.
            dwi_nib: Loaded DWI NIfTI image (used as resampling reference for BIDS).

        Returns:
            Resampled parcellation as a :class:`nib.Nifti1Image`.
        """
        if self.data_reading == "hcp":
            return nimage.resample_to_img(
                files.aparc_aseg,
                files.nodif_mask,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

        # BIDS branch
        return nimage.resample_img(
            files.aparc_aseg,
            target_affine=dwi_nib.affine,
            target_shape=dwi_nib.shape[:3],
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

    def _select_tissue_labels(
        self, labels: dict, tissue_type: str
    ) -> list | None:
        """Return a filtered list of parcel labels for the requested tissue type.

        For ``"gray"`` tissue only cortical and subcortical labels are kept;
        for ``"white"`` all labels are used (returns ``None`` to signal no
        filtering).  HCP data always returns ``None`` (handled upstream).

        Args:
            labels: Mapping returned by :func:`extract_selected_labels`.
            tissue_type: ``"gray"`` or ``"white"``.

        Returns:
            Filtered label list, or ``None`` when no filtering is needed.
        """
        if tissue_type == "gray":
            return [
                k
                for k in labels
                if (
                    ("ctx" in k)
                    or ("thalamus" in k)
                    or ("caudate" in k)
                    or ("putamen" in k)
                    or ("pallidum" in k)
                )
            ]
        # white matter or HCP: no label filtering
        return None

    def compute_microstructure(self, subject_id: str) -> None:
        """Compute microstructure maps and project them to the surface/skeleton.

        This method:
        1. Creates the output derivatives directory.
        2. Loads raw inputs (DWI image, b-values, b-vectors, parcellation).
        3. Resamples the parcellation to DWI space.
        4. Selects tissue-specific parcel labels and builds tissue masks.
        5. Calls :func:`compute_save_and_project_metric` to compute, save, and
           project the requested metric.

        Args:
            subject_id: Subject identifier.
        """
        try:
            derivatives_dir = self._prepare_derivatives_dir(subject_id)

            files = self._get_required_raw_files(subject_id)
            assert files is not None

            dwi_nib = nib.load(files.dwi_data)
            bvals = np.loadtxt(files.bvals)
            labels = extract_selected_labels(files.aparc_aseg, tissue_type=self.tissue_type)

            bvecs = (
                np.loadtxt(files.bvecs).T
                if self.data_reading == "hcp"
                else np.loadtxt(files.bvecs)
            )

            aparc_resampled = self._resample_parcellation_to_dwi(files, dwi_nib)

            selected_labels = (
                None
                if self.data_reading == "hcp"
                else self._select_tissue_labels(labels, self.tissue_type)
            )

            ctx_mask, vent_mask = create_masks(
                aparc_resampled, labels, selected_labels, tissue_type=self.tissue_type
            )

            compute_save_and_project_metric(
                metric=self.metric,
                dwi_nib=dwi_nib,
                ctx_mask=ctx_mask,
                vent_mask=vent_mask,
                bvals=bvals,
                bvecs=bvecs,
                big_delta=self.big_delta,
                small_delta=self.small_delta,
                big_delta_per_bvalue=self.big_delta_per_bvalue,
                surfaces=files.surfaces,
                derivatives_dir=derivatives_dir,
                subject_id=subject_id,
                layouts=getattr(self, "layouts", None),
                target_space=getattr(self, "surface_space", "fslr_32k"),
                data_reading=self.data_reading,
                tissue_type=self.tissue_type,
            )
        except (
            FileNotFoundError,
            OSError,
            ImageFileError,
            KeyError,
            ValueError,
            IndexError,
        ) as e:
            print(f"[{subject_id}] Expected error during microstructure: {e}")
            logger.error("[%s] Expected error during microstructure: %s", subject_id, e)

    # ------------------------------------------------------------------
    # Analysis hooks (subclass responsibility)
    # ------------------------------------------------------------------

    @abstractmethod
    def run_analysis(self) -> None:
        """Load derived files and populate ``self.results``.

        Subclasses must implement this method.
        """

    # ------------------------------------------------------------------
    # Resampling hooks (optional override)
    # ------------------------------------------------------------------

    def verify_resampling(self, subject_id: str) -> bool:
        """Return ``True`` if derived data is already in template space.

        Default implementation always returns ``True``.  Override in subclasses
        that produce data in native space.

        Args:
            subject_id: Subject identifier.
        """
        return True

    def resample_data(self, subject_id: str) -> None:
        """Resample derived data from native to template space.

        Default implementation is a no-op.  Override when an explicit
        resampling step is required.

        Args:
            subject_id: Subject identifier.
        """

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_csv(self) -> pd.DataFrame:
        """Collect ``self.results`` into a ``DataFrame`` indexed by subject ID.

        Returns:
            ``DataFrame`` with one row per subject.

        Raises:
            ValueError: When ``self.results`` is empty.
        """
        if not self.results:
            raise ValueError("No results to save.")
        print("Exporting data to DataFrame...")
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.index.name = "subject_id"
        return df

    # ------------------------------------------------------------------
    # Per-subject processing logic
    # ------------------------------------------------------------------

    def _process_subject(self, subject_id: str, recompute: bool) -> None:
        """Compute microstructure and apply resampling for a single subject.

        Decision logic:
        - Skip immediately if raw files are missing.
        - Compute (or recompute) microstructure as needed.
        - Verify resampling; apply it if not already done.

        Args:
            subject_id: Subject identifier.
            recompute: When ``True``, recompute even if derived files exist.
        """
        if not self.verify_raw_files(subject_id):
            logger.warning("[%s] Missing raw files, skipping", subject_id)
            return

        files_present = self.verify_subject_files(subject_id, self.metric, self.tissue_type)
        computed_microstructure = False

        if files_present and recompute:
            logger.info(
                "[%s] Recomputing microstructure for %s matter.", subject_id, self.tissue_type
            )
            print(f"[{subject_id}] Recomputing microstructure for {self.tissue_type} matter.")
            self.compute_microstructure(subject_id)
            computed_microstructure = True
        elif not files_present:
            logger.info(
                "[%s] Computing microstructure for %s matter.", subject_id, self.tissue_type
            )
            print(f"[{subject_id}] Computing microstructure for {self.tissue_type} matter.")
            self.compute_microstructure(subject_id)
            computed_microstructure = True
        else:
            logger.info(
                "[%s] Microstructure files for %s matter already present.",
                subject_id,
                self.tissue_type,
            )
            print(
                f"[{subject_id}] Microstructure files for {self.tissue_type} matter already present."
            )

        if computed_microstructure:
            if not self.verify_resampling(subject_id):
                logger.warning(
                    "[%s] Data was just computed but resampling check failed, attempting resampling",
                    subject_id,
                )
                print(
                    f"[{subject_id}] Data was just computed but resampling check failed, attempting resampling"
                )
                self.resample_data(subject_id)
            else:
                logger.debug("[%s] Data properly resampled during computation", subject_id)
                print(f"[{subject_id}] Data properly resampled during computation")
        else:
            if not self.verify_resampling(subject_id):
                logger.info("[%s] Existing data needs resampling, resampling now", subject_id)
                print(f"[{subject_id}] Existing data needs resampling, resampling now")
                self.resample_data(subject_id)
            else:
                logger.debug("[%s] Existing data already properly resampled", subject_id)
                print(f"[{subject_id}] Existing data already properly resampled")

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def run_pipeline(
        self, cluster_conf: Dict, slurm_cfg: Dict, recompute: bool = False
    ) -> None:
        """Dispatch per-subject processing jobs then signal readiness.

        Args:
            cluster_conf: Cluster/parallelism config object (must expose
                ``parallel_type``, ``n_jobs``, ``wait_for_results``).
            slurm_cfg: SLURM job configuration passed to :func:`run_jobs`.
            recompute: When ``True``, recompute microstructure even if derived
                files already exist.
        """
        subject_list = _parse_subject_ids(self.dataset_config)

        parallel_type = (
            None
            if cluster_conf.parallel_type not in ["slurm", "joblib"]
            else cluster_conf.parallel_type
        )

        run_jobs(
            run_fn=process_subject_wrapper,
            fn_kwargs_list=[
                {
                    "subject_id": subject_id,
                    "pipeline_cls": type(self),
                    "dataset_config": self.dataset_config,
                    "recompute": recompute,
                }
                for subject_id in subject_list
            ],
            parallel_type=parallel_type,
            slurm_cfg=slurm_cfg,
            n_jobs=cluster_conf.n_jobs,
            wait_for_results=cluster_conf.wait_for_results,
        )

        print("All required files are ready. Now you can run analysis!")
        logger.info("All required files are ready. Now you can run analysis!")

    def load_features(self) -> pd.DataFrame:
        """Run analysis and return results as a ``DataFrame``.

        Assumes microstructure files are already present (call
        :meth:`run_pipeline` first).

        Returns:
            ``DataFrame`` produced by :meth:`export_to_csv`.
        """
        logger.info(
            "All data should be preprocessed already. Getting microstructure files..."
        )
        self.run_analysis()
        print("Computing df...")
        return self.export_to_csv()

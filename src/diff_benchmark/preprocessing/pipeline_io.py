"""I/O dataclasses and subject-processing helpers for the preparation pipeline.

Contains:
- :class:`DiffusionInputs` – paths to all raw files needed for one subject.
- :class:`ProcessingResult` – accumulates valid/invalid subject IDs.
- :func:`process_subject_wrapper` – top-level (picklable) wrapper used by
  joblib/SLURM parallelism.

These are separated from :mod:`preparation_pipeline` so that the main pipeline
module stays focused on orchestration logic and to make the dataclasses
independently importable without pulling in the full pipeline class hierarchy.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class DiffusionInputs:
    """Paths to all raw files required to process a single DWI subject.

    Attributes:
        dwi_data: Path to the DWI 4-D NIfTI file.
        bvals: Path to the b-values text file.
        bvecs: Path to the b-vectors text file.
        aparc_aseg: Path to the FreeSurfer ``aparc+aseg`` segmentation.
        nodif_mask: Optional path to the b0 (no-diffusion) brain mask.
        surfaces: Mapping of surface identifiers to GIFTI surface paths.
            Common keys: ``"L.white"``, ``"L.pial"``, ``"R.white"``,
            ``"R.pial"``.
    """

    dwi_data: Path
    bvals: Path
    bvecs: Path
    aparc_aseg: Path
    nodif_mask: Path | None = None
    surfaces: Dict[str, Path] = field(default_factory=dict)

    def iter_paths(self) -> Dict[str, Path]:
        """Return all relevant file paths as a flat dictionary.

        Surfaces are included with the key prefix ``"surface:"``.

        Returns:
            Dict mapping name → Path for every required file (and optional
            ``nodif_mask`` when provided).
        """
        base = {
            "dwi_data": self.dwi_data,
            "bvals": self.bvals,
            "bvecs": self.bvecs,
            "aparc_aseg": self.aparc_aseg,
        }
        if self.nodif_mask is not None:
            base["nodif_mask"] = self.nodif_mask
        return {**base, **{f"surface:{k}": v for k, v in self.surfaces.items()}}


@dataclass
class ProcessingResult:
    """Accumulates valid and invalid subject IDs during a pipeline run.

    Attributes:
        valid_subjects: Subject IDs that completed processing successfully.
        invalid_subjects: Subject IDs that failed or were skipped.
    """

    valid_subjects: list[str] = field(default_factory=list)
    invalid_subjects: list[str] = field(default_factory=list)

    def add_valid(self, subject_id: str) -> None:
        """Mark a subject as successfully processed."""
        self.valid_subjects.append(subject_id)

    def add_invalid(self, subject_id: str) -> None:
        """Mark a subject as failed or skipped."""
        self.invalid_subjects.append(subject_id)


def process_subject_wrapper(subject_id, pipeline_cls, dataset_config, recompute):
    """Top-level wrapper so joblib/SLURM can pickle the call.

    Instantiates a fresh pipeline from ``pipeline_cls`` and ``dataset_config``
    and delegates to :meth:`BrainDataPreparationPipeline._process_subject`.

    Args:
        subject_id: Subject identifier string.
        pipeline_cls: Concrete subclass of
            :class:`~preparation_pipeline.BrainDataPreparationPipeline`.
        dataset_config: :class:`~datasets_dataclasses.DatasetConfig` instance.
        recompute: Whether to recompute even when output files already exist.
    """
    # A new pipeline instance is created here deliberately so the object is
    # not shared across processes (avoids pickle issues with BIDS layouts).
    # pipeline._process_subject(subject_id, recompute)  # old direct-instance call
    pipeline = pipeline_cls(dataset_config)
    pipeline._process_subject(subject_id, recompute)

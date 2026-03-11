from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.utils.job_manager import run_jobs
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


def resample_subject_wrapper(subject_id, pipeline_cls, dataset_config):
    """
    Wrapper to call the resample method safely.
    This function is top-level, so joblib can pickle it.
    """
    pipeline = pipeline_cls(dataset_config)
    pipeline.resample_data(subject_id)


@hydra.main(
    version_base="1.3",
    config_path="pkg://diff_benchmark.configs",
    config_name="main",
)
def main(cfg: DictConfig) -> None:
    """
    CLI entrypoint:
        diffbenchmark resample [hydra overrides]

    Resamples existing scalar.gii files from native space to template space.
    This is useful for data that was preprocessed before the automatic resampling feature.

    The resampled data overwrites the original files, so they will be in template space.
    """

    print("Resampling existing data to template space")
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]

    dataset_selected = DatasetConfig(
        **dataset_cfg,
        base_dir=Path(cluster_cfg.base_dir),
        results_dir=Path(cluster_cfg.results_dir),
    )

    pipeline = DefaultPipeline(dataset_selected)

    # Find all subjects that have scalar files
    scalar_files = sorted(
        pipeline.results_root.glob(
            f"derivatives/sub-*/dwi/*_hemi-L_param-{pipeline.metric}.scalar.gii"
        )
    )

    subject_ids = []
    for left_file in scalar_files:
        subject_id = left_file.stem.split("_")[0].replace("sub-", "")
        subject_ids.append(subject_id)

    print(f"Found {len(subject_ids)} subjects with scalar data")

    if "bids" not in pipeline.data_reading:
        print(
            "Dataset is not BIDS format - no resampling needed (data should already be in template space)"
        )
        return

    print(
        f"Resampling {len(subject_ids)} subjects from native space to {pipeline.surface_space} template space"
    )
    print("WARNING: This will overwrite the existing files!")

    # Get parallel config from hydra config if available
    parallel_type = cfg.cluster.conf.parallel_type  # Default to slurm
    n_jobs = cfg.get("n_jobs", 35)  # Default to 35 parallel jobs

    # Resample all subjects in parallel using run_jobs
    run_jobs(
        run_fn=resample_subject_wrapper,
        fn_kwargs_list=[
            {
                "subject_id": subject_id,
                "pipeline_cls": type(pipeline),
                "dataset_config": dataset_selected,
            }
            for subject_id in subject_ids
        ],
        parallel_type=parallel_type,
        slurm_cfg=cfg.cluster.slurm_cfg,
        n_jobs=n_jobs,
    )

    print(f"Resampling complete! Processed {len(subject_ids)} subjects")


if __name__ == "__main__":
    main()

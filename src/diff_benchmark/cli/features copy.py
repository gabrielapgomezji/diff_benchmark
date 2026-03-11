from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from diff_benchmark.preprocessing.brain_feature_extraction import MeshPipeline
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig


@hydra.main(
    version_base="1.3",
    config_path="pkg://diff_benchmark.configs",
    config_name="main",
)
def main(cfg: DictConfig) -> None:
    """
    CLI entrypoint:
        diffbenchmark features [hydra overrides]

    Computes microstructure features for configured datasets.
    """
    print("Running feature extraction")

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]

    dataset_selected = DatasetConfig(
        **dataset_cfg,
        base_dir=Path(cluster_cfg.base_dir),
        results_dir=Path(cluster_cfg.results_dir),
    )

    pipeline = MeshPipeline(dataset_selected, surface)
    pipeline.run_pipeline(
        recompute=True, cluster_conf=cfg.cluster.conf, slurm_cfg=cfg.cluster.slurm_cfg
    )


if __name__ == "__main__":
    main()

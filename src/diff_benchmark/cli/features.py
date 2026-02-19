import hydra
from omegaconf import DictConfig
from pathlib import Path
from omegaconf import OmegaConf

from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline
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
    # Optional: basic logging
    print("Running feature extraction")
    
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]

    dataset_selected = DatasetConfig(
        **dataset_cfg,
        base_dir=Path(cluster_cfg.base_dir),
        results_dir=Path(cluster_cfg.results_dir),
    )

    pipeline = DefaultPipeline(dataset_selected)
    # subject_id = "101915" #HCP #"29006" #abide #"CC721707" #CamCAN #"76884"  # wand # 
    # pipeline.compute_microstructure(subject_id)
    # breakpoint()
    pipeline.run_pipeline(recompute=True, cluster_conf=cfg.cluster.conf, slurm_cfg=cfg.cluster.slurm_cfg)


if __name__ == "__main__":
    main()

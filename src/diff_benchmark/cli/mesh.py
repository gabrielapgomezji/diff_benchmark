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
    """CLI entry point for surface mesh feature extraction.

    Computes microstructure maps for all subjects and projects them onto the
    cortical surface mesh.  Outputs are stored under
    ``<results_dir>/mesh/derivatives/sub-<id>/dwi/``.

    Usage::

        diffbenchmark-mesh
        diffbenchmark-mesh dataset=camcan
    """
    print("Running mesh feature extraction")

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]

    dataset_selected = DatasetConfig(
        **dataset_cfg,
        base_dir=Path(cluster_cfg.base_dir),
        results_dir=Path(cluster_cfg.results_dir),
    )

    surface_type = getattr(dataset_selected, "mesh_surface_type", "midthickness")
    pipeline = MeshPipeline(dataset_selected, surface_type=surface_type)
    force_recompute = bool(getattr(cfg.runtime, "force", False))
    pipeline.run_pipeline(
        cluster_conf=cfg.cluster.conf,
        slurm_cfg=cfg.cluster.slurm_cfg,
        recompute=force_recompute,
    )
    pipeline.run_analysis(mesh_cfg=cfg.mesh_pipeline)


if __name__ == "__main__":
    main()

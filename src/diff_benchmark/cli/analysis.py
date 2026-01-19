import hydra
from omegaconf import DictConfig
from pathlib import Path
from omegaconf import OmegaConf

from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.analysis.plot_debug import plot_debug_run
from diff_benchmark.analysis.plot_script import plot_run
from diff_benchmark.analysis.plot_summary import plot_metrics_summary
from pathlib import Path
import pandas as pd


def build_global_metrics(
    metrics_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Merge metrics_<run_id>.parquet files into a single metrics.parquet
    """
    if output_path.exists():
        return pd.read_parquet(output_path)

    dfs = []
    for p in metrics_dir.glob("metrics_*.parquet"):
        dfs.append(pd.read_parquet(p))

    if not dfs:
        raise RuntimeError(f"No metrics_*.parquet found in {metrics_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(output_path, index=False)
    return df

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
    results_dir = Path("./data/results")
    metrics_dir = results_dir / "parquet/analysis_results"
    metrics_path = metrics_dir / "metrics.parquet"
    debug_dir=Path("./data/results/parquet/debug")
    # breakpoint()
    # 1) Build / refresh global metrics
    df_metrics = build_global_metrics(
        metrics_dir=metrics_dir,
        output_path=metrics_path,
    )
    run_id = "2dcnn_39fc8501"
    # 2) Per-run plots
    if run_id:
        plot_debug_run(
            run_id=run_id,
            debug_dir=debug_dir,
            output_root=results_dir / "plots",
        )
        plot_run(
            run_id=run_id, #2dcnn_be425892'2dcnn_08ef30ab', '2dcnn_341c8a9b', 
            # '2dcnn_76059b89',
        #    '2dcnn_be425892', 'linear_2ddaa507', 'linear_3addbf07',
        #    'linear_cf6ab721'
            metrics_dir=metrics_dir / "metrics.parquet",
            predictions_path=results_dir / "parquet/data/predictions.parquet",
            targets_path=results_dir / "parquet/data/targets.parquet",
            output_root=results_dir / "plots",
        )

    # 3) Global summary
    plot_metrics_summary(
        metrics_path=metrics_path,
        output_dir=results_dir / "plots",
    )

if __name__ == "__main__":
    main()

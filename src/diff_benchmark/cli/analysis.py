import hydra
from omegaconf import DictConfig
from pathlib import Path
from omegaconf import OmegaConf

from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.analysis.plot_debug import plot_debug_run
from diff_benchmark.analysis.plot_script import plot_run
from diff_benchmark.analysis.plot_summary import plot_metrics_summary
from diff_benchmark.analysis.print_summary_table import is_successful_experiment, print_table, select_best_runs, table_best_means, table_detailed, table_folds_wide
from pathlib import Path
import pandas as pd


def build_global_metrics(experiments_root: Path, output_path: Path) -> pd.DataFrame:
    all_dfs = []

    for exp_dir in experiments_root.glob("exp_*"):
        metrics_file = exp_dir / "metrics" / "fold_metrics.parquet"
        config_file = exp_dir / "config.yaml"

        if not metrics_file.exists() or not config_file.exists():
            continue

        df = pd.read_parquet(metrics_file)
        cfg = OmegaConf.load(config_file)
        
        # ---- canonical experiment identity ----
        df["run_id"] = exp_dir.name.replace("exp_", "")
        df["dataset"] = cfg.dataset.name
        df["prediction_task"] = cfg.pred_head.prediction_task
        df["model_name"] = cfg.model.name

        # optional but often useful - default to "gray" if not specified
        df["tissue_type"] = cfg.dataset.get("tissue_type", "gray")
        df["primary_metric"] = cfg.dataset.metric_to_compute

        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No valid experiments found")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_parquet(output_path, index=False)
    return df_all



def build_summary_metrics(df_folds: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    df = (
        df_folds
        .groupby([
            "run_id",
            "model_name",
            "dataset",
            "prediction_task",
            "tissue_type",
            "primary_metric",
            "split",
            "metric",
        ])
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
        )
        .reset_index()
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
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
    results_dir = Path("./exp_outputs")
    experiments_root = results_dir / "experiments"
    plots_root = results_dir / "plots"
    summary_root = results_dir / "summary"

    metrics_folds_path = summary_root / "metrics_folds.parquet"
    df_folds = build_global_metrics(experiments_root, metrics_folds_path)
    summary_metrics_path = summary_root / "metrics_summary.parquet"
    df_summary = build_summary_metrics(df_folds, summary_metrics_path)
    
    # -----------------------------------------------------------------
    # 3) Print tables
    # -----------------------------------------------------------------

    # df_metrics = load_global_metrics(summary_metrics_path)
    print("\n=== BEST MEAN RESULTS ===")
    print("--- Primary Metric: accuracy ---")
    df_best = table_best_means(df_summary, primary_metric="accuracy")
    print_table(df_best)
    print("--- Primary Metric: rmse ---")
    df_best = table_best_means(df_summary, primary_metric="rmse")
    print_table(df_best)
    print("--- Primary Metric: mae ---")
    df_best = table_best_means(df_summary, primary_metric="mae")
    print_table(df_best)
    # try:
    #     print("--- Primary Metric: correlation ---")
    #     df_best = table_best_means(df_summary, primary_metric="rmse")
    #     print_table(df_best)
    # except Exception as e:
    #     print(f"Skipping RMSE table: {e}")

    best_runs = select_best_runs(df_summary, primary_metric="accuracy")
    print("\n=== DETAILED RESULTS (BEST RUN PER MODEL) ===")
    df_detailed = table_detailed(df_folds, best_runs, primary_metric="accuracy")
    print_table(df_detailed)

    # df_wide_train = table_folds_wide(df_folds, best_runs, split="train", primary_metric="accuracy")
    # print("\n=== WIDE-FORMAT RESULTS TRAIN ===")
    # print_table(df_wide_train)

    print("\n=== WIDE-FORMAT RESULTS TEST ===")
    df_wide_test = table_folds_wide(df_folds, best_runs, split="test", primary_metric="accuracy")
    print_table(df_wide_test)
    
    # for metric, df_m in df_summary.groupby("metric"):
    #     print(f"\n### Metric: {metric}")
    #     print_table(table_best_means(df_m, metric))
    
    # -----------------------------------------------------------------
    # Group by tissue type for tissue-specific analysis
    # -----------------------------------------------------------------
    print("\n" + "="*80)
    print("TISSUE-SPECIFIC ANALYSIS")
    print("="*80)
    
    for tissue_type, df_tissue in df_summary.groupby("tissue_type"):
        print(f"\n{'='*80}")
        print(f"TISSUE TYPE: {tissue_type.upper()}")
        print(f"{'='*80}")
        
        # Best results by dataset for this tissue type
        for ds, df_ds in df_tissue.groupby("dataset"):
            print(f"\n### Dataset: {ds} ({tissue_type})")
            print_table(table_best_means(df_ds))
        
        # Best results by primary metric for this tissue type
        for primary_metric, df_pm in df_tissue.groupby("primary_metric"):
            print(f"\n### Primary Metric: {primary_metric} ({tissue_type})")
            if "binary" in df_pm["prediction_task"].values or "classification" in df_pm["prediction_task"].values:
                print(f"--- Binary Classification ---")
                print_table(table_best_means(df_pm, primary_metric="accuracy"))
            if "regression" in df_pm["prediction_task"].values:
                print(f"--- Regression ---")
                print_table(table_best_means(df_pm, primary_metric="rmse"))
    
    # -----------------------------------------------------------------
    # Cross-tissue comparison
    # -----------------------------------------------------------------
    print("\n" + "="*80)
    print("CROSS-TISSUE COMPARISON")
    print("="*80)
    
    for ds, df_ds in df_summary.groupby("dataset"):
        print(f"\n### Dataset: {ds}")
        print_table(table_best_means(df_ds))

    for primary_metric, df_pm in df_summary.groupby("primary_metric"):
        print(f"\n### Primary Metric: {primary_metric} - Binary Classification")
        print_table(table_best_means(df_pm, primary_metric="accuracy"))
        print(f"\n### Primary Metric: {primary_metric} - Regression")
        print_table(table_best_means(df_pm, primary_metric="rmse"))

    # -----------------------------------------------------------------
    # 4) Generate plots per experiment
    # -----------------------------------------------------------------
    for exp_dir in experiments_root.glob("exp_*"):
        try:
            if not is_successful_experiment(exp_dir):
                print(f"Skipping {exp_dir.name}: not successful")
                continue
            run_id = exp_dir.name.replace("exp_", "")
            print(f"Processing plots for run: {run_id}")

            # Paths
            metrics_path = exp_dir / "metrics" / "fold_metrics.parquet"
            predictions_path = exp_dir / "predictions" / "predictions.parquet"
            targets_path = exp_dir / "predictions" / "targets.parquet"
            debug_dir = exp_dir / "debug"
            run_plots_dir = plots_root

            # Skip if plots already exist
            if run_plots_dir.exists() and any(run_plots_dir.glob(f"*{run_id}*.png")):
                print(f"Plots already exist for {run_id}, skipping...")
                continue

            # Debug plots if debug data exists
            if debug_dir.exists() and any(debug_dir.iterdir()):
                plot_debug_run(run_id=run_id, debug_dir=debug_dir, output_root=run_plots_dir)

            # Main experiment plots
            if metrics_path.exists() and predictions_path.exists() and targets_path.exists():
                plot_run(
                    run_id=run_id,
                    metrics_dir=metrics_path,
                    predictions_path=predictions_path,
                    targets_path=targets_path,
                    output_root=run_plots_dir,
                )
            else:
                print(f"Missing required files for {run_id}, skipping plots...")
        except Exception as e:
            print(f"Error processing {exp_dir.name}: {e}")
            continue

    print("\nAll experiments processed. Summaries and plots are saved under:")
    print(f"Summary metrics: {summary_metrics_path}")
    print(f"Plots root: {plots_root}")


if __name__ == "__main__":
    main()

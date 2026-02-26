from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from diff_benchmark.analysis.plot_debug import plot_debug_run
from diff_benchmark.analysis.plot_script import plot_run
from diff_benchmark.analysis.print_summary_table import (
    is_successful_experiment,
    print_table,
    select_best_runs,
    table_all_runs,
    table_best_means,
    table_folds_wide,
    table_model_aggregate,
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _infer_learning_curve_x_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "config.data.data_partition.train_size",
        "config.data.train_size",
        "config.data.partition.train_size",
        "config.runtime.learning_curve_train_size",
        "config.runtime.learning_curve_fraction",
        "config.runtime.learning_curve_sample_size",
        "config.runtime.learning_curve_step",
        "config.runtime.learning_curve_point",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_lc_metrics(prediction_task: str, df_curve: pd.DataFrame) -> tuple:
    """Return (test_mean_col, test_std_col, train_mean_col, train_std_col, y_label)
    for a learning-curve group based on the prediction task and available columns."""
    if "regression" in prediction_task:
        if "r2_test_mean" in df_curve.columns:
            return "r2_test_mean", "r2_test_std", "r2_train_mean", "r2_train_std", "Mean R2"
        if "pearson_correlation_test_mean" in df_curve.columns:
            return (
                "pearson_correlation_test_mean",
                "pearson_correlation_test_std",
                "pearson_correlation_train_mean",
                "pearson_correlation_train_std",
                "Mean Pearson Correlation",
            )
        return (
            "mae_weighted_test_mean",
            "mae_weighted_test_std",
            "mae_weighted_train_mean",
            "mae_weighted_train_std",
            "Mean MAE (weighted)",
        )
    return (
        "accuracy_weighted_test_mean",
        "accuracy_weighted_test_std",
        "accuracy_weighted_train_mean",
        "accuracy_weighted_train_std",
        "Mean Accuracy (weighted)",
    )


def _build_lc_filename(lc_id: str, df_curve: pd.DataFrame) -> str:
    """Construct an output filename for a learning-curve plot."""
    safe_lc_id = str(lc_id).replace("/", "_").replace(" ", "_")
    if "run_id" in df_curve.columns and not df_curve["run_id"].isna().all():
        run_id = str(df_curve["run_id"].iloc[0])
        parts = run_id.split("_")
        if len(parts) >= 3:
            base_run_id = "_".join(parts[:-1])
            return f"learning_curve_{base_run_id}_{safe_lc_id}.png"
    return f"learning_curve_{safe_lc_id}.png"


def _plot_single_learning_curve(
    lc_id: str,
    df_curve: pd.DataFrame,
    x_col: str,
    output_dir: Path,
) -> bool:
    """Plot and save one learning curve. Returns True if a file was written."""
    prediction_task = str(df_curve["prediction_task"].iloc[0]).lower()
    test_mean_col, test_std_col, train_mean_col, train_std_col, y_label = (
        _resolve_lc_metrics(prediction_task, df_curve)
    )

    if test_mean_col not in df_curve.columns:
        print(f"Skipping learning_curve_id={lc_id}: missing column `{test_mean_col}`")
        return False

    cols_to_keep = [x_col, test_mean_col]
    for c in [test_std_col, train_mean_col, train_std_col]:
        if c in df_curve.columns:
            cols_to_keep.append(c)

    df_plot = df_curve[cols_to_keep].copy()
    for c in cols_to_keep:
        if c != x_col:
            df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")

    df_plot = df_plot[df_plot[test_mean_col].notna()]
    if df_plot.empty:
        print(f"Skipping learning_curve_id={lc_id}: no valid `{test_mean_col}` values")
        return False

    agg_dict = {test_mean_col: "mean"}
    for c in [test_std_col, train_mean_col, train_std_col]:
        if c in df_plot.columns:
            agg_dict[c] = "mean"

    df_mean = df_plot.groupby(x_col, as_index=False).agg(agg_dict).sort_values(x_col)
    if df_mean.empty:
        return False

    model_name = str(df_curve["model_name"].iloc[0])
    dataset = str(df_curve["dataset"].iloc[0])
    tissue_type = str(df_curve["tissue_type"].iloc[0])
    metric_to_compute = str(df_curve["primary_metric"].iloc[0])
    target = str(df_curve["target"].iloc[0])
    title = f"{model_name} {dataset} - {tissue_type} - {metric_to_compute} - {target}"

    plt.figure(figsize=(10, 6))

    plt.plot(df_mean[x_col], df_mean[test_mean_col], marker="o", linewidth=2, label="Test", color="red")
    if test_std_col in df_mean.columns:
        plt.fill_between(
            df_mean[x_col],
            df_mean[test_mean_col] - df_mean[test_std_col],
            df_mean[test_mean_col] + df_mean[test_std_col],
            color="red",
            alpha=0.2,
        )

    if train_mean_col in df_mean.columns:
        plt.plot(df_mean[x_col], df_mean[train_mean_col], marker="s", linestyle="--", linewidth=2, label="Train", color="blue")
        if train_std_col in df_mean.columns:
            plt.fill_between(
                df_mean[x_col],
                df_mean[train_mean_col] - df_mean[train_std_col],
                df_mean[train_mean_col] + df_mean[train_std_col],
                color="blue",
                alpha=0.2,
            )

    xlabel = x_col.replace("config.data.", "data.").replace("config.runtime.", "runtime.")
    plt.xlabel(xlabel)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = output_dir / _build_lc_filename(lc_id, df_curve)
    plt.savefig(out_file, dpi=160)
    plt.close()
    return True


def plot_learning_curves_from_comprehensive_table(
    comprehensive_table_path: Path,
    output_root: Path,
) -> None:
    """
    Plot learning curves from the comprehensive results parquet.

    Uses only rows where config.runtime.learning_curve=True and groups curves by
    config.runtime.learning_curve_id. For each curve point (x-axis inferred from runtime
    learning-curve configuration), it plots the mean performance:
      - classification: accuracy_weighted_test_mean
      - regression: mae_weighted_test_mean
    """
    if not comprehensive_table_path.exists():
        print(
            f"Learning-curve plotting skipped: file not found: {comprehensive_table_path}"
        )
        return

    df = pd.read_parquet(comprehensive_table_path)
    required_cols = [
        "config.runtime.learning_curve_id",
        "prediction_task",
        "model_name",
        "dataset",
        "tissue_type",
        "primary_metric",
        "target",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Learning-curve plotting skipped: missing columns: {missing_cols}")
        return

    lc_flag_col = None
    if "config.runtime.learning_curve" in df.columns:
        lc_flag_col = "config.runtime.learning_curve"
    elif "config.runtime.learning_curve_exp" in df.columns:
        lc_flag_col = "config.runtime.learning_curve_exp"

    if lc_flag_col is None:
        print("Learning-curve plotting skipped: missing learning-curve flag column.")
        return

    lc_mask = df[lc_flag_col].apply(_as_bool)
    df_lc = df[lc_mask].copy()

    if df_lc.empty:
        print(f"No learning-curve experiments found ({lc_flag_col}=True).")
        return

    x_col = _infer_learning_curve_x_column(df_lc)
    if x_col is None:
        print(
            "Learning-curve plotting skipped: no x-axis column found. "
            "Expected e.g. config.data.data_partition.train_size."
        )
        return

    df_lc[x_col] = pd.to_numeric(df_lc[x_col], errors="coerce")
    df_lc = df_lc[df_lc[x_col].notna()]
    if df_lc.empty:
        print("Learning-curve plotting skipped: x-axis values are not numeric.")
        return

    output_dir = output_root / "_learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["config.runtime.learning_curve_id"]
    n_plots = 0

    for lc_id, df_curve in df_lc.groupby(group_cols, dropna=False):
        if isinstance(lc_id, tuple):
            lc_id = lc_id[0]
        if _plot_single_learning_curve(lc_id, df_curve, x_col, output_dir):
            n_plots += 1

    print(f"✓ Learning curves saved to: {output_dir} ({n_plots} plots)")


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
        df_folds.groupby(
            [
                "run_id",
                "model_name",
                "dataset",
                "prediction_task",
                "tissue_type",
                "primary_metric",
                "split",
                "metric",
            ]
        )
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
        )
        .reset_index()
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def flatten_config(cfg: DictConfig, prefix: str = "") -> dict:
    """
    Recursively flatten a nested OmegaConf config into a flat dictionary.
    Converts all OmegaConf types to native Python types for parquet compatibility.

    Args:
        cfg: OmegaConf config object
        prefix: Prefix for nested keys

    Returns:
        Flat dictionary with dot-separated keys and native Python values
    """
    flat = {}

    for key, value in cfg.items():
        full_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, DictConfig):
            # Recursively flatten nested configs
            flat.update(flatten_config(value, prefix=f"{full_key}."))
        else:
            # Convert to native Python type using OmegaConf.to_container
            # This handles ListConfig, DictConfig, and other OmegaConf types
            try:
                native_value = OmegaConf.to_container(value, resolve=True)
                # Convert complex types to string for parquet compatibility
                if isinstance(native_value, (list, tuple, dict)):
                    flat[full_key] = str(native_value)
                else:
                    flat[full_key] = native_value
            except Exception:
                # Fallback to string representation if conversion fails
                flat[full_key] = str(value)

    return flat


def build_comprehensive_table(
    experiments_root: Path, output_path: Path
) -> pd.DataFrame:
    """
    Build a comprehensive table with all experiment information including:
    - Basic identifiers (run_id, model_name, dataset, tissue_type, etc.)
    - All metrics from all folds and splits as columns
    - Mean and std of metrics across folds for each split
    - All model hyperparameters from config

    Args:
        experiments_root: Path to experiments directory
        output_path: Path where to save the output parquet file

    Returns:
        Comprehensive DataFrame with all experiment information
    """
    all_experiments = []

    for exp_dir in experiments_root.glob("exp_*"):
        metrics_file = exp_dir / "metrics" / "fold_metrics.parquet"
        config_file = exp_dir / "config.yaml"

        if not config_file.exists():
            continue

        # Load config
        cfg = OmegaConf.load(config_file)

        # Base experiment info
        exp_info = {
            "run_id": exp_dir.name.replace("exp_", ""),
            "model_name": cfg.model.name,
            "dataset": cfg.dataset.name,
            "tissue_type": cfg.dataset.get("tissue_type", "gray"),
            "primary_metric": cfg.dataset.metric_to_compute,
            "target": str(cfg.target.target_column),
            "prediction_task": cfg.pred_head.prediction_task,
        }

        # Load metrics if available
        if metrics_file.exists():
            df_metrics = pd.read_parquet(metrics_file)

            # Create columns for each metric-fold-split combination
            for _, row in df_metrics.iterrows():
                metric = row["metric"]
                fold = row.get("fold", 0)
                split = row["split"]
                value = row["value"]

                col_name = f"{metric}_{split}_fold{fold}"
                exp_info[col_name] = value

            # Compute mean and std for each metric-split combination
            for split in df_metrics["split"].unique():
                df_split = df_metrics[df_metrics["split"] == split]

                for metric in df_split["metric"].unique():
                    df_metric = df_split[df_split["metric"] == metric]

                    values = df_metric["value"].values
                    exp_info[f"{metric}_{split}_mean"] = np.mean(values)
                    exp_info[f"{metric}_{split}_std"] = np.std(values)

        # Flatten and add all config parameters
        # Focus on model, backend, pred_head, data, target and runtime sections
        sections_to_include = [
            "model",
            "backend",
            "pred_head",
            "data",
            "target",
            "runtime",
        ]

        for section in sections_to_include:
            if section in cfg:
                section_cfg = cfg[section]
                flat_params = flatten_config(section_cfg, prefix=f"config.{section}.")

                # Add all parameters, will be NaN for models that don't have them
                for param_key, param_value in flat_params.items():
                    exp_info[param_key] = param_value

        # Ensure learning-curve runtime fields always exist for downstream analysis
        # (older experiments/configs may not define runtime or learning-curve params)
        exp_info.setdefault("config.runtime.learning_curve_exp", None)
        exp_info.setdefault("config.runtime.learning_curve_id", None)

        all_experiments.append(exp_info)

    if not all_experiments:
        raise RuntimeError("No valid experiments found")

    # Create DataFrame - pandas will automatically fill missing columns with NaN
    df_comprehensive = pd.DataFrame(all_experiments)

    # Sort columns for better readability
    # 1. Identifiers
    id_cols = [
        "run_id",
        "model_name",
        "dataset",
        "tissue_type",
        "primary_metric",
        "target",
        "prediction_task",
    ]

    # 2. Metric columns (mean/std first, then individual folds)
    metric_cols = [
        col
        for col in df_comprehensive.columns
        if col not in id_cols and not col.startswith("config.")
    ]
    mean_std_cols = [col for col in metric_cols if "_mean" in col or "_std" in col]
    fold_cols = [col for col in metric_cols if col not in mean_std_cols]

    # Sort mean/std columns
    mean_std_cols.sort()
    # Sort fold columns
    fold_cols.sort()

    # 3. Config columns
    config_cols = [col for col in df_comprehensive.columns if col.startswith("config.")]
    config_cols.sort()

    # Reorder columns
    ordered_cols = id_cols + mean_std_cols + fold_cols + config_cols

    # Keep only columns that exist
    ordered_cols = [col for col in ordered_cols if col in df_comprehensive.columns]

    df_comprehensive = df_comprehensive[ordered_cols]

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_comprehensive.to_parquet(output_path, index=False)

    return df_comprehensive


def _find_best_run(
    df_group_orig: pd.DataFrame,
    prediction_task: str,
) -> tuple:
    """Return (metric_col, metric_name, lower_is_better, best_run, metrics_priority).

    Falls back to the first row when no usable metric column is found.
    """
    if "classification" in prediction_task or "binary" in prediction_task:
        metrics_priority = ["accuracy_weighted", "accuracy", "roc_auc", "f1_weighted"]
    elif "regression" in prediction_task:
        metrics_priority = ["r2"]
    else:
        metrics_priority = ["accuracy_weighted", "rmse_weighted", "accuracy", "rmse"]

    metric_col = None
    metric_name = None

    candidates = []
    for m in metrics_priority:
        for split in ["test", "val"]:
            col = f"{m}_{split}_mean"
            if col in df_group_orig.columns:
                valid_count = df_group_orig[col].count()
                if valid_count > 0:
                    candidates.append(
                        {
                            "col": col,
                            "metric": m,
                            "split": split,
                            "count": valid_count,
                            "priority_idx": metrics_priority.index(m),
                            "split_score": 1 if split == "test" else 0,
                        }
                    )

    if candidates:
        candidates.sort(key=lambda x: (-x["count"], x["priority_idx"], -x["split_score"]))
        metric_col = candidates[0]["col"]
        metric_name = candidates[0]["metric"]

    lower_is_better = bool(metric_name and any(x in metric_name.lower() for x in ["r2"]))

    if metric_col is None:
        return "run_id", "N/A", False, df_group_orig.iloc[0], metrics_priority

    if lower_is_better:
        best_idx = df_group_orig[metric_col].idxmin()
    else:
        best_idx = df_group_orig[metric_col].idxmax()

    if pd.isna(best_idx):
        return "run_id", "N/A", False, df_group_orig.iloc[0], metrics_priority

    return metric_col, metric_name, lower_is_better, df_group_orig.loc[best_idx], metrics_priority


def _build_report_row(
    row: pd.Series,
    best_run_id: str,
    metric_col: str,
    metric_name: str,
    metrics_priority: list,
    best_primary_metric,
    variable_config_cols: list,
    best_config_vals: dict,
) -> list:
    """Build a single data row for the per-group report table."""
    row_data = []

    rid = str(row["run_id"])
    if row["run_id"] == best_run_id:
        rid += " *"
    row_data.append(rid)

    if metric_col != "run_id":
        val = row[metric_col]
        current_metric_name = metric_name
        if pd.isna(val):
            found_fallback = False
            for m in metrics_priority:
                for split in ["test", "val"]:
                    col_candidate = f"{m}_{split}_mean"
                    if col_candidate in row and pd.notna(row[col_candidate]):
                        val = row[col_candidate]
                        current_metric_name = m
                        found_fallback = True
                        break
                if found_fallback:
                    break
        if pd.notna(val):
            score = f"{val:.4f}"
            if current_metric_name != metric_name:
                score += f" ({current_metric_name})"
        else:
            score = "nan"
        row_data.append(score)
    else:
        row_data.append("-")

    p_metric = str(row["primary_metric"])
    if p_metric != str(best_primary_metric):
        p_metric = f"|{p_metric}|"
    row_data.append(p_metric)

    for c in variable_config_cols:
        val = row[c]
        val_str = str(val) if pd.notna(val) else "-"
        v_str = str(val) if pd.notna(val) else "nan"
        b_str = str(best_config_vals[c]) if pd.notna(best_config_vals[c]) else "nan"
        if v_str != b_str:
            val_str = f"|{val_str}|"
        if len(val_str) > 30:
            val_str = val_str[:27] + "..."
        row_data.append(val_str)

    return row_data


def _render_report_group(
    df_group_orig: pd.DataFrame,
    config_cols: list,
    group_desc: str,
    prediction_task: str,
) -> list:
    """Return report lines for a single (model, tissue, task, target) group."""
    report_lines = [f"\nGROUP: {group_desc}", "-" * 140]

    if len(df_group_orig) == 0:
        return report_lines

    metric_col, metric_name, lower_is_better, best_run, metrics_priority = _find_best_run(
        df_group_orig, prediction_task
    )

    best_run_id = best_run["run_id"]

    if metric_col == "run_id":
        report_lines.append(
            f"  Warning: Could not find performance metric column. Task: {prediction_task}"
        )
    else:
        val_disp = f"{best_run[metric_col]:.4f}"
        report_lines.append(
            f"  Best Run ID: {best_run_id} (Metric: {metric_col}, Score: {val_disp})"
        )
    report_lines.append(
        "  Highlighting: Parameters different from Best Run are enclosed in |...|"
    )

    variable_config_cols = sorted(
        c
        for c in config_cols
        if c in df_group_orig.columns and len(df_group_orig[c].astype(str).unique()) > 1
    )

    short_col_map = {
        c: c.replace("config.", "").replace("model.", "").replace("optimizer.", "opt.").replace("backend.", "bk.")
        for c in variable_config_cols
    }
    headers = ["RunID", "Score", "PrimaryMetric"] + [short_col_map[c] for c in variable_config_cols]

    df_sorted = df_group_orig.copy()
    df_sorted["is_best"] = df_sorted["run_id"] == best_run_id
    if metric_col != "run_id":
        df_sorted = df_sorted.sort_values(
            ["is_best", metric_col], ascending=[False, lower_is_better]
        )

    best_config_vals = {c: best_run[c] for c in variable_config_cols}
    best_primary_metric = best_run["primary_metric"]

    table_data = [
        _build_report_row(
            row, best_run_id, metric_col, metric_name, metrics_priority,
            best_primary_metric, variable_config_cols, best_config_vals,
        )
        for _, row in df_sorted.iterrows()
    ]

    col_widths = [min(max(len(h), max((len(r[i]) for r in table_data), default=0)) + 2, 50) for i, h in enumerate(headers)]
    fmt = "".join([f"{{:<{w}}}" for w in col_widths])

    try:
        report_lines.append(fmt.format(*headers))
        report_lines.append("-" * sum(col_widths))
    except Exception as e:
        report_lines.append(f"Error formatting table: {e}")

    for row in table_data:
        try:
            report_lines.append(fmt.format(*row))
        except Exception:
            pass

    report_lines.append("\n" + "=" * 40 + "\n")
    return report_lines


def generate_dataset_reports(df_comprehensive: pd.DataFrame, output_dir: Path) -> None:
    """
    Generates text reports per dataset comparing experiments.
    For each (model, tissue, task, target) group:
      - Identifies the best run (based on primary metric on test set)
      - Lists all runs
      - Highlights hyperparameters that differ from the best run
    """
    print(f"\nGenerating dataset reports in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_filtered = df_comprehensive[
        ~df_comprehensive["model_name"].astype(str).str.contains("dummy", case=False, na=False)
    ]

    config_cols = [c for c in df_filtered.columns if c.startswith("config.")]

    for dataset_name, df_dataset in df_filtered.groupby("dataset"):
        report_lines = [f"DATASET REPORT: {dataset_name}", "=" * 140]

        group_keys = ["model_name", "tissue_type", "prediction_task", "target"]
        active_keys = [k for k in group_keys if k in df_dataset.columns]

        df_dataset_safe = df_dataset.copy()
        for k in active_keys:
            df_dataset_safe[k] = df_dataset_safe[k].fillna("NaN")

        for group_values, df_group in df_dataset_safe.groupby(active_keys):
            df_group_orig = df_dataset.loc[df_group.index]
            group_desc = ", ".join(f"{k}={v}" for k, v in zip(active_keys, group_values))
            prediction_task = str(df_group["prediction_task"].iloc[0]).lower()
            report_lines.extend(
                _render_report_group(df_group_orig, config_cols, group_desc, prediction_task)
            )

        out_file = output_dir / f"{dataset_name}_report.txt"
        try:
            with open(out_file, "w") as f:
                f.write("\n".join(report_lines))
            print(f"✓ Report saved to: {out_file}")
        except Exception as e:
            print(f"Failed to write report for {dataset_name}: {e}")


def build_coverage_table(df_comprehensive: pd.DataFrame, output_dir: Path) -> None:
    """
    Build a coverage table showing which dataset/microstructure combinations
    have been tested with which models and tissue/target configurations.

    The table format is:
    - Rows: dataset + microstructure (e.g., "hcp_sh", "camcan_md")
    - Columns: models (DUMMY, PCA_L, LINEAR, etc.)
    - Cells: tissue_target codes (e.g., "gg", "ga", "wg", "wa")
      where first letter = tissue type (g=gray, w=white)
      and second letter = target (g=gender, a=age, etc.)

    Args:
        df_comprehensive: The comprehensive results DataFrame
        output_dir: Directory where to save the coverage table
    """
    # Model name mapping to abbreviations
    model_abbrev = {
        "dummy_classifier": "DUMMY",
        "dummy_regressor": "DUMMY",
        "pca_linear": "PCA_L",
        "linear": "LIN",
        "pca_forest": "PCA_F",
        "forest": "FOR",
        "pca_svm": "PCA_S",
        "svm": "SVM",
        "medicalnet": "MNET",
        "dinov2": "DINO",
        "vit": "VIT",
        "curia": "CURIA",
    }

    # Target name mapping to abbreviations
    target_abbrev = {
        "age": "a",
        "gender": "g",
        "sex": "g",  # Sometimes gender is called sex
        "diagnosis": "d",
        "dxgroup": "d",
    }

    # Create coverage dictionary
    # Structure: {(dataset, microstructure): {model: [tissue_target_codes]}}
    coverage = {}

    for _, row in df_comprehensive.iterrows():
        dataset = row["dataset"]
        microstructure = row["primary_metric"]  # sh, md, rtop, mk, etc.
        model = row["model_name"]
        tissue = (
            row["tissue_type"][0] if pd.notna(row["tissue_type"]) else "u"
        )  # g, w, or u=unknown

        # Extract target abbreviation
        target_str = row["target"].lower()
        target_code = "u"  # unknown
        for target_name, abbrev in target_abbrev.items():
            if target_name in target_str:
                target_code = abbrev
                break

        # Create the tissue_target code (e.g., "gg", "wa")
        tissue_target = f"{tissue}{target_code}"

        # Get model abbreviation
        model_code = model_abbrev.get(model, model.upper()[:6])

        # Add to coverage
        key = (dataset, microstructure)
        if key not in coverage:
            coverage[key] = {}
        if model_code not in coverage[key]:
            coverage[key][model_code] = set()
        coverage[key][model_code].add(tissue_target)

    # Get all unique models and sort them
    all_models = sorted(
        set(
            model_code
            for model_dict in coverage.values()
            for model_code in model_dict.keys()
        )
    )

    # Get all unique dataset_microstructure combinations and sort them
    all_combos = sorted(coverage.keys())

    # Build the table as a list of strings
    table_lines = []

    # Standard view: Dataset-microstructure as rows, models as columns
    header = f"{'Dataset_Microstructure':<23}"
    for model in all_models:
        header += f" | {model:^11}"
    table_lines.append(header)
    table_lines.append("=" * min(163, 23 + len(all_models) * 14))

    # Data rows
    for dataset, microstructure in all_combos:
        row_label = f"{dataset}_{microstructure}"
        row = f"{row_label:<23}"

        for model in all_models:
            if model in coverage[(dataset, microstructure)]:
                # Get all tissue_target codes for this combination
                codes = coverage[(dataset, microstructure)][model]
                # Join multiple codes with comma (e.g., "gg,ga,wg")
                cell_value = ",".join(sorted(codes))
            else:
                cell_value = "-"

            row += f" | {cell_value:^11}"

        table_lines.append(row)

    # Add summary statistics
    table_lines.append("=" * min(163, 23 + len(all_models) * 14))

    # Add transposed view for better readability
    table_lines.append("")
    table_lines.append("=" * 80)
    table_lines.append("TRANSPOSED VIEW (for easier reading)")
    table_lines.append("=" * 80)
    table_lines.append("")

    # For each dataset-microstructure combination
    for dataset, microstructure in all_combos:
        table_lines.append(f"\n{dataset}_{microstructure}:")
        table_lines.append("-" * 40)

        for model in all_models:
            if model in coverage[(dataset, microstructure)]:
                codes = ",".join(sorted(coverage[(dataset, microstructure)][model]))
                table_lines.append(f"  {model:8s} : {codes}")
            else:
                table_lines.append(f"  {model:8s} : -")

    table_lines.append("\n" + "=" * 80)
    table_lines.append("")
    table_lines.append("LEGEND:")
    table_lines.append("  First letter = tissue type: g=gray, w=white")
    table_lines.append("  Second letter = target: g=gender, a=age, d=diagnosis")
    table_lines.append("  Example: 'gg' = gray matter, gender prediction")
    table_lines.append("           'wa' = white matter, age prediction")
    table_lines.append("  '-' = no experiments for this combination")
    table_lines.append("")

    # Count statistics
    table_lines.append("STATISTICS:")
    table_lines.append(
        f"  Total dataset-microstructure combinations: {len(all_combos)}"
    )
    table_lines.append(f"  Total models: {len(all_models)}")
    table_lines.append(f"  Total experiments: {len(df_comprehensive)}")

    # Model coverage
    table_lines.append("")
    table_lines.append("MODEL COVERAGE:")
    for model in all_models:
        count = sum(1 for combo_dict in coverage.values() if model in combo_dict)
        percentage = (count / len(all_combos)) * 100 if all_combos else 0
        table_lines.append(
            f"  {model:>10}: {count:3d}/{len(all_combos):3d} combinations ({percentage:5.1f}%)"
        )

    # Dataset coverage
    table_lines.append("")
    table_lines.append("DATASET COVERAGE:")
    datasets = sorted(set(dataset for dataset, _ in all_combos))
    for dataset in datasets:
        count = sum(1 for d, _ in all_combos if d == dataset)
        table_lines.append(f"  {dataset:>10}: {count:2d} microstructures")

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "coverage_table.txt"

    with open(output_file, "w") as f:
        f.write("\n".join(table_lines))

    print(f"\n✓ Coverage table saved to: {output_file}")

    # Also print to console
    print("\n" + "=" * 80)
    print("EXPERIMENT COVERAGE TABLE")
    print("=" * 80)
    for line in table_lines[: min(50, len(table_lines))]:  # Print first 50 lines
        print(line)
    if len(table_lines) > 50:
        print(f"\n... ({len(table_lines) - 50} more lines in file)")


def process_experiment_plots(
    exp_dir: Path, plots_root: Path, force_plots: bool, debug_mode: bool
) -> None:
    try:
        is_successful = is_successful_experiment(exp_dir)
        has_debug_info = (exp_dir / "debug").exists() and any(
            (exp_dir / "debug").iterdir()
        )

        # If we are in debug mode, we want to plot debug info for running experiments too
        # Otherwise, we only look at successful experiments
        if not is_successful and not (debug_mode and has_debug_info):
            # Skip if failed/running AND we don't want to debug running experiments
            # OR if valid but no debug info and not successful
            # Actually logic:
            # If successful -> process normally
            # If not successful -> skip unless debug=true and has_debug_info
            return

        run_id = exp_dir.name.replace("exp_", "")
        print(f"Processing plots for run: {run_id}")

        # Paths
        metrics_path = exp_dir / "metrics" / "fold_metrics.parquet"
        predictions_path = exp_dir / "predictions" / "predictions.parquet"
        targets_path = exp_dir / "predictions" / "targets.parquet"
        debug_dir = exp_dir / "debug"
        run_plots_dir = plots_root / run_id

        # Check if main plots already exist
        main_plots_exist = False
        if not force_plots and run_plots_dir.exists():
            main_plot_patterns = [
                "confusion_*.png",
                "roc_curve.png",
                "regression_*.png",
                "metrics_summary.png",
            ]
            main_plots_exist = any(
                run_plots_dir.glob(pattern) for pattern in main_plot_patterns
            )

        # Check if debug plots already exist
        debug_plots_exist = False
        debug_plots_dir = run_plots_dir / "debug"
        if not force_plots and debug_plots_dir.exists():
            debug_plots_exist = any(debug_plots_dir.glob("debug_training_*.png"))

        # Debug plots if debug data exists and plots don't exist yet
        # MODIFIED: Allow plotting even if experiment isn't fully successful if debug info is there
        if debug_dir.exists() and any(debug_dir.iterdir()):
            if not debug_plots_exist or force_plots:
                print(f"  Creating debug plots for {run_id}...")
                plot_debug_run(
                    run_id=run_id, debug_dir=debug_dir, output_root=plots_root
                )
            else:
                print(f"  Debug plots already exist for {run_id}, skipping...")

        # Main experiment plots if not already computed (Requires success usually implies metrics exist)
        # Only try to plot main results if metrics exist (which usually implies success or at least partial success)
        if not main_plots_exist:
            if (
                metrics_path.exists()
                and predictions_path.exists()
                and targets_path.exists()
            ):
                print(f"  Creating main plots for {run_id}...")
                plot_run(
                    run_id=run_id,
                    metrics_dir=metrics_path,
                    predictions_path=predictions_path,
                    targets_path=targets_path,
                    output_root=plots_root,
                )
            else:
                if is_successful:
                    print(
                        f"  Missing required files for main plots ({run_id}), despite success flag. Skipping..."
                    )
                else:
                    print(f"  Skipping main plots for {run_id} (incomplete run).")
        else:
            print(f"  Main plots already exist for {run_id}, skipping...")
    except Exception as e:
        print(f"Error processing {exp_dir.name}: {e}")
        import traceback

        traceback.print_exc()


@hydra.main(
    version_base="1.3",
    config_path="pkg://diff_benchmark.configs",
    config_name="main",
)
def main(cfg: DictConfig) -> None:
    """
    CLI entrypoint:
        diffbenchmark-analysis [tables=true/false] [plots=true/false] [debug=true/false]

    Analyzes experiment results, generates summary tables and plots.

    Options:
        tables=false: Skip printing summary tables
        plots=false: Skip generating plots
        force_plots=true: Force recomputing plots even if they exist (default: false)
        analysis.debug=true: Generate debug plots even for running/incomplete experiments (default: false)
        (no options): Do both tables and plots (default)

    Examples:
        poetry run diffbenchmark-analysis                    # Both tables and plots
        poetry run diffbenchmark-analysis plots=false        # Only tables
        poetry run diffbenchmark-analysis force_plots=true   # Force plots
        poetry run diffbenchmark-analysis tables=false       # Only plots
        poetry run diffbenchmark-analysis analysis.debug=true # plots also for partial runs
    """
    # Get flags from config (with defaults)
    show_tables = cfg.analysis.tables
    show_plots = cfg.analysis.plots
    force_plots = cfg.analysis.get("force_plots", False)
    debug_mode = cfg.analysis.get("debug", False)

    results_dir = Path("./exp_outputs")
    experiments_root = results_dir / "experiments"
    plots_root = results_dir / "plots"
    summary_root = results_dir / "summary"

    comprehensive_table_path = summary_root / "comprehensive_results.parquet"
    metrics_folds_path = summary_root / "metrics_folds.parquet"
    summary_metrics_path = summary_root / "metrics_summary.parquet"

    # -----------------------------------------------------------------
    # 3) Print tables and reports
    # -----------------------------------------------------------------
    if show_tables:
        # Build comprehensive table with all experiment information
        print(f"\nBuilding comprehensive results table...")
        df_comprehensive = build_comprehensive_table(
            experiments_root, comprehensive_table_path
        )
        print(f"✓ Comprehensive table saved to: {comprehensive_table_path}")
        print(
            f"  Shape: {df_comprehensive.shape[0]} experiments × {df_comprehensive.shape[1]} columns"
        )

        # Build coverage table showing which experiments have been run
        tables_dir = summary_root / "tables"
        build_coverage_table(df_comprehensive, tables_dir)

        # Generate detailed reports per dataset
        reports_dir = summary_root / "reports"
        generate_dataset_reports(df_comprehensive, reports_dir)

        df_folds = build_global_metrics(experiments_root, metrics_folds_path)
        df_summary = build_summary_metrics(df_folds, summary_metrics_path)

        # -----------------------------------------------------------------
        # MODEL AGGREGATE STATISTICS (mean across all runs per model)
        # -----------------------------------------------------------------
        print("\n" + "=" * 80)
        print("MODEL AGGREGATE STATISTICS (Across All Runs)")
        print("=" * 80)
        print("Shows mean/std/min/max for each model across all datasets and runs")

        print("\n--- Classification: Accuracy ---")
        df_model_acc = table_model_aggregate(
            df_summary, primary_metric="accuracy_weighted"
        )
        print_table(df_model_acc)

        print("\n--- Regression: MAE ---")
        df_model_mae = table_model_aggregate(df_summary, primary_metric="mae_weighted")
        print_table(df_model_mae)

        print("\n--- Regression: RMSE ---")
        df_model_rmse = table_model_aggregate(
            df_summary, primary_metric="rmse_weighted"
        )
        print_table(df_model_rmse)

        print("\n--- Regression: Pearson Correlation ---")
        df_model_corr = table_model_aggregate(
            df_summary, primary_metric="pearson_correlation"
        )
        print_table(df_model_corr)

        # -----------------------------------------------------------------
        # ALL RUNS RESULTS (shows every run, not just best)
        # -----------------------------------------------------------------
        print("\n" + "=" * 80)
        print("ALL RUNS RESULTS (Every Model Run)")
        print("=" * 80)
        print("\n--- Classification: Accuracy ---")
        df_all_acc = table_all_runs(df_summary, primary_metric="accuracy_weighted")
        print_table(df_all_acc)

        print("\n--- Regression: MAE ---")
        df_all_mae = table_all_runs(df_summary, primary_metric="mae_weighted")
        print_table(df_all_mae)

        print("\n--- Regression: RMSE ---")
        df_all_rmse = table_all_runs(df_summary, primary_metric="rmse_weighted")
        print_table(df_all_rmse)

        print("\n--- Regression: Pearson Correlation ---")
        df_all_corr = table_all_runs(df_summary, primary_metric="pearson_correlation")
        print_table(df_all_corr)

        # -----------------------------------------------------------------
        # BEST MEAN RESULTS (best run per model/dataset combo)
        # -----------------------------------------------------------------
        print("\n" + "=" * 80)
        print("BEST MEAN RESULTS (Best Run per Model/Dataset)")
        print("=" * 80)
        print("\n--- Primary Metric: accuracy ---")
        df_best = table_best_means(df_summary, primary_metric="accuracy_weighted")
        print_table(df_best)
        print("\n--- Primary Metric: rmse ---")
        df_best = table_best_means(df_summary, primary_metric="rmse_weighted")
        print_table(df_best)
        print("\n--- Primary Metric: mae ---")
        df_best = table_best_means(df_summary, primary_metric="mae_weighted")
        print_table(df_best)
        print("\n--- Primary Metric: pearson_correlation ---")
        df_best = table_best_means(df_summary, primary_metric="pearson_correlation")
        print_table(df_best)

        best_runs = select_best_runs(df_summary, primary_metric="accuracy_weighted")

        print("\n=== WIDE-FORMAT RESULTS TEST ===")
        df_wide_test = table_folds_wide(
            df_folds, best_runs, split="test", primary_metric="accuracy_weighted"
        )
        print_table(df_wide_test)

        # -----------------------------------------------------------------
        # Group by tissue type for tissue-specific analysis
        # -----------------------------------------------------------------
        print("\n" + "=" * 80)
        print("TISSUE-SPECIFIC ANALYSIS")
        print("=" * 80)

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
                if (
                    "binary" in df_pm["prediction_task"].values
                    or "classification" in df_pm["prediction_task"].values
                ):
                    print(f"--- Binary Classification ---")
                    print_table(
                        table_best_means(df_pm, primary_metric="accuracy_weighted")
                    )
                if "regression" in df_pm["prediction_task"].values:
                    print(f"--- Regression ---")
                    print_table(table_best_means(df_pm, primary_metric="rmse_weighted"))

        # -----------------------------------------------------------------
        # Cross-tissue comparison
        # -----------------------------------------------------------------
        print("\n" + "=" * 80)
        print("CROSS-TISSUE COMPARISON")
        print("=" * 80)

        for ds, df_ds in df_summary.groupby("dataset"):
            print(f"\n### Dataset: {ds}")
            print_table(table_best_means(df_ds, primary_metric="accuracy_weighted"))

        for primary_metric, df_pm in df_summary.groupby("primary_metric"):
            print(f"\n### Primary Metric: {primary_metric} - Binary Classification")
            print_table(table_best_means(df_pm, primary_metric="accuracy_weighted"))
            print(f"\n### Primary Metric: {primary_metric} - Regression")
            print_table(table_best_means(df_pm, primary_metric="rmse_weighted"))

    # -----------------------------------------------------------------
    # 4) Generate plots per experiment
    # -----------------------------------------------------------------
    if show_plots:
        # Build learning curve plots from the comprehensive results table
        plot_learning_curves_from_comprehensive_table(
            comprehensive_table_path, plots_root
        )
        for exp_dir in experiments_root.glob("exp_*"):
            process_experiment_plots(exp_dir, plots_root, force_plots, debug_mode)

        print("\nPlots generation complete!")
        print(f"Plots saved to: {plots_root}")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    if show_tables and show_plots:
        print("Analysis complete! Tables printed and plots generated.")
    elif show_tables:
        print("Analysis complete! Tables printed.")
    elif show_plots:
        print("Analysis complete! Plots generated.")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  • Comprehensive results: {comprehensive_table_path}")
    print(f"  • Summary metrics: {summary_metrics_path}")
    print(f"  • Fold-level metrics: {metrics_folds_path}")
    if show_plots:
        print(f"  • Plots: {plots_root}")
    if show_plots:
        print(f"Plots root: {plots_root}")


if __name__ == "__main__":
    main()

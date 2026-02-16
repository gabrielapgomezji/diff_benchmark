import hydra
from omegaconf import DictConfig
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.analysis.plot_debug import plot_debug_run
from diff_benchmark.analysis.plot_script import plot_run
from diff_benchmark.analysis.plot_summary import plot_metrics_summary
from diff_benchmark.analysis.print_summary_table import is_successful_experiment, print_table, select_best_runs, table_best_means, table_detailed, table_folds_wide, table_weighted_aggregate, table_all_runs, table_model_aggregate
from pathlib import Path
import pandas as pd
from diff_benchmark.utils.job_manager import run_jobs


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


def build_comprehensive_table(experiments_root: Path, output_path: Path) -> pd.DataFrame:
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
                metric = row['metric']
                fold = row.get('fold', 0)
                split = row['split']
                value = row['value']
                
                col_name = f"{metric}_{split}_fold{fold}"
                exp_info[col_name] = value
            
            # Compute mean and std for each metric-split combination
            for split in df_metrics['split'].unique():
                df_split = df_metrics[df_metrics['split'] == split]
                
                for metric in df_split['metric'].unique():
                    df_metric = df_split[df_split['metric'] == metric]
                    
                    values = df_metric['value'].values
                    exp_info[f"{metric}_{split}_mean"] = np.mean(values)
                    exp_info[f"{metric}_{split}_std"] = np.std(values)
        
        # Flatten and add all config parameters
        # Focus on model, backend, pred_head, data, and target sections
        sections_to_include = ['model', 'backend', 'pred_head', 'data', 'target']
        
        for section in sections_to_include:
            if section in cfg:
                section_cfg = cfg[section]
                flat_params = flatten_config(section_cfg, prefix=f"config.{section}.")
                
                # Add all parameters, will be NaN for models that don't have them
                for param_key, param_value in flat_params.items():
                    exp_info[param_key] = param_value
        
        all_experiments.append(exp_info)
    
    if not all_experiments:
        raise RuntimeError("No valid experiments found")
    
    # Create DataFrame - pandas will automatically fill missing columns with NaN
    df_comprehensive = pd.DataFrame(all_experiments)
    
    # Sort columns for better readability
    # 1. Identifiers
    id_cols = ["run_id", "model_name", "dataset", "tissue_type", "primary_metric", "target", "prediction_task"]
    
    # 2. Metric columns (mean/std first, then individual folds)
    metric_cols = [col for col in df_comprehensive.columns if col not in id_cols and not col.startswith("config.")]
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
    
    # Filter out dummy models from report
    df_filtered = df_comprehensive[~df_comprehensive['model_name'].astype(str).str.contains('dummy', case=False, na=False)]

    # Identify config columns
    config_cols = [c for c in df_filtered.columns if c.startswith('config.')]
    
    # Group by dataset
    for dataset_name, df_dataset in df_filtered.groupby('dataset'):
        report_lines = []
        report_lines.append(f"DATASET REPORT: {dataset_name}")
        report_lines.append("=" * 140)
        
        # Group by model/tissue/task/target
        # Using fillna for grouping columns to handle potential NaNs safely
        # Removed primary_metric from grouping as requested
        group_keys = ['model_name', 'tissue_type', 'prediction_task', 'target'] 
        # Filter keys present in current df
        active_keys = [k for k in group_keys if k in df_dataset.columns]
        
        # We replace NaNs with string 'NaN' for grouping purposes to avoid dropping data
        df_dataset_safe = df_dataset.copy()
        for k in active_keys:
            df_dataset_safe[k] = df_dataset_safe[k].fillna('NaN')
            
        for group_values, df_group in df_dataset_safe.groupby(active_keys):
            # Restore original dataframe rows for this group to get correct types
            df_group_orig = df_dataset.loc[df_group.index]
            
            # Create a string describing the group
            group_desc = ", ".join(f"{k}={v}" for k, v in zip(active_keys, group_values))
            report_lines.append(f"\nGROUP: {group_desc}")
            report_lines.append("-" * 140)
            
            if len(df_group_orig) == 0:
                continue

            # Find best run
            prediction_task = str(df_group['prediction_task'].iloc[0]).lower()
            
            # Determine target metrics based on task
            metrics_priority = []
            if "classification" in prediction_task or "binary" in prediction_task:
                metrics_priority = ["accuracy_weighted", "accuracy", "roc_auc", "f1_weighted"]
            elif "regression" in prediction_task:
                metrics_priority = ["rmse_weighted", "rmse", "mae_weighted", "mae"]
            else:
                metrics_priority = ["accuracy_weighted", "rmse_weighted", "accuracy", "rmse"]

            metric_col = None
            metric_name = None
            
            # Find the best metric column (prioritizing test set but also coverage)
            # We want a metric that is available for most runs
            candidates = []
            for m in metrics_priority:
                for split in ['test', 'val']:
                    col = f"{m}_{split}_mean"
                    if col in df_group_orig.columns:
                        valid_count = df_group_orig[col].count()
                        if valid_count > 0:
                            candidates.append({
                                'col': col,
                                'metric': m,
                                'split': split,
                                'count': valid_count,
                                'priority_idx': metrics_priority.index(m)
                            })
            
            if candidates:
                # Sort candidates: 
                # 1. Coverage (descending)
                # 2. Priority of metric (ascending index)
                # 3. Split (Test preferred over Val? alphabetical test comes before val? No, test/val order in loop matters)
                # Let's prioritize coverage mostly, but break ties with metric priority and split
                
                # To prioritize test split over val split if counts are equal:
                # assign score to split: test=1, val=0
                for c in candidates:
                    c['split_score'] = 1 if c['split'] == 'test' else 0

                # Sort: most counts -> best metric priority -> test split
                candidates.sort(key=lambda x: (-x['count'], x['priority_idx'], -x['split_score']))
                
                best_candidate = candidates[0]
                metric_col = best_candidate['col']
                metric_name = best_candidate['metric']
                
            
            # Determine direction (default higher is better, switch for loss/error)
            lower_is_better = False
            if metric_name:
                 lower_is_better = any(x in metric_name.lower() for x in ['mae', 'rmse', 'mse', 'loss', 'error'])

            if metric_col is None:
                report_lines.append(f"  Warning: Could not find performance metric column. Task: {prediction_task}")
                # Fallback to just listing runs without sorting by metric
                metric_col = "run_id" # Dummy
                metric_name = "N/A"
                best_run = df_group_orig.iloc[0] 
                lower_is_better = False
            else:
                if lower_is_better:
                    best_idx = df_group_orig[metric_col].idxmin()
                else:
                    best_idx = df_group_orig[metric_col].idxmax()
                
                # Safety check for nan index
                if pd.isna(best_idx):
                     report_lines.append(f"  Warning: Metric column {metric_col} contained only NaNs.")
                     metric_col = "run_id"
                     metric_name = "N/A"
                     best_run = df_group_orig.iloc[0]
                     lower_is_better = False
                else:
                    best_run = df_group_orig.loc[best_idx]
            
            best_run_id = best_run['run_id']
            val_disp = f"{best_run[metric_col]:.4f}" if metric_col != "run_id" else "N/A"
            report_lines.append(f"  Best Run ID: {best_run_id} (Metric: {metric_col}, Score: {val_disp})")
            report_lines.append(f"  Highlighting: Parameters different from Best Run are enclosed in |...|")
            
            # Identify variable config params for this group
            # We filter for columns that have > 1 unique value across the group
            variable_config_cols = []
            for c in config_cols:
                if c not in df_group_orig.columns:
                    continue
                # Convert to string to compare uniqueness safely including NaNs
                unique_vals = df_group_orig[c].astype(str).unique()
                if len(unique_vals) > 1:
                    variable_config_cols.append(c)
            
            # Sort variable config cols by name
            variable_config_cols.sort()
            
            # Prepare table
            # Columns: RunID | Score | PrimaryMetric | VarConfig1 | VarConfig2 ...
            display_cols = ['run_id']
            if metric_col != "run_id":
                display_cols.append(metric_col)
            
            # Headers
            short_col_map = {c: c.replace('config.', '').replace('model.', '').replace('optimizer.', 'opt.').replace('backend.', 'bk.') for c in variable_config_cols}
            headers = ['RunID', 'Score', 'PrimaryMetric'] + [short_col_map[c] for c in variable_config_cols]
            
            # Data rows
            table_data = []
            
            # Sort: Best first, then by metric
            df_sorted = df_group_orig.copy()
            df_sorted['is_best'] = (df_sorted['run_id'] == best_run_id)
            
            if metric_col != "run_id":
                 if lower_is_better:
                    df_sorted = df_sorted.sort_values(['is_best', metric_col], ascending=[False, True])
                 else:
                    df_sorted = df_sorted.sort_values(['is_best', metric_col], ascending=[False, False])
            
            best_config_vals = {c: best_run[c] for c in variable_config_cols}
            best_primary_metric = best_run['primary_metric']

            for _, row in df_sorted.iterrows():
                row_data = []
                # RunID
                rid = str(row['run_id'])
                if row['run_id'] == best_run_id:
                    rid += " *" # Mark best
                row_data.append(rid)
                
                # Score
                if metric_col != "run_id":
                    val = row[metric_col]
                    current_metric_name = metric_name
                    
                    # Fallback if primary metric is missing
                    if pd.isna(val):
                        found_fallback = False
                        for m in metrics_priority:
                             for split in ['test', 'val']:
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
                    
                # Primary Metric
                p_metric = str(row['primary_metric'])
                if p_metric != str(best_primary_metric):
                     p_metric = f"|{p_metric}|"
                row_data.append(p_metric)
                
                # Configs
                for c in variable_config_cols:
                    val = row[c]
                    # Convert to string
                    val_str = str(val) if pd.notna(val) else "-"
                    
                    # Compare
                    best_val = best_config_vals[c]
                    
                    is_diff = False
                    # Comparison logic
                    v_str = str(val) if pd.notna(val) else "nan"
                    b_str = str(best_val) if pd.notna(best_val) else "nan"
                    
                    if v_str != b_str:
                        is_diff = True
                        
                    if is_diff:
                        # Highlight diff
                        val_str = f"|{val_str}|" # Using pipes to highlight
                    
                    # Truncate very long strings for readability
                    if len(val_str) > 30:
                        val_str = val_str[:27] + "..."
                        
                    row_data.append(val_str)
                
                table_data.append(row_data)
            
            # Render table
            col_widths = [len(h) for h in headers]
            for row in table_data:
                for i, val in enumerate(row):
                    if i < len(col_widths):
                        col_widths[i] = max(col_widths[i], len(val))
            
            # Add padding
            col_widths = [w + 2 for w in col_widths]
            
            # Cap column width to avoid explosion
            col_widths = [min(w, 50) for w in col_widths]
            
            fmt = "".join([f"{{:<{w}}}" for w in col_widths])
            
            # Print header
            try:
                report_lines.append(fmt.format(*headers))
                report_lines.append("-" * sum(col_widths))
            except Exception as e:
                report_lines.append(f"Error formatting table: {e}")
            
            # Print rows
            for row in table_data:
                try:
                    report_lines.append(fmt.format(*row))
                except Exception:
                    pass # Skip row if format fails
                    
            report_lines.append("\n" + "=" * 40 + "\n")

        # Write file
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
        'dummy_classifier': 'DUMMY',
        'dummy_regressor': 'DUMMY',
        'pca_linear': 'PCA_L',
        'linear': 'LIN',
        'pca_forest': 'PCA_F',
        'forest': 'FOR',
        'pca_svm': 'PCA_S',
        'svm': 'SVM',
        'medicalnet': 'MNET',
        'dinov2': 'DINO',
        'vit': 'VIT',
        'curia': 'CURIA',
    }
    
    # Target name mapping to abbreviations
    target_abbrev = {
        'age': 'a',
        'gender': 'g',
        'sex': 'g',  # Sometimes gender is called sex
        'diagnosis': 'd',
        'dxgroup': 'd',
    }
    
    # Create coverage dictionary
    # Structure: {(dataset, microstructure): {model: [tissue_target_codes]}}
    coverage = {}
    
    for _, row in df_comprehensive.iterrows():
        dataset = row['dataset']
        microstructure = row['primary_metric']  # sh, md, rtop, mk, etc.
        model = row['model_name']
        tissue = row['tissue_type'][0] if pd.notna(row['tissue_type']) else 'u'  # g, w, or u=unknown
        
        # Extract target abbreviation
        target_str = row['target'].lower()
        target_code = 'u'  # unknown
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
    all_models = sorted(set(
        model_code 
        for model_dict in coverage.values() 
        for model_code in model_dict.keys()
    ))
    
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
    table_lines.append(f"  Total dataset-microstructure combinations: {len(all_combos)}")
    table_lines.append(f"  Total models: {len(all_models)}")
    table_lines.append(f"  Total experiments: {len(df_comprehensive)}")
    
    # Model coverage
    table_lines.append("")
    table_lines.append("MODEL COVERAGE:")
    for model in all_models:
        count = sum(1 for combo_dict in coverage.values() if model in combo_dict)
        percentage = (count / len(all_combos)) * 100 if all_combos else 0
        table_lines.append(f"  {model:>10}: {count:3d}/{len(all_combos):3d} combinations ({percentage:5.1f}%)")
    
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
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"\n✓ Coverage table saved to: {output_file}")
    
    # Also print to console
    print("\n" + "="*80)
    print("EXPERIMENT COVERAGE TABLE")
    print("="*80)
    for line in table_lines[:min(50, len(table_lines))]:  # Print first 50 lines
        print(line)
    if len(table_lines) > 50:
        print(f"\n... ({len(table_lines) - 50} more lines in file)")


def process_experiment_plots(exp_dir: Path, plots_root: Path, force_plots: bool, debug_mode: bool) -> None:
    try:
        is_successful = is_successful_experiment(exp_dir)
        has_debug_info = (exp_dir / "debug").exists() and any((exp_dir / "debug").iterdir())

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
            main_plot_patterns = ["confusion_*.png", "roc_curve.png", "regression_*.png", "metrics_summary.png"]
            main_plots_exist = any(run_plots_dir.glob(pattern) for pattern in main_plot_patterns)
        
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
                plot_debug_run(run_id=run_id, debug_dir=debug_dir, output_root=plots_root)
            else:
                print(f"  Debug plots already exist for {run_id}, skipping...")

        # Main experiment plots if not already computed (Requires success usually implies metrics exist)
        # Only try to plot main results if metrics exist (which usually implies success or at least partial success)
        if not main_plots_exist:
            if metrics_path.exists() and predictions_path.exists() and targets_path.exists():
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
                    print(f"  Missing required files for main plots ({run_id}), despite success flag. Skipping...")
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
        df_comprehensive = build_comprehensive_table(experiments_root, comprehensive_table_path)
        print(f"✓ Comprehensive table saved to: {comprehensive_table_path}")
        print(f"  Shape: {df_comprehensive.shape[0]} experiments × {df_comprehensive.shape[1]} columns")

        # Build coverage table showing which experiments have been run
        tables_dir = summary_root / "tables"
        build_coverage_table(df_comprehensive, tables_dir)

        # Generate detailed reports per dataset
        reports_dir = summary_root / "reports"
        generate_dataset_reports(df_comprehensive, reports_dir)

        df_folds = build_global_metrics(experiments_root, metrics_folds_path)
        df_summary = build_summary_metrics(df_folds, summary_metrics_path)
        # df_metrics = load_global_metrics(summary_metrics_path)
        
        # -----------------------------------------------------------------
        # MODEL AGGREGATE STATISTICS (mean across all runs per model)
        # -----------------------------------------------------------------
        print("\n" + "="*80)
        print("MODEL AGGREGATE STATISTICS (Across All Runs)")
        print("="*80)
        print("Shows mean/std/min/max for each model across all datasets and runs")
        
        print("\n--- Classification: Accuracy ---")
        df_model_acc = table_model_aggregate(df_summary, primary_metric="accuracy_weighted")
        print_table(df_model_acc)
        
        print("\n--- Regression: MAE ---")
        df_model_mae = table_model_aggregate(df_summary, primary_metric="mae_weighted")
        print_table(df_model_mae)
        
        print("\n--- Regression: RMSE ---")
        df_model_rmse = table_model_aggregate(df_summary, primary_metric="rmse_weighted")
        print_table(df_model_rmse)
        
        print("\n--- Regression: Pearson Correlation ---")
        df_model_corr = table_model_aggregate(df_summary, primary_metric="pearson_correlation")
        print_table(df_model_corr)
        
        # -----------------------------------------------------------------
        # ALL RUNS RESULTS (shows every run, not just best)
        # -----------------------------------------------------------------
        print("\n" + "="*80)
        print("ALL RUNS RESULTS (Every Model Run)")
        print("="*80)
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
        print("\n" + "="*80)
        print("BEST MEAN RESULTS (Best Run per Model/Dataset)")
        print("="*80)
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
        # try:
        #     print("--- Primary Metric: correlation ---")
        #     df_best = table_best_means(df_summary, primary_metric="rmse")
        #     print_table(df_best)
        # except Exception as e:
        #     print(f"Skipping RMSE table: {e}")

        best_runs = select_best_runs(df_summary, primary_metric="accuracy_weighted")
        # print("\n=== DETAILED RESULTS (BEST RUN PER MODEL) ===")
        # df_detailed = table_detailed(df_folds, best_runs, primary_metric="accuracy")
        # print_table(df_detailed)

        # df_wide_train = table_folds_wide(df_folds, best_runs, split="train", primary_metric="accuracy")
        # print("\n=== WIDE-FORMAT RESULTS TRAIN ===")
        # print_table(df_wide_train)

        print("\n=== WIDE-FORMAT RESULTS TEST ===")
        df_wide_test = table_folds_wide(df_folds, best_runs, split="test", primary_metric="accuracy_weighted")
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
                    print_table(table_best_means(df_pm, primary_metric="accuracy_weighted"))
                if "regression" in df_pm["prediction_task"].values:
                    print(f"--- Regression ---")
                    print_table(table_best_means(df_pm, primary_metric="rmse_weighted"))
        
        # -----------------------------------------------------------------
        # Cross-tissue comparison
        # -----------------------------------------------------------------
        print("\n" + "="*80)
        print("CROSS-TISSUE COMPARISON")
        print("="*80)
        
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
        for exp_dir in experiments_root.glob("exp_*"):
            process_experiment_plots(exp_dir, plots_root, force_plots, debug_mode)

        print("\nPlots generation complete!")
        print(f"Plots saved to: {plots_root}")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "="*80)
    if show_tables and show_plots:
        print("Analysis complete! Tables printed and plots generated.")
    elif show_tables:
        print("Analysis complete! Tables printed.")
    elif show_plots:
        print("Analysis complete! Plots generated.")
    print("="*80)
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

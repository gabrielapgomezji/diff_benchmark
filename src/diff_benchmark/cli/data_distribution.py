import hydra
from omegaconf import DictConfig
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, Any
import json

from diff_benchmark.data.prepare_data import DatasetPreparation
from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from skrub import TableReport


def generate_skrub_report(demographics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate a comprehensive HTML report using skrub's TableReport.
    
    Args:
        demographics_df: DataFrame with demographics data
        output_dir: Directory to save the report
    """    
    print("\n=== Generating Skrub TableReport ===")
    try:
        # Create TableReport
        report = TableReport(demographics_df)
        
        # Save HTML report
        html_path = output_dir / 'skrub_table_report.html'
        html_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the report
        with open(html_path, 'w') as f:
            f.write(report._repr_html_())
        
        print(f"Saved skrub TableReport: {html_path}")
    except Exception as e:
        print(f"Error generating skrub report: {e}")



def compute_variable_statistics(series: pd.Series) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a variable.
    
    Args:
        series: Pandas Series containing the variable data
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "name": series.name,
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "dtype": str(series.dtype),
    }
    
    # Check if numeric (but exclude boolean types which can cause issues)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        # Drop NaN values for statistics computation
        clean_series = series.dropna()
        if len(clean_series) > 0:
            stats.update({
                "type": "numeric",
                "mean": float(clean_series.mean()),
                "std": float(clean_series.std()),
                "min": float(clean_series.min()),
                "q25": float(clean_series.quantile(0.25)),
                "median": float(clean_series.median()),
                "q75": float(clean_series.quantile(0.75)),
                "max": float(clean_series.max()),
                "skewness": float(clean_series.skew()),
                "kurtosis": float(clean_series.kurtosis()),
            })
        else:
            # All values are NaN
            stats.update({
                "type": "numeric",
                "mean": None,
                "std": None,
                "min": None,
                "q25": None,
                "median": None,
                "q75": None,
                "max": None,
                "skewness": None,
                "kurtosis": None,
            })
    else:
        # Categorical variable (includes boolean)
        value_counts = series.value_counts()
        stats.update({
            "type": "categorical",
            "n_unique": int(series.nunique()),
            "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "value_counts": {str(k): int(v) for k, v in value_counts.to_dict().items()},
        })
    
    return stats


def plot_variable_distribution(series: pd.Series, output_path: Path, is_target: bool = False):
    """
    Create distribution plot for a variable.
    
    Args:
        series: Pandas Series containing the variable data
        output_path: Path to save the plot
        is_target: Whether this is a target variable
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if pd.api.types.is_numeric_dtype(series):
        # Numeric variable - histogram with KDE
        ax.hist(series.dropna(), bins=30, alpha=0.7, edgecolor='black', density=True)
        
        # Add KDE if enough samples
        if len(series.dropna()) > 10:
            from scipy import stats as scipy_stats
            kde = scipy_stats.gaussian_kde(series.dropna())
            x_range = np.linspace(series.min(), series.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            ax.legend()
        
        ax.set_xlabel(series.name)
        ax.set_ylabel('Density')
        title = f'Distribution of {series.name}'
        if is_target:
            title += ' (Target Variable)'
        ax.set_title(title)
        
        # Add statistics text box
        stats_text = (
            f'Mean: {series.mean():.2f}\n'
            f'Std: {series.std():.2f}\n'
            f'Min: {series.min():.2f}\n'
            f'Max: {series.max():.2f}\n'
            f'N: {series.count()}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Categorical variable - bar plot
        value_counts = series.value_counts()
        value_counts.plot(kind='bar', ax=ax, alpha=0.7, edgecolor='black')
        ax.set_xlabel(series.name)
        ax.set_ylabel('Count')
        title = f'Distribution of {series.name}'
        if is_target:
            title += ' (Target Variable)'
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        
        # Add counts on bars
        for i, (idx, val) in enumerate(value_counts.items()):
            ax.text(i, val, str(val), ha='center', va='bottom')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {output_path}")


def generate_summary_report(demographics_df: pd.DataFrame, target_columns: list, 
                           output_dir: Path) -> pd.DataFrame:
    """
    Generate comprehensive summary report for all variables in the dataset.
    
    Args:
        demographics_df: DataFrame with demographics data
        target_columns: List of target column names
        output_dir: Directory to save outputs
        
    Returns:
        DataFrame with summary statistics
    """
    print("\n=== Generating Summary Report ===")
    
    # Compute statistics for all variables
    all_stats = []
    for col in demographics_df.columns:
        if col == 'Subject':
            continue
        
        stats = compute_variable_statistics(demographics_df[col])
        stats['is_target'] = col in target_columns
        all_stats.append(stats)
    
    # Create summary DataFrame for numeric variables
    numeric_summary_data = []
    categorical_summary_data = []
    
    for stats in all_stats:
        if stats['type'] == 'numeric':
            numeric_summary_data.append({
                'variable': stats['name'],
                'is_target': stats['is_target'],
                'count': stats['count'],
                'missing': stats['missing'],
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'q25': stats['q25'],
                'median': stats['median'],
                'q75': stats['q75'],
                'max': stats['max'],
                'skewness': stats['skewness'],
                'kurtosis': stats['kurtosis'],
            })
        else:
            categorical_summary_data.append({
                'variable': stats['name'],
                'is_target': stats['is_target'],
                'count': stats['count'],
                'missing': stats['missing'],
                'n_unique': stats['n_unique'],
                'top_value': stats['top_value'],
                'top_frequency': stats['top_frequency'],
            })
    
    # Save detailed JSON
    json_path = output_dir / 'summary_detailed.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved detailed summary: {json_path}")
    
    # Create and save summary DataFrames
    if numeric_summary_data:
        numeric_df = pd.DataFrame(numeric_summary_data)
        numeric_parquet_path = output_dir / 'summary_numeric.parquet'
        numeric_df.to_parquet(numeric_parquet_path, index=False)
        print(f"Saved numeric summary: {numeric_parquet_path}")
        
        # Also save as CSV for easy viewing
        numeric_csv_path = output_dir / 'summary_numeric.csv'
        numeric_df.to_csv(numeric_csv_path, index=False)
        print(f"Saved numeric summary CSV: {numeric_csv_path}")
    
    if categorical_summary_data:
        categorical_df = pd.DataFrame(categorical_summary_data)
        categorical_parquet_path = output_dir / 'summary_categorical.parquet'
        categorical_df.to_parquet(categorical_parquet_path, index=False)
        print(f"Saved categorical summary: {categorical_parquet_path}")
        
        categorical_csv_path = output_dir / 'summary_categorical.csv'
        categorical_df.to_csv(categorical_csv_path, index=False)
        print(f"Saved categorical summary CSV: {categorical_csv_path}")
    
    # Create a combined summary parquet with all variables and their distribution info
    # This can be used to determine which variables to plot
    combined_summary = []
    for stats in all_stats:
        summary_row = {
            'variable': stats['name'],
            'is_target': stats['is_target'],
            'type': stats['type'],
            'count': stats['count'],
            'missing': stats['missing'],
            'missing_pct': (stats['missing'] / (stats['count'] + stats['missing']) * 100) if (stats['count'] + stats['missing']) > 0 else 0,
        }
        
        if stats['type'] == 'numeric':
            summary_row.update({
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'median': stats['median'],
                'max': stats['max'],
                'n_unique': None,
            })
        else:
            summary_row.update({
                'mean': None,
                'std': None,
                'min': None,
                'median': None,
                'max': None,
                'n_unique': stats['n_unique'],
            })
        
        combined_summary.append(summary_row)
    
    combined_df = pd.DataFrame(combined_summary)
    combined_parquet_path = output_dir / 'summary_all_variables.parquet'
    combined_df.to_parquet(combined_parquet_path, index=False)
    print(f"Saved combined summary: {combined_parquet_path}")
    
    # Also save as CSV
    combined_csv_path = output_dir / 'summary_all_variables.csv'
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined summary CSV: {combined_csv_path}")
    
    # Combine for return
    all_summary_data = numeric_summary_data + categorical_summary_data
    if all_summary_data:
        return pd.DataFrame(all_summary_data)
    else:
        return pd.DataFrame()


def plot_target_distributions(demographics_df: pd.DataFrame, target_columns: list, 
                              output_dir: Path, plots_dir: Path):
    """
    Generate plots for all target variables.
    
    Args:
        demographics_df: DataFrame with demographics data
        target_columns: List of target column names
        output_dir: Directory for output files
        plots_dir: Directory for plots
    """
    print("\n=== Plotting Target Distributions ===")
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plotted_targets = []
    
    for target in target_columns:
        if target not in demographics_df.columns:
            print(f"  Warning: Target '{target}' not found in demographics data")
            continue
        
        # Check if plot already exists
        plot_path = plots_dir / f'{target}_distribution.png'
        
        if plot_path.exists():
            print(f"  Plot already exists for '{target}': {plot_path}")
        else:
            print(f"  Creating plot for target '{target}'...")
            plot_variable_distribution(
                demographics_df[target], 
                plot_path, 
                is_target=True
            )
            plotted_targets.append(target)
    
    # Save metadata about plotted targets
    metadata_path = output_dir / 'plots_metadata.json'
    metadata = {
        'target_columns': list(target_columns),  # Convert to regular list for JSON serialization
        'plotted_targets': plotted_targets,
        'plots_directory': str(plots_dir),
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved plot metadata: {metadata_path}")


def plot_all_variables(demographics_df: pd.DataFrame, output_dir: Path, 
                       target_columns: list):
    """
    Create distribution plots for all variables in the dataset.
    
    Args:
        demographics_df: DataFrame with demographics data
        output_dir: Directory to save plots
        target_columns: List of target column names
    """
    print("\n=== Plotting All Variable Distributions ===")
    
    all_plots_dir = output_dir / 'plots' / 'all_variables'
    all_plots_dir.mkdir(parents=True, exist_ok=True)
    
    for col in demographics_df.columns:
        if col == 'Subject':
            continue
        
        plot_path = all_plots_dir / f'{col}_distribution.png'
        
        if not plot_path.exists():
            print(f"  Creating plot for '{col}'...")
            is_target = col in target_columns
            plot_variable_distribution(demographics_df[col], plot_path, is_target)


@hydra.main(
    version_base="1.3",
    config_path="pkg://diff_benchmark.configs",
    config_name="main",
)
def main(cfg: DictConfig) -> None:
    """
    CLI entrypoint:
        diffbenchmark-distribution [hydra overrides]

    Computes data distribution information and generates plots for configured datasets.
    """
    print("=" * 80)
    print("RUNNING DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Setup dataset configuration
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]

    dataset_selected = DatasetConfig(
        **dataset_cfg,
        base_dir=Path(cluster_cfg.base_dir),
        results_dir=Path(cluster_cfg.results_dir),
    )

    # Prepare data
    print(f"\nDataset: {dataset_selected.name}")
    print(f"Target columns: {cfg.target.target_column}")
    
    # Setup output directories early
    output_root = Path("./exp_outputs/datasets") / dataset_selected.name
    output_root.mkdir(parents=True, exist_ok=True)
    subjects_cache_path = output_root / 'available_subjects.json'
    demographics_parquet_path = output_root / 'full_demographics.parquet'
    breakpoint()
    # Check if we can load demographics from parquet (fast path)
    if demographics_parquet_path.exists() and not cfg.runtime.get('force', False):
        print("\nLoading demographics from cached parquet file...")
        demographics_df = pd.read_parquet(demographics_parquet_path)
        print(f"Loaded demographics shape: {demographics_df.shape}")
        print(f"  (Skipping brain data loading - use runtime.force=true to recompute)")
        
        # We still need to generate reports if they don't exist
        need_full_processing = False
    else:
        # Need to do full processing: load brain data and demographics from source
        need_full_processing = True
        
        # First, check if we have cached available_subjects to avoid recomputing brain data
        if subjects_cache_path.exists() and not cfg.runtime.get('force', False):
            print("\nLoading cached available subjects...")
            with open(subjects_cache_path, 'r') as f:
                available_subjects = json.load(f)
            print(f"Loaded {len(available_subjects)} subjects from cache")
            
            # Still need to get demographics file path, but skip brain loading
            if dataset_selected.name == "hcp":
                cog_file = cfg.cluster.paths[dataset_selected.name].csv_file
            else:
                # We need layouts for non-HCP datasets
                print("  Note: Loading brain preparator for file paths (not recomputing features)...")
                from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline
                brain_preparator = DefaultPipeline(dataset_selected)
                from diff_benchmark.data.prepare_data import DatasetPreparation
                temp_preparator = DatasetPreparation(cfg=cfg, source_dataset=dataset_selected)
                cog_file = temp_preparator._extract_participants_files_from_layouts(brain_preparator.layouts)
        else:
            # First run or force recompute: get brain data to know which subjects have brain imaging available
            print("\nLoading brain data...")
            print("  (This may take a while on first run, but will be cached for future runs)")
            # We need to initialize the brain preparator without calling get_model
            # For distribution analysis, we can use the default pipeline directly
            from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline
            brain_preparator = DefaultPipeline(dataset_selected)
            brain_df = brain_preparator.load_features().reset_index()
            available_subjects = brain_df["subject_id"].astype(str).unique().tolist()
            print(f"Found {len(available_subjects)} subjects with brain data")
            
            # Get demographics file path
            if dataset_selected.name == "hcp":
                cog_file = cfg.cluster.paths[dataset_selected.name].csv_file
            else:
                # Extract from BIDS layouts
                from diff_benchmark.data.prepare_data import DatasetPreparation
                temp_preparator = DatasetPreparation(cfg=cfg, source_dataset=dataset_selected)
                cog_file = temp_preparator._extract_participants_files_from_layouts(brain_preparator.layouts)
            
            # Save available_subjects for future reuse
            with open(subjects_cache_path, 'w') as f:
                json.dump(available_subjects, f, indent=2)
            print(f"Cached available subjects to: {subjects_cache_path}")
        
        # Get FULL demographics dataframe (all columns) filtered by available subjects
        print("\nLoading full demographics data (all potential targets)...")
        from diff_benchmark.preprocessing.preparation_pipeline import DemographicsPreparationPipeline
        preprocessor = DemographicsPreparationPipeline(cog_file)
        demographics_df = preprocessor.get_full_demographics(available_subjects)
        print(f"Demographics shape: {demographics_df.shape}")
        print(f"Available columns: {list(demographics_df.columns)}")
    plots_dir = output_root / "plots" / "targets"
    
    # Only generate reports and summaries if doing full processing (first run or force)
    if need_full_processing:
        # Generate skrub TableReport
        # Note: For large datasets with many columns, we create a simplified version
        # to avoid compilation issues with too many distribution plots
        print("\nGenerating Skrub TableReport...")
        try:
            # For datasets with many columns, create a summary version
            if demographics_df.shape[1] > 50:
                print(f"  Dataset has {demographics_df.shape[1]} columns. Creating summary version.")
                # Select only numeric columns and a subset for the report
                numeric_cols = [col for col in demographics_df.columns 
                               if pd.api.types.is_numeric_dtype(demographics_df[col])]
                report_df = demographics_df[['Subject'] + numeric_cols[:20]]  # Limit to 20 numeric columns
                print(f"  Using {len(report_df.columns)} columns for TableReport")
            else:
                report_df = demographics_df
            
            generate_skrub_report(report_df, output_root)
        except Exception as e:
            print(f"  Warning: Could not generate skrub report: {e}")
            print("  Continuing with other analyses...")
        
        # Save the full demographics dataframe as parquet
        # This allows for efficient reloading and selective plotting later
        demographics_df.to_parquet(demographics_parquet_path, index=False)
        print(f"\nSaved full demographics to: {demographics_parquet_path}")
        print("  (Can be reloaded on subsequent runs for fast target plotting)")
        
        # Identify all potential target columns (numeric columns, excluding Subject)
        potential_targets = [
            col for col in demographics_df.columns 
            if col != 'Subject' and pd.api.types.is_numeric_dtype(demographics_df[col])
        ]
        print(f"\nIdentified {len(potential_targets)} potential numeric target columns")
        
        # Generate summary report for ALL columns
        summary_df = generate_summary_report(
            demographics_df, 
            cfg.target.target_column,  # Mark which ones are currently configured as targets
            output_root
        )
    else:
        print("\nSkipping report generation (parquet already exists)")
        print("  To regenerate reports, use runtime.force=true")
    
    # ALWAYS check and plot distributions for configured target variables
    # This works efficiently on subsequent runs by loading from parquet
    print(f"\nChecking target variable plots...")
    plot_target_distributions(
        demographics_df,
        cfg.target.target_column,
        output_root,
        plots_dir
    )
    
    print("\n" + "=" * 80)
    print("DATA DISTRIBUTION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_root}")
    if need_full_processing:
        print(f"  ✓ Generated full reports and cached data")
    else:
        print(f"  ✓ Used cached data (fast!)")
    print(f"\nKey files:")
    print(f"  - Full demographics: {demographics_parquet_path}")
    print(f"  - Target plots: {plots_dir}")
    print(f"  - Skrub report: {output_root / 'skrub_table_report.html'}")
    print(f"  - Summary stats: {output_root / 'summary_*.parquet'}")
    print(f"\nWorkflow tips:")
    print(f"  - Change target: Just run again with different target= (plots only new targets)")
    print(f"  - Force recompute: Use runtime.force=true to regenerate everything")
    print(f"  - Plot other vars: Use the helper script or load the parquet file")



if __name__ == "__main__":
    main()

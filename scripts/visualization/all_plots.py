from __future__ import annotations

import argparse
from pathlib import Path

from combined_plot import plot_combined
from delta_to_linear_plot import plot_delta_to_linear
from feature_vs_b0_plot import plot_feature_vs_b0
from feature_plots import plot_feature_heatmap
from prep_sensitivity_plot import plot_prep_sensitivity
from show_spread import plot_model_family_spread
from strip_plots import generate_strip_plots
from white_vs_gray_plots import plot_white_vs_gray_tscore


DEFAULT_INPUT = "exp_outputs/summary/comprehensive_results.parquet"
DEFAULT_OUTDIR = "exp_outputs/summary/plots"


def generate_all_plots(
    parquet_path: str = DEFAULT_INPUT,
    out_dir: str = DEFAULT_OUTDIR,
    best_run: bool = True,
) -> Path:
    out_path = Path(out_dir)
    folds_dir = out_path / "folds"
    features_dir = out_path / "features"

    generate_strip_plots(parquet_path, str(folds_dir), best_run=best_run)
    print(f"[1/8] Strip plots generated in {folds_dir}")

    plot_white_vs_gray_tscore(parquet_path, str(folds_dir))
    print(f"[2/8] White-vs-gray plots generated in {folds_dir}")

    plot_feature_heatmap(parquet_path, str(features_dir))
    print(f"[3/8] Feature heatmap generated in {features_dir}")

    plot_model_family_spread(parquet_path, str(folds_dir))
    print(f"[4/8] Spread plot generated in {folds_dir}")

    plot_delta_to_linear(parquet_path, str(folds_dir))
    print(f"[5/8] Delta-to-linear plot generated in {folds_dir}")

    plot_feature_vs_b0(parquet_path, str(folds_dir))
    print(f"[6/8] Feature-vs-b0 deltas plot generated in {folds_dir}")

    plot_prep_sensitivity(parquet_path, str(folds_dir))
    print(f"[7/8] Prep-sensitivity plot generated in {folds_dir}")

    plot_combined(parquet_path, str(folds_dir))
    print(f"[8/8] Combined plot generated in {folds_dir}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all visualization plots")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input parquet file",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Base output directory for generated plots",
    )
    parser.add_argument(
        "--no-best-run",
        action="store_false",
        dest="best_run",
        help="Disable best-run filtering in strip plots",
    )
    args = parser.parse_args()

    out_path = generate_all_plots(args.input, args.outdir, best_run=args.best_run)
    print("Saved all plots to", out_path)


if __name__ == "__main__":
    main()

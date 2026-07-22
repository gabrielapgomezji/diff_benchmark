from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import (
    DEFAULT_COMBOS,
    build_strip_data,
    choose_fold_metric,
    clean_target,
    filter_combos,
    format_label,
)

from config import apply_miccai_style


DEFAULT_INPUT = "exp_outputs/summary/comprehensive_results.parquet"


def _parse_models_arg(models: str | None) -> list[str] | None:
    if models is None:
        return None
    parsed = [m.strip() for m in models.split(",") if m.strip()]
    return parsed or None


def generate_strip_plots(
    parquet_path: str,
    out_dir: str = "analysis_results/visualization_demo/plots/folds",
    best_run: bool = True,
    model_names: list[str] | None = None,
    use_default_combos: bool = False,
) -> Path:
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)

    if model_names:
        model_set = set(model_names)
        df = df[df["model_name"].astype(str).isin(model_set)].copy()

    if use_default_combos:
        df = filter_combos(df, DEFAULT_COMBOS)

    if df.empty:
        raise RuntimeError("No rows left after filtering combos")

    for (dataset, target, task), group in df.groupby(
        ["dataset", "target_clean", "prediction_task"]
    ):
        fold_prefix, metric_label, higher_is_better = choose_fold_metric(group, task)
        strip_df = build_strip_data(
            group, task, fold_prefix, higher_is_better, best_run
        )
        if strip_df.empty:
            continue

        labels = strip_df["label"].unique().tolist()
        labels.sort()

        fig_w = max(9, 0.45 * len(labels))
        plt.figure(figsize=(fig_w, 5))
        rng = np.random.default_rng(7)

        for i, label in enumerate(labels):
            vals = strip_df[strip_df["label"] == label]["value"].values
            jitter = (rng.random(len(vals)) - 0.5) * 0.15
            plt.scatter(np.full(len(vals), i) + jitter, vals, s=14, alpha=0.7)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            plt.errorbar(
                i,
                mean_val,
                yerr=std_val,
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
                markersize=4,
                zorder=5,
            )

        pretty_labels = [format_label(l) for l in labels]
        plt.xticks(
            np.arange(len(labels)), pretty_labels, rotation=45, ha="right", fontsize=8
        )
        plt.ylabel(metric_label)
        if metric_label == "R2":
            plt.ylim(0, 1)
        title = f"{dataset} | {target} | {task}"
        plt.title(format_label(title))
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        filename = f"strip_{dataset}_{target}_{task}.pdf"
        plt.savefig(out_path / filename, dpi=300)
        plt.close()

        # Box plot
        plt.figure(figsize=(fig_w, 5))

        data_vals = [
            strip_df[strip_df["label"] == label]["value"].values for label in labels
        ]
        plt.boxplot(data_vals, positions=np.arange(len(labels)))

        plt.xticks(
            np.arange(len(labels)), pretty_labels, rotation=45, ha="right", fontsize=8
        )
        plt.ylabel(metric_label)
        if metric_label == "R2":
            plt.ylim(0, 1)
        plt.title(format_label(title))
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        filename_box = f"box_{dataset}_{target}_{task}.pdf"
        plt.savefig(out_path / filename_box, dpi=300)
        plt.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate strip plots per dataset/task combination"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Input parquet file"
    )
    parser.add_argument(
        "--outdir",
        default="analysis_results/visualization_demo/plots/folds",
        help="Output directory",
    )
    parser.add_argument("--no-best-run", action="store_false", dest="best_run")
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model names to include "
            "(e.g. pointnet,region_elasticnet,region_group_lasso)"
        ),
    )
    parser.add_argument(
        "--use-default-combos",
        action="store_true",
        help="Restrict to built-in dataset/target/task combos",
    )
    args = parser.parse_args()

    model_names = _parse_models_arg(args.models)

    out_path = generate_strip_plots(
        args.input,
        args.outdir,
        best_run=args.best_run,
        model_names=model_names,
        use_default_combos=args.use_default_combos,
    )
    print("Saved strip plots to", out_path)


if __name__ == "__main__":
    main()

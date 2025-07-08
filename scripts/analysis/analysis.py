from pathlib import Path

import yaml

from diff_benchmark.analysis.plot_results import plot_predictions_vs_targets

with open(Path(__file__).parent.parent.parent / "configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

# -------- PLOT MEAN PRED VS TARGETS --------
plot_predictions_vs_targets(
    summary_path=Path(config["results_path_2"])
    / "analysis_results"
    / "cca_summary.json",
    output_dir=Path(config["results_path_2"]) / "analysis_results" / "plots",
)

# -------- PLOT PER FOLD PRED VS TARGETS --------

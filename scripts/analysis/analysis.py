from pathlib import Path

import yaml

from diff_benchmark.analysis.plot_results import (
    plot_folds_predictions_vs_targets,
    plot_predictions_vs_targets,
)
from diff_benchmark.analysis.scores_summary import summarize_folds_to_csv

with open(Path(__file__).parent.parent.parent / "configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

# -------- PLOT MEAN PRED VS TARGETS --------
# plot_predictions_vs_targets(
#     summary_path=Path(config["results_path_2"])
#     / "analysis_results"
#     / "cca_summary.json",
#     output_dir=Path(config["results_path_2"]) / "analysis_results" / "plots",
# )

# -------- PLOT PER FOLD PRED VS TARGETS --------
plot_folds_predictions_vs_targets(
    summary_path=Path(config["results_path_2"])
    / "analysis_results"
    / f"{config["model_name"]}_fold_results.json",
    output_dir=Path(config["results_path_2"]) / "analysis_results" / "plots",
)

# -------- PER FOLD SCORE TABLE --------
summarize_folds_to_csv(
    fold_results_path=Path(config["results_path_2"])
    / "analysis_results"
    / f"{config["model_name"]}_fold_results.json",
    output_csv_path=Path(config["results_path_2"])
    / "analysis_results"
    / f"{config["model_name"]}_score_stats.csv",
)

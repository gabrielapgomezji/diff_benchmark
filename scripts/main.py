import os
from pathlib import Path

import yaml

from diff_benchmark.analysis.plot_results import plot_folds_predictions_vs_targets
from diff_benchmark.analysis.save_results import save_fold_results
from diff_benchmark.analysis.scores_summary import summarize_folds_to_csv
from diff_benchmark.dataloaders.dataloaders import PreprocessedData
from diff_benchmark.dataset.generate_dataset import (
    CustomDataset,
    CustomDatasetBuilder,
)
from diff_benchmark.dataset.loading_strategies import AttenuationStrategy
from diff_benchmark.dataset.read_save_dataset import load_dataset
from diff_benchmark.models.model_configurations import get_model
from diff_benchmark.preprocessing.preprocess_brain_data import (
    DefaultBrainPreprocessor,
)
from diff_benchmark.preprocessing.preprocess_demographic_data import (
    DefaultDemographicsPreprocessor,
)
from diff_benchmark.raw_data.process_raw_data import DWIProcessor
from diff_benchmark.scores.scores import mse_score
from diff_benchmark.utils.file_renaming import rename_files_in_parallel

config_path = Path(__file__).parent.parent / "configuration.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
DEBUG = config["debugging_analysis"]

from diff_benchmark.raw_data.process_raw_data import DWISchaeferProcessor

processor = DWISchaeferProcessor(config)
processor.run_parallel()
breakpoint()
# ----------- FILE RENAMING -----------
# rename_files_in_parallel(
#     base_path=Path(config["results_path"]),
#     old_file_name="mapmri_default_all_bvals.h5",
#     new_file_name="7_mapmri_default_all_bvals.h5",
#     n_jobs=config["n_jobs"],
# )

if os.path.exists(Path(config["results_path"]) / "datasets" / "dataset.h5"):
    # ----------- LOAD DATASET IF EXISTS ALREADY -----------
    print("Dataset already exists, loading from file...")
    X, y, gender = load_dataset(
        Path(config["results_path"]) / "datasets" / "dataset.h5"
    )
    dataset = CustomDataset(X, y, gender)
else:
    # ---------- RUN PREPROCESSING FOR RAW DATA ----------

    processor = DWIProcessor(config)
    processor.run_parallel()

    # ---------- RUN PREPROCESSING FOR INPUT DATA ----------

    preprocessor = DefaultBrainPreprocessor(config)
    preprocessor.preprocess_dataset()

    # ---------- RUN PREPROCESSING FOR TARGET DATA ----------

    preprocessor = DefaultDemographicsPreprocessor(config["csv_file"])
    df_clean = preprocessor.preprocess(config["target_columns"])

    # ----------- SAVE PROCESSED DATA ----------

    name = "mapmri_default"
    loading_strategy = AttenuationStrategy()
    builder = CustomDatasetBuilder(
        base_path=config["results_path"],
        loading_strategy=loading_strategy,
        df_targets=df_clean,
        h5_filename=f"{name}_all_bvals.h5",
        output_dataset_filename=Path(config["results_path"])
        / "datasets"
        / "dataset.h5",
    )
    X, y, subjects, gender = builder.create_dataset(n_jobs=config["n_jobs"])
    dataset = CustomDataset(X, y, gender)

# ----------- CROSS VALIDATION + TRAINING + TESTING -----------


preprocessed = PreprocessedData(
    X, y, gender, n_splits=config["n_splits"], random_state=config["random_state"]
)

specs = preprocessed.get_specs()
print(specs)

# folds = preprocessed.get_folds_as_dataloaders(batch_size=16)
indices = preprocessed.get_fold_indices()

train_scores, test_scores = [], []
train_preds, test_preds = [], []
train_targets, test_targets = [], []
per_fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(indices):
    print(
        f"Fold {fold_idx+1} - Train samples: {len(train_idx)}, test_samples: {len(test_idx)}"
    )
    train_loader, test_loader = preprocessed.get_dataloader_fold(
        dataset, fold_idx, indices
    )
    _, y_train, _, _, y_test, _ = preprocessed.get_arrays_from_indices(
        dataset, fold_idx, indices
    )

    model = get_model(config["model_name"], config)

    # --------- Train / Val / Test Model ---------
    print("Training...")
    model.fit(train_loader)
    train_pred = model.predict(train_loader)
    train_score = mse_score(y_train, train_pred)
    print(train_score)

    train_scores.append(train_score)
    train_preds.append(train_pred.tolist())
    train_targets.append(y_train.tolist())

    print("Testing...")
    test_pred = model.predict(test_loader)
    test_score = mse_score(y_test, test_pred)
    print(test_score)

    test_scores.append(test_score)
    test_preds.append(test_pred.tolist())
    test_targets.append(y_test.tolist())

    print(f"Done Fold {fold_idx + 1}")
    if DEBUG:
        per_fold_results.append(
            {
                "model": config["model_name"],
                "fold": fold_idx,
                "train": {
                    "score": float(train_score),
                    "predictions": train_pred.tolist(),
                    "targets": y_train.tolist(),
                },
                "test": {
                    "score": float(test_score),
                    "predictions": test_pred.tolist(),
                    "targets": y_test.tolist(),
                },
            }
        )


print("\n Saving results...")
if DEBUG:
    save_fold_results(
        model_name=config["model_name"],
        fold_results=per_fold_results,
        output_dir=Path(config["results_path"]) / "analysis_results",
    )

# ------------ EVALUATION AND ANALYSIS ------------


# -------- PLOT PER FOLD PRED VS TARGETS --------
plot_folds_predictions_vs_targets(
    summary_path=Path(config["results_path"])
    / "analysis_results"
    / f"{config["model_name"]}_fold_results.json",
    output_dir=Path(config["results_path"]) / "analysis_results" / "plots",
)

# -------- PER FOLD SCORE TABLE --------
summarize_folds_to_csv(
    fold_results_path=Path(config["results_path"])
    / "analysis_results"
    / f"{config["model_name"]}_fold_results.json",
    output_csv_path=Path(config["results_path"])
    / "analysis_results"
    / f"{config["model_name"]}_score_stats.csv",
)

import copy
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

from diff_benchmark.analysis.plot_results import plot_folds_predictions_vs_targets
from diff_benchmark.analysis.save_results import save_fold_results
from diff_benchmark.analysis.scores_summary import summarize_folds_to_csv
from diff_benchmark.dataloaders.dataloaders import PreprocessedData
from diff_benchmark.dataset.generate_dataset import CustomDataset
from diff_benchmark.models.model_configurations import get_model
from diff_benchmark.preprocessing.preprocess_demographic_data import (
    DefaultDemographicsPreprocessor,
)
from diff_benchmark.preprocessing.wrapper_brain_data import (  # LcotEmbedHcpPipeline,
    DefaultHcpPipeline,
)
from diff_benchmark.scores.scores import accuracy_score  # , mse_score

DEBUG = True

config_path = Path(__file__).parent.parent / "configuration.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["metric_to_compute"] = "md"
brain_preparator = DefaultHcpPipeline(config)
brain_df = brain_preparator.run_pipeline()
brain_df = brain_df.reset_index()
# breakpoint()


##### NEXT TESTING STEPS
preprocessor = DefaultDemographicsPreprocessor(config["csv_file"])
demographics_df = preprocessor.preprocess(config["target_columns"])
# breakpoint()

common_subjects = set(brain_df["subject_id"].astype(str)) & set(
    demographics_df["Subject"].astype(str)
)
demographics_filtered = demographics_df[
    demographics_df["Subject"].astype(str).isin(common_subjects)
]
brain_filtered = brain_df[brain_df["subject_id"].astype(str).isin(common_subjects)]

# DATASET GENERATION
X = brain_filtered.drop(columns=["subject_id"]).to_numpy()
y = np.array(demographics_filtered["Gender"])
gender = np.array(demographics_filtered["Gender"])

dataset = CustomDataset(X, y, gender)

# ----------- CROSS VALIDATION + TRAINING + TESTING -----------


preprocessed = PreprocessedData(
    X, y, gender, n_splits=config["n_splits"], random_state=config["random_state"]
)

specs = preprocessed.get_specs()
print(specs)

# folds = preprocessed.get_folds_as_dataloaders(batch_size=16)
indices = preprocessed.get_fold_indices()


def run_single_model(model_name, config, dataset, preprocessed, indices, results_path):
    local_config = copy.deepcopy(config)
    local_config["model_name"] = model_name

    train_scores, test_scores = [], []
    train_preds, test_preds = [], []
    train_targets, test_targets = [], []
    per_fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(indices):
        # print(
        #     f"Fold {fold_idx+1} - Train samples: {len(train_idx)}, test_samples: {len(test_idx)}"
        # )
        train_loader, test_loader = preprocessed.get_dataloader_fold(
            dataset, fold_idx, indices
        )
        _, y_train, _, _, y_test, _ = preprocessed.get_arrays_from_indices(
            dataset, fold_idx, indices
        )

        model = get_model(model_name, local_config)

        # --------- Train / Val / Test Model ---------
        # print("Training...")
        model.fit(train_loader)
        train_pred = model.predict(train_loader)
        train_score = accuracy_score(y_train, train_pred)
        print(train_score)

        train_scores.append(train_score)
        train_preds.append(train_pred.tolist())
        train_targets.append(y_train.tolist())

        # print("Testing...")
        test_pred = model.predict(test_loader)
        test_score = accuracy_score(y_test, test_pred)
        print(test_score)

        test_scores.append(test_score)
        test_preds.append(test_pred.tolist())
        test_targets.append(y_test.tolist())

        # print(f"Done Fold {fold_idx + 1}")
        if DEBUG:
            per_fold_results.append(
                {
                    "model": model_name,
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

    # print("\n Saving results...")
    if DEBUG:
        save_fold_results(
            model_name=model_name,
            fold_results=per_fold_results,
            output_dir=Path(results_path) / "analysis_results",
        )

    return model_name, per_fold_results


models_to_run = config["model_name"]

# Execute models in parallel
results = Parallel(n_jobs=1)(  # len(models_to_run)
    delayed(run_single_model)(
        model_name, config, dataset, preprocessed, indices, config["results_path"]
    )
    for model_name in models_to_run
)

# results is a list of (model_name, per_fold_results)
for model_name, per_fold_results in results:
    print(f"Completed model: {model_name}")


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


###### VERY EARLY IN TESTING
# preparator = LcotEmbedHcpPipeline(config)
# preparator.extract_raw_data("100206")
# preparator.compute_microstructure("100206")
# breakpoint()

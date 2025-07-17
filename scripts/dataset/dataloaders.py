from pathlib import Path

import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

from diff_benchmark.analysis.save_results import save_fold_results, save_summary_results
from diff_benchmark.dataset.generate_dataset import CustomDataset
from diff_benchmark.dataset.read_save_dataset import load_dataset
from diff_benchmark.models.model_configurations import get_model
from diff_benchmark.scores.scores import mse_score

with open(Path(__file__).parent.parent.parent / "configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
DEBUG = config["debugging_analysis"]

# ---------- LOAD DATASET ----------
X, y, gender = load_dataset(Path(config["results_path_2"]) / "datasets" / "dataset.h5")

skf = StratifiedKFold(
    n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"]
)
folds = list(skf.split(X, gender))

X_test = X.mean(-1).mean(-2)
y_test = y[:, 2]
dataset = CustomDataset(X_test, y_test, gender)
# dataset = CustomDataset(X, y, gender)


train_scores, test_scores = [], []
train_preds, test_preds = [], []
train_targets, test_targets = [], []
per_fold_results = []

# --------- BALANCED SPLIT IN TRAIN, VALIDATION AND TEST ---------
for fold_idx, (train_idx, test_idx) in enumerate(folds):
    print(f"\nFold {fold_idx + 1}/{config["n_splits"]}")

    # Subsets for this fold
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    model = get_model(config["model_name"], config)

    # --------- Train / Val / Test Model ---------
    print("Training...")
    model.fit(train_loader)
    train_pred = model.predict(train_loader)
    _, train_tgt = model._dataloader_to_numpy(train_loader)
    train_score = mse_score(train_tgt, train_pred)
    print(train_score)

    train_scores.append(train_score)
    train_preds.append(train_pred.tolist())
    train_targets.append(train_tgt.tolist())

    print("Testing...")
    test_pred = model.predict(test_loader)
    _, test_tgt = model._dataloader_to_numpy(test_loader)
    test_score = mse_score(test_tgt, test_pred)
    print(test_score)

    test_scores.append(test_score)
    test_preds.append(test_pred.tolist())
    test_targets.append(test_tgt.tolist())

    print(f"Done Fold {fold_idx + 1}")

    # --------- SAVE RESULTS ---------
    if DEBUG:
        per_fold_results.append(
            {
                "model": config["model_name"],
                "fold": fold_idx,
                "train": {
                    "score": float(train_score),
                    "predictions": train_pred.tolist(),
                    "targets": train_tgt.tolist(),
                },
                "test": {
                    "score": float(test_score),
                    "predictions": test_pred.tolist(),
                    "targets": test_tgt.tolist(),
                },
            }
        )

print("\n Saving results...")
if DEBUG:
    save_fold_results(
        model_name=config["model_name"],
        fold_results=per_fold_results,
        output_dir=Path(config["results_path_2"]) / "analysis_results",
    )

# save_summary_results(
#     model_name=config["model_name"],
#     train_scores=train_scores,
#     test_scores=test_scores,
#     train_preds=train_preds,
#     test_preds=test_preds,
#     train_targets=train_targets,
#     test_targets=test_targets,
#     output_dir=Path(config["results_path_2"]) / "analysis_results",
# )

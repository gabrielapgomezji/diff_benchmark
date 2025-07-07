from pathlib import Path

import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

from diff_benchmark.dataset.generate_dataset import CustomDataset
from diff_benchmark.dataset.read_save_dataset import load_dataset

with open("configuration.yaml", "r") as file:
    config = yaml.safe_load(file)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD DATASET ----------
X, y, gender = load_dataset(Path(config["results_path"]) / "datasets" / "dataset.h5")
# ##########
# import numpy as np

# num_samples = 1000
# num_features = 20
# X = np.random.randn(num_samples, num_features)
# y = np.random.randint(0, 2, size=num_samples)
# gender = np.random.randint(0, 2, size=num_samples)
# ##########

skf = StratifiedKFold(
    n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"]
)
folds = list(skf.split(X, gender))

dataset = CustomDataset(X, y, gender)


# --------- BALANCED SPLIT IN TRAIN, VALIDATION AND TEST ---------
for fold_idx, (trainval_idx, test_idx) in enumerate(folds):
    print(f"\nFold {fold_idx + 1}/{config["n_splits"]}")

    # Stratified split of trainval into train and val
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=config["val_size"],
        stratify=gender[trainval_idx],
        random_state=config["random_state"],
    )

    # Subsets for this fold
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # ---- Train / Val / Test Model----
    print("Training...")
    for X_batch, y_batch, _ in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        # Insert training step here
        pass

    print("Validating...")
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            # Insert validation step here
            pass

    print("Testing...")
    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            # Insert test step here
            pass

    print(f"Done Fold {fold_idx + 1}")

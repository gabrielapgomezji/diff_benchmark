from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# import pandas as pd
from tqdm import tqdm

from diff_benchmark.dataset.load_data import load_embeddings_and_power_from_h5
from diff_benchmark.dataset.read_save_dataset import save_dataset


class CustomDataset(Dataset):
    def __init__(self, X, y, gender):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.gender[idx]


def build_dataset(
    base_path,
    df_targets,
    h5_filename="mapmri_default_embeddings.h5",  # Model data. (This file is for the computed embeddings)
    output_dataset_filename="dataset.h5",
):
    """
    Builds a dataset by processing subject directories and extracting embeddings and target values.
    Args:
        base_path (Path): Path to the base directory containing subject subdirectories.
        df_targets (pd.DataFrame): Path to the CSV file containing target information for subjects.
        h5_filename (str, optional): Name of the HDF5 file containing embeddings for each subject.
            Defaults to "mapmri_default_embeddings.h5".
        output_dataset_filename (str, optional): Name of the output dataset file. Defaults to "dataset.h5".
    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Array of feature vectors extracted from embeddings.
            - y (numpy.ndarray): Array of target values corresponding to the subjects.
            - subjects_included (list): List of subject IDs included in the dataset.
    Notes:
        - The function filters subjects based on the existence of the HDF5 file and the validity of embeddings.
        - Embeddings are concatenated into feature vectors, and targets are extracted from the provided CSV file.
        - Only subjects with embeddings matching the desired shape are included in the final dataset.
    """
    X = []
    y = []
    subjects_included = []
    genders = []

    for subject_dir in tqdm(Path(base_path).iterdir()):
        if not subject_dir.is_dir():
            continue

        # --------- GET THE INPUT DATA ---------
        subject_id = subject_dir.name
        h5_path = subject_dir / "processed" / h5_filename

        if not h5_path.exists():
            continue

        try:
            embeddings, power, metadata = load_embeddings_and_power_from_h5(h5_path)
            if not is_valid_embedding(embeddings):
                continue
        except Exception as e:
            print(f"Failed to process subject {subject_id}: {e}")
            continue

        # Optional: flatten or format embeddings/power into feature vector
        features = np.concatenate(
            [v for v in embeddings.values()]
        )  # + [power.flatten()])

        # --------- GET THE DEMOGRAPHICS TARGETS ---------
        if int(subject_id) in df_targets["Subject"].astype(int).tolist():
            target = (
                df_targets.loc[df_targets["Subject"] == int(subject_id)]
                .drop(columns=["Subject"])
                .values.astype(float)
            )
            gender_subject = df_targets.loc[df_targets["Subject"] == int(subject_id), "Gender"].values[0]
            X.append(features)
            y.append(target)
            subjects_included.append(subject_id)
            genders.append(gender_subject)

    # --------- FILTER THE DATA ---------
    # Conditions for filtering:
    desired_shape = (3, 536, 10000)
    # Filter out subjects whose embeddings do not match the desired shape
    valid_indices = [i for i, x in enumerate(X) if x.shape == desired_shape]
    # Filter out subjects whose targets are not valid
    X_filtered = [X[i] for i in valid_indices]
    y_filtered = [y[i] for i in valid_indices]
    subjects_filtered = [subjects_included[i] for i in valid_indices]
    genders_filtered = [genders[i] for i in valid_indices]

    X = np.stack(X_filtered)
    # X = np.stack([x.astype(np.float16) for x in X_filtered])
    y = np.stack(y_filtered).squeeze(1)
    subjects_included = subjects_filtered
    breakpoint()
    print("Verify genders_filtered is a list likesubjects_included")
    genders = genders_filtered

    # --------- SAVE DATASET TO AVOID RE-RUNNING ---------
    save_dataset(X, y, genders, output_file=output_dataset_filename)
    
    # --------- SAVE SUBJECTS IN THE DATASET AND ANALYSIS ---------
    return X, y, subjects_included

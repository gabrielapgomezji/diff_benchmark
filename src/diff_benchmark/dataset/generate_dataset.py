from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# import pandas as pd
from tqdm import tqdm

from diff_benchmark.dataset.read_save_dataset import save_dataset
from diff_benchmark.dataset.base import DatasetBuilder

from joblib import Parallel, delayed


# class CustomDataset(Dataset):

#     def __init__(self, list_path_subjects):
#         self.list_subjects = list_path_subjects

#     def __len__(self):
#         return len(self.list_subjects)
    
#     def __getitem__(self, idx):
#         # load self.list_path_subjects[idx]



# When you define a torch dataset; you have two options
# 1. You build a tensor that contains all your data, and then you simply return data[idx] when you need index idx -> method you're using right now
# Advantages: the data is already loaded into memory so it's fast, and it's simple
# Inconvenients: when the data is too heavy for the memory (on ram), can't be done
# 2. When you're asked to load index idx, you load the corresponding file into memory and you return it
# Advantage: very low memory consumption
# Inconvenient: slow because you need to load data everytime you need it


class PreprocessedData:

    def __init__(self, config):

        
        if config["mode"] == "all":
            self.X, self.y = self.build_dataset()
            self.mode = "all"



    def build_dataset(self):
        pass
    def get_folds_as_dataloaders(self):
        pass
    


class CustomDataset(Dataset):
    def __init__(self, X, y, gender):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.gender[idx]


class CustomDatasetBuilder(DatasetBuilder):
    def __init__(self, base_path, loading_strategy, df_targets, h5_filename="mapmri_default_embeddings.h5", output_dataset_filename="dataset.h5"):
        super().__init__(base_path, h5_filename, output_dataset_filename)
        self.df_targets = df_targets
        self.strategy = loading_strategy

    def verify_files(self, subject_dir: Path) -> bool:
        h5_path = subject_dir / "processed" / self.h5_filename
        
        return h5_path.exists() and int(subject_dir.name) in self.df_targets["Subject"].astype(int).tolist()

    def filter_features(self, features) -> bool:
        return features.shape[1] == 536
    
    def extract_information(self, subject_dir: Path):
        subject_id = subject_dir.name
        h5_path = subject_dir / "processed" / self.h5_filename

        try:
            data = self.strategy.load_data(h5_path)
            if not self.strategy.is_valid(data):
                return None
        except Exception as e:
            print(f"Failed to load data for subject {subject_id}: {e}")
            return None

        features = self.strategy.to_features(data)

        if int(subject_id) not in self.df_targets["Subject"].astype(int).tolist():
            return None

        target = (
            self.df_targets.loc[self.df_targets["Subject"] == int(subject_id)]
            .drop(columns=["Subject"])
            .values.astype(float)
        )

        gender_subject = self.df_targets.loc[
            self.df_targets["Subject"] == int(subject_id), "Gender"
        ].values[0]

        return features, target, subject_id, gender_subject

    def save_dataset(self, X, y, genders):
        save_dataset(X, y, genders, output_file=self.output_dataset_filename)

    def create_dataset(self, n_jobs=8):
        subject_dirs = [d for d in self.base_path.iterdir() if d.is_dir() and self.verify_files(d)]

        print(f"Processing {len(subject_dirs)} subjects in parallel...")

        def process(subject_dir):
            result = self.extract_information(subject_dir)
            if result is None:
                return None
            features, target, subject_id, gender = result
            if not self.filter_features(features):
                return None
            return features, target, subject_id, gender

        results = Parallel(n_jobs=n_jobs)(
            delayed(process)(d) for d in tqdm(subject_dirs)
        )

        # Filter out failed entries
        results = [r for r in results if r is not None]

        if not results:
            print("No valid data found.")
            return None

        features_list, targets_list, subject_ids, genders = zip(*results)

        try:
            X = np.stack(features_list)
            y = np.stack(targets_list).squeeze(1) if targets_list[0].ndim == 2 and targets_list[0].shape[1] == 1 else np.stack(targets_list)
        except Exception as e:
            print(f"Error stacking arrays: {e}")
            return None

        self.save_dataset(X, y, genders)
        return X, y, subject_ids, genders
    
    # def create_dataset(self):
    #     X, y, subjects, genders = [], [], [], []

    #     for subject_dir in tqdm(self.base_path.iterdir()):
    #         if not subject_dir.is_dir():
    #             continue
    #         if not self.verify_files(subject_dir):
    #             continue

    #         result = self.extract_information(subject_dir)
    #         if result is None:
    #             continue
            
    #         features, target, subject_id, gender = result
    #         if not self.filter_features(features):
    #             continue

    #         X.append(features)
    #         y.append(target)
    #         subjects.append(subject_id)
    #         genders.append(gender)

    #     # Optional: add filtering logic
    #     X = np.stack(X)
    #     y = np.stack(y).squeeze(1)

    #     self.save_dataset(X, y, genders)
    #     return X, y, subjects, genders

# def build_dataset(
#     base_path,
#     df_targets,
#     h5_filename="mapmri_default_embeddings.h5",  # Model data. (This file is for the computed embeddings)
#     output_dataset_filename="dataset.h5",
# ):
#     """
#     Builds a dataset by processing subject directories and extracting embeddings and target values.
#     Args:
#         base_path (Path): Path to the base directory containing subject subdirectories.
#         df_targets (pd.DataFrame): Path to the CSV file containing target information for subjects.
#         h5_filename (str, optional): Name of the HDF5 file containing embeddings for each subject.
#             Defaults to "mapmri_default_embeddings.h5".
#         output_dataset_filename (str, optional): Name of the output dataset file. Defaults to "dataset.h5".
#     Returns:
#         tuple: A tuple containing:
#             - X (numpy.ndarray): Array of feature vectors extracted from embeddings.
#             - y (numpy.ndarray): Array of target values corresponding to the subjects.
#             - subjects_included (list): List of subject IDs included in the dataset.
#     Notes:
#         - The function filters subjects based on the existence of the HDF5 file and the validity of embeddings.
#         - Embeddings are concatenated into feature vectors, and targets are extracted from the provided CSV file.
#         - Only subjects with embeddings matching the desired shape are included in the final dataset.
#     """
#     X = []
#     y = []
#     subjects_included = []
#     genders = []

#     for subject_dir in tqdm(Path(base_path).iterdir()):
#         if not subject_dir.is_dir():
#             continue

#         # --------- GET THE INPUT DATA ---------
#         subject_id = subject_dir.name
#         h5_path = subject_dir / "processed" / h5_filename

#         if not h5_path.exists():
#             continue

#         try:
#             embeddings, power, metadata = load_embeddings_and_power_from_h5(h5_path)
#             if not is_valid_embedding(embeddings):
#                 continue
#         except Exception as e:
#             print(f"Failed to process subject {subject_id}: {e}")
#             continue

#         # Optional: flatten or format embeddings/power into feature vector
#         features = np.concatenate(
#             [v for v in embeddings.values()]
#         )  # + [power.flatten()])

#         # --------- GET THE DEMOGRAPHICS TARGETS ---------
#         if int(subject_id) in df_targets["Subject"].astype(int).tolist():
#             target = (
#                 df_targets.loc[df_targets["Subject"] == int(subject_id)]
#                 .drop(columns=["Subject"])
#                 .values.astype(float)
#             )
#             gender_subject = df_targets.loc[
#                 df_targets["Subject"] == int(subject_id), "Gender"
#             ].values[0]
#             X.append(features)
#             y.append(target)
#             subjects_included.append(subject_id)
#             genders.append(gender_subject)

#     # --------- FILTER THE DATA ---------
#     # Conditions for filtering:
#     desired_shape = (3, 536, 10000)
#     # Filter out subjects whose embeddings do not match the desired shape
#     valid_indices = [i for i, x in enumerate(X) if x.shape == desired_shape]
#     # Filter out subjects whose targets are not valid
#     X_filtered = [X[i] for i in valid_indices]
#     y_filtered = [y[i] for i in valid_indices]
#     subjects_filtered = [subjects_included[i] for i in valid_indices]
#     genders_filtered = [genders[i] for i in valid_indices]

#     X = np.stack(X_filtered)
#     # X = np.stack([x.astype(np.float16) for x in X_filtered])
#     y = np.stack(y_filtered).squeeze(1)
#     subjects_included = subjects_filtered
#     genders = genders_filtered

#     # --------- SAVE DATASET TO AVOID RE-RUNNING ---------
#     save_dataset(X, y, genders, output_file=output_dataset_filename)

#     # --------- SAVE SUBJECTS IN THE DATASET AND ANALYSIS ---------
#     return X, y, subjects_included

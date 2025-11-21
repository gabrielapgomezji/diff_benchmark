from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch

# from joblib import Parallel, delayed
from torch.utils.data import Dataset
from diff_benchmark.dataset.utils_dataset import load_precomputed_coordinates

# from tqdm import tqdm

# from diff_benchmark.dataset.base import DatasetBuilder
# from diff_benchmark.dataset.read_save_dataset import save_dataset

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
    """
    PreprocessedData is a class for handling and processing datasets for machine learning tasks.
    Attributes:
        X (any): The features of the dataset.
        y (any): The labels of the dataset.
        mode (str): The mode of dataset processing, e.g., "all".
    Methods:
        build_dataset(): Constructs the dataset based on the provided configuration.
        get_folds_as_dataloaders(): Retrieves the dataset folds as dataloaders for training and validation.
    """

    def __init__(self, config):

        if config["mode"] == "all":
            self.features, self.targets = self.build_dataset()
            self.mode = "all"

    def build_dataset(self):
        """
        Builds the dataset for the project.
        This method is responsible for generating and preparing the dataset
        needed for the benchmarking process. It may involve loading data,
        processing it, and saving it in the required format.
        Currently, this method is not implemented.
        """

    def get_folds_as_dataloaders(self):
        """
        Retrieves the dataset folds as PyTorch DataLoader instances.
        This method is intended to be implemented to generate and return
        DataLoader objects for each fold of the dataset, which can be used
        for training and validation in a machine learning context.
        Returns:
            List[DataLoader]: A list of DataLoader instances, each corresponding
            to a different fold of the dataset.
        """


class CustomDataset(Dataset):
    """
    Initializes the dataset object with features, labels, and gender information.
    Parameters:
        X (array-like): The input features for the dataset.
        y (array-like): The target labels for the dataset.
        gender (array-like): The gender information associated with each sample.
    Attributes:
        X (torch.Tensor): A tensor representation of the input features.
        y (torch.Tensor): A tensor representation of the target labels.
        gender (torch.Tensor): A tensor representation of the gender information.
    """

    def __init__(self, features, targets, gender, transform=None):
        # self.features = torch.tensor(features, dtype=torch.float32)
        self.features = features.drop(columns=["subject_id"])
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.int64)
        self.transform = transform

        self.mode = self.get_features_model()

        if self.mode == "features":
            self.features = self.features.to_numpy()
            self.features = torch.tensor(self.features, dtype=torch.float32)
        if self.mode == "paths":
            self.features = self.features[0].tolist()

    def __len__(self):
        """
        Returns the number of elements in the dataset.
        This method overrides the built-in __len__ method to provide the length
        of the dataset, which is determined by the number of samples in the
        attribute `self.X`.
        Returns:
            int: The number of samples in the dataset.
        """

        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample from the dataset.
        Args:
            idx (int): The index of the data sample to retrieve.
        Returns:
            tuple: A tuple containing the features (self.X[idx]),
                   the target variable (self.y[idx]),
                   and the gender information (self.gender[idx])
                   corresponding to the specified index.
        """
        if self.mode == "features":
            final_features = self.features[idx]
        if self.mode == "paths":
            try:
                if Path(self.features[idx]).name.endswith("_lcotembedding.h5"):
                    final_features = self._load_h5(Path(self.features[idx]))
                if Path(self.features[idx]).name.endswith("_spheres.h5"):
                    final_features = self._load_h5_spheres(Path(self.features[idx]))
                else:
                    img = nib.load(Path(self.features[idx]))
                    # target_affine = np.diag([1.25, 1.25, 1.25, 1.0])
                    # target_shape = (180, 224, 224)
                    # resampled_img = resample_img(img, target_affine=target_affine, target_shape=target_shape, interpolation='continuous', copy_header=True, force_resample=True)
                    resampled_img = img
                    data = np.nan_to_num(resampled_img.get_fdata()).clip(0, 7)
                    data /= 7.0
                    final_features = torch.tensor(data, dtype=torch.float32)
                    # features = nib.Nifti1Image(data, affine=img.affine)
                    if self.transform is not None:
                        slices = []
                        for i in range(
                            final_features.shape[0]
                        ):  # iterate through depth dimension
                            slice_2d = final_features[
                                i, :, :
                            ]  # .unsqueeze(0)  # (1,H,W)
                            slice_2d = self.transform(slice_2d)
                            slices.append(slice_2d)
                        final_features = torch.stack(slices, dim=0)  # (D,1,H,W)
                        final_features = final_features.permute(
                            1, 0, 2, 3
                        )  # (C=1,D,H,W)
                        # final_features = final_features.squeeze(0)  # (D,H,W)
            except (OSError, FileNotFoundError) as e:
                print(f"[Warning] Dropping subject {Path(self.features[idx])}: {e}")
                return None

        return final_features, self.targets[idx], self.gender[idx]

    def _load_h5(self, path):
        """
        Load the HDF5 file and convert it to a suitable tensor.
        Combines all 'attenuation' datasets from all vertices and bvals into a single array.
        """
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        with h5py.File(path, "r") as f:
            # Load metadata
            meta = f["metadata"]
            bvals = list(meta.attrs["bvals"])
            # Load embeddings
            emb_grp = f["embeddings"]
            embeddings_per_bval = []

            for bval in bvals:
                bval_str = str(bval)
                if bval_str not in emb_grp:
                    raise KeyError(
                        f"Missing embeddings for bval {bval_str} in file {path}"
                    )
                data = emb_grp[bval_str][:]
                embeddings_per_bval.append(data)

            # Stack into a single numpy array
            # embeddings_per_bval: list of [num_values, emb_dim] arrays
            data_array = np.stack(
                embeddings_per_bval, axis=1
            )  # shape: (num_values, num_bvals, emb_dim)

            power = f["power"][:]
        breakpoint()
        return {
            "embeddings": torch.tensor(data_array, dtype=torch.float32),
            "power": torch.tensor(power, dtype=torch.float32),
        }

    def _load_h5_spheres(self, path):
        """
        Load an HDF5 file where embeddings are stored per-bval and per-vertex:
        f["embeddings"]["1000"][vertex_id]["attenuation"], f["embeddings"]["2000"][...], ...
        Returns dict with:
            "attenuations": torch.tensor shape (num_vertices, num_bvals, att_dim)
            "power": torch.tensor shape (num_vertices, num_bvals)  # sum of attenuation**2 across att_dim
            "vertices": list of vertex ids (strings) in the order used
        Behavior:
            - Uses the intersection of vertices present in all bvals to ensure consistent ordering.
            - Pads attenuation vectors with NaN when lengths differ, so sum uses nan-safe reduction.
        """
        try:
            coords_L, coords_R = load_precomputed_coordinates()
        except FileNotFoundError as e:
            exit(f"[Error] Cannot load precomputed coordinates: {e}")
            
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        merged_coords = {**coords_L, **coords_R}
        with h5py.File(path, "r") as f:
            candidate_bvals = [str(k) for k in f.keys() if isinstance(f[k], (h5py.Group,))]

            if not candidate_bvals:
                raise KeyError(f"No b-value groups found in file {path}")

            # sort bvals numerically when possible (keeps deterministic order)
            try:
                bvals = sorted(candidate_bvals, key=lambda x: int(x))
            except Exception:
                bvals = sorted(candidate_bvals)

            # collect vertex id sets per bval
            vertex_sets = []
            for bstr in bvals:
                grp = f[bstr]
                # convert keys to str (h5py may return bytes)
                vertex_sets.append({str(k) for k in grp.keys()})

            # use intersection to ensure consistent vertices across bvals
            common_vertices = sorted(
                set.intersection(*vertex_sets),
                key=lambda x: int(x) if x.isdigit() else x,
            )

            if len(common_vertices) == 0:
                raise KeyError(f"No common vertices across bvals in {path}")

            # determine maximum attenuation vector length across chosen vertices and bvals
            max_len = 0
            for bstr in bvals:
                grp = f[bstr]
                for v in common_vertices:
                    node = grp.get(v)
                    if node is None or "attenuation" not in node:
                        raise KeyError(f"Missing attenuation for vertex {v} under bval {bstr} in {path}")
                    arr = np.asarray(node["attenuation"][:]).reshape(-1)
                    if arr.size > max_len:
                        max_len = arr.size

            num_vertices = len(common_vertices)
            num_bvals = len(bvals)
            att_array = np.full((num_vertices, num_bvals, max_len), np.nan, dtype=np.float32)

            # fill attenuation array
            for i, bstr in enumerate(bvals):
                grp = f[bstr]
                for j, v in enumerate(common_vertices):
                    node = grp.get(v)
                    # previous checks guarantee node exists and has 'attenuation'
                    arr = np.asarray(node["attenuation"][:]).reshape(-1).astype(np.float32)
                    att_array[j, i, : arr.size] = arr

            # compute power per vertex and bval: sum of squared attenuation across att_dim
            power = np.nansum(att_array * att_array, axis=2).astype(np.float32)
  
        coords_common = {v: merged_coords[v] for v in common_vertices if v in merged_coords}
        if len(coords_common) != len(common_vertices):
            missing = set(common_vertices) - set(coords_common.keys())
            raise KeyError(
                f"Coordinate files do not contain coordinates for these vertices: {sorted(missing)}"
            )
        return {
            "attenuations": torch.tensor(att_array, dtype=torch.float32),
            "power": torch.tensor(power, dtype=torch.float32),
            # "vertices": common_vertices,
            "coords": coords_common, 
            # "bvals": bvals,
        }

    def get_features_model(self):
        """
        Determines the mode of the features based on their data type.
        This method checks if the first column of the features DataFrame, excluding
        the 'subject_id' column, is of a numeric subtype. If it is numeric, the mode
        is set to "features"; otherwise, it is set to "paths".
        Returns:
            str: The mode of the features, either "features" or "paths".
        """

        if np.issubdtype(self.features.dtypes[0], np.number):
            self.mode = "features"
        else:
            self.mode = "paths"
        return self.mode


# class CustomDatasetBuilder(DatasetBuilder):
#     """
#     Initializes the dataset generator.
#     Parameters:
#         base_path (str): The base path for the dataset.
#         loading_strategy (str): The strategy to use for loading the dataset.
#         df_targets (DataFrame): The DataFrame containing target values.
#         h5_filename (str, optional): The name of the HDF5 file for embeddings. Defaults to "mapmri_default_embeddings.h5".
#         output_dataset_filename (str, optional): The name of the output dataset file. Defaults to "dataset.h5".
#     """

#     def __init__(
#         self,
#         base_path,
#         loading_strategy,
#         df_targets,
#         h5_filename="mapmri_default_embeddings.h5",
#         output_dataset_filename="dataset.h5",
#     ):
#         super().__init__(base_path, h5_filename, output_dataset_filename)
#         self.df_targets = df_targets
#         self.strategy = loading_strategy

#     def verify_files(self, subject_dir: Path) -> bool:
#         """
#         Verifies the existence of a specific HDF5 file and checks if the subject directory name is present in the DataFrame of targets.
#         Args:
#             subject_dir (Path): The directory path of the subject to verify.
#         Returns:
#             bool: True if the HDF5 file exists and the subject directory name is in the DataFrame, False otherwise.
#         """

#         h5_path = subject_dir / "processed" / self.h5_filename

#         return (
#             h5_path.exists()
#             and int(subject_dir.name) in self.df_targets["Subject"].astype(int).tolist()
#         )

#     def filter_features(self, features: np.array) -> bool:
#         """
#         Filters features based on their shape.
#         This method checks if the number of columns in the features array
#         is equal to 536. It returns True if the condition is met,
#         indicating that the features are valid, and False otherwise.
#         Args:
#             features (numpy.ndarray): The input features to be checked.
#         Returns:
#             bool: True if the number of columns is 536, False otherwise.
#         """

#         return features.shape[1] == 536

#     def extract_information(self, subject_dir: Path) -> tuple:
#         """
#         Extracts information from the dataset for a given subject directory.
#         Args:
#             subject_dir (Path): The path to the subject directory containing the dataset.
#         Returns:
#             tuple: A tuple containing the extracted features, target values, subject ID, and gender of the subject.
#                    Returns None if data loading fails, if the data is invalid, or if the subject ID is not found in the targets.
#         Raises:
#             Exception: If there is an error loading the data from the specified h5_path.
#         """

#         subject_id = subject_dir.name
#         h5_path = subject_dir / "processed" / self.h5_filename

#         try:
#             data = self.strategy.load_data(h5_path)
#             if not self.strategy.is_valid(data):
#                 return None
#         except Exception as e:
#             print(f"Failed to load data for subject {subject_id}: {e}")
#             return None

#         features = self.strategy.to_features(data)

#         if int(subject_id) not in self.df_targets["Subject"].astype(int).tolist():
#             return None

#         target = (
#             self.df_targets.loc[self.df_targets["Subject"] == int(subject_id)]
#             .drop(columns=["Subject"])
#             .values.astype(float)
#         )

#         gender_subject = self.df_targets.loc[
#             self.df_targets["Subject"] == int(subject_id), "Gender"
#         ].values[0]

#         return features, target, subject_id, gender_subject

#     def save_dataset(self, features, targets, genders) -> None:
#         """
#         Saves the dataset to a specified output file.
#         Parameters:
#             features (array-like): The input features of the dataset.
#             targets (array-like): The target labels corresponding to the input features.
#             genders (array-like): The gender information associated with the dataset.
#         This method calls the save_dataset function to write the dataset to the
#         output file defined by self.output_dataset_filename.
#         """

#         save_dataset(
#             features, targets, genders, output_file=self.output_dataset_filename
#         )

#     def create_dataset(self, n_jobs: int = 8) -> tuple:
#         """
#         Creates a dataset by processing subject directories in parallel.
#         This method iterates through subject directories in the base path, extracts
#         features and targets from the files, filters the features, and stacks the
#         results into arrays. It utilizes parallel processing to speed up the
#         information extraction.
#         Parameters:
#             n_jobs (int): The number of jobs to run in parallel. Default is 8.
#         Returns:
#             tuple: A tuple containing:
#                 - features (np.ndarray): Stacked features array.
#                 - targets (np.ndarray): Stacked targets array.
#                 - subject_ids (list): List of subject IDs.
#                 - genders (list): List of genders.
#         Raises:
#             Exception: If there is an error stacking the arrays.
#             None: If no valid data is found or if an error occurs during processing.
#         """

#         subject_dirs = [
#             d for d in self.base_path.iterdir() if d.is_dir() and self.verify_files(d)
#         ]

#         print(f"Processing {len(subject_dirs)} subjects in parallel...")

#         def process(subject_dir):
#             result = self.extract_information(subject_dir)
#             if result is None:
#                 return None
#             features, target, subject_id, gender = result
#             if not self.filter_features(features):
#                 return None
#             return features, target, subject_id, gender

#         results = Parallel(n_jobs=n_jobs)(
#             delayed(process)(d) for d in tqdm(subject_dirs)
#         )

#         # Filter out failed entries
#         results = [r for r in results if r is not None]

#         if not results:
#             print("No valid data found.")
#             return None

#         features_list, targets_list, subject_ids, genders = zip(*results)

#         try:
#             features = np.stack(features_list)
#             targets = (
#                 np.stack(targets_list).squeeze(1)
#                 if targets_list[0].ndim == 2 and targets_list[0].shape[1] == 1
#                 else np.stack(targets_list)
#             )
#         except Exception as e:
#             print(f"Error stacking arrays: {e}")
#             return None

#         self.save_dataset(features, targets, genders)
#         return features, targets, subject_ids, genders

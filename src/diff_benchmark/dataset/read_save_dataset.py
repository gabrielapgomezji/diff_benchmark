import os

import h5py


def save_dataset(features, targets, gender, output_file="dataset.h5"):
    """
    Saves the dataset to an HDF5 file.
    Parameters:
        features (array-like): The input features to be saved.
        targets (array-like): The target labels to be saved.
        gender (array-like): The gender information to be saved.
        output_file (str, optional): The path to the output HDF5 file. Defaults to "dataset.h5".
    Raises:
        OSError: If the output file cannot be created or written to.
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("targets", data=targets)
        f.create_dataset("gender", data=gender)


def load_dataset(dataset_filename):
    """
    Load a dataset from an HDF5 file.
    Parameters:
        dataset_filename (str): The path to the HDF5 file containing the dataset.
    Returns:
        tuple: A tuple containing three elements:
            - features (numpy.ndarray): The input features from the dataset.
            - targets (numpy.ndarray): The labels corresponding to the input features.
            - gender (numpy.ndarray): The gender information associated with the dataset.
    """

    with h5py.File(dataset_filename, "r") as f:
        features = f["features"][:]
        targets = f["targets"][:]
        gender = f["gender"][:]
    return features, targets, gender

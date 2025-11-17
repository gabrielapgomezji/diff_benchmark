from pathlib import Path

import numpy as np

from diff_benchmark.dataset.base import DataLoadingStrategy
from diff_benchmark.dataset.load_data import (
    load_attenuation_from_h5,
    load_embeddings_and_power_from_h5,
)
from diff_benchmark.dataset.utils_dataset import is_valid_embedding


class MapMRIEmbeddingStrategy(DataLoadingStrategy):
    """
    MapMRIEmbeddingStrategy is a subclass of DataLoadingStrategy that handles the loading and processing of MRI embeddings.
    Methods:
        load_data(h5_path: Path) -> dict:
            Loads embeddings, power, and metadata from the specified HDF5 file path.
        is_valid(data: dict) -> bool:
            Validates the loaded data by checking the embeddings.
        to_features(data: dict) -> np.ndarray:
            Converts the loaded embeddings into a NumPy array of features by concatenating the values.
    """

    def load_data(self, h5_path: Path):
        embeddings, power, metadata = load_embeddings_and_power_from_h5(h5_path)
        return {"embeddings": embeddings, "power": power, "metadata": metadata}

    def is_valid(self, data: dict) -> bool:
        return is_valid_embedding(data["embeddings"])

    def to_features(self, data: dict) -> np.ndarray:
        # return np.concatenate([v for v in data["embeddings"].values()])
        # The line above works. Below is to fix a pylint error
        return np.concatenate(data["embeddings"].values())


class AttenuationStrategy(DataLoadingStrategy):
    """
    AttenuationStrategy is a subclass of DataLoadingStrategy that handles the loading,
    validation, and conversion of attenuation data from a specified HDF5 file.
    Methods:
        load_data(h5_path: Path) -> dict:
            Loads attenuation data and metadata from the specified HDF5 file path.
        is_valid(data: dict) -> bool:
            Validates the loaded data by checking if any of the attenuation values have a
            non-zero shape.
        to_features(data: dict) -> np.ndarray:
            Converts the loaded attenuation data into a NumPy array of features.
    """

    def load_data(self, h5_path: Path):
        atten_data, meta = load_attenuation_from_h5(h5_path)
        return {"attenuation": atten_data, "metadata": meta}

    def is_valid(self, data: dict) -> bool:
        return any(v.shape[0] > 0 for v in data["attenuation"].values())

    def to_features(self, data: dict) -> np.ndarray:
        # return np.stack([v for v in data["attenuation"].values()])
        # The line above works. Below is to fix a pylint error
        return np.stack(data["attenuation"].values())

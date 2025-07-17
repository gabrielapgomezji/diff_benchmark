from pathlib import Path
from diff_benchmark.dataset.load_data import load_embeddings_and_power_from_h5, load_attenuation_from_h5
import numpy as np
from diff_benchmark.dataset.utils_dataset import is_valid_embedding
from diff_benchmark.dataset.base import DataLoadingStrategy

class MapMRIEmbeddingStrategy(DataLoadingStrategy):
    def load_data(self, h5_path: Path):
        embeddings, power, metadata = load_embeddings_and_power_from_h5(h5_path)
        return {"embeddings": embeddings, "power": power, "metadata": metadata}

    def is_valid(self, data: dict) -> bool:
        return is_valid_embedding(data["embeddings"])

    def to_features(self, data: dict) -> np.ndarray:
        return np.concatenate([v for v in data["embeddings"].values()])

class AttenuationStrategy(DataLoadingStrategy):
    def load_data(self, h5_path: Path):
        atten_data, meta = load_attenuation_from_h5(h5_path)
        return {"attenuation": atten_data, "metadata": meta}

    def is_valid(self, data: dict) -> bool:
        return any(v.shape[0] > 0 for v in data["attenuation"].values())

    def to_features(self, data: dict) -> np.ndarray:
        return np.stack([v for v in data["attenuation"].values()])

    
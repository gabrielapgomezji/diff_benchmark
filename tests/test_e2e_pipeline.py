"""
Minimal end-to-end structural integrity test for the diff_benchmark ML pipeline.

Goals
-----
- Verify that all pipeline pieces (PreprocessedData, CustomDataset, SklearnTrainer,
  compute_metrics) can be composed without error on a tiny synthetic dataset.
- Assert that result dictionaries contain the expected metric keys.
- No production code is modified; no real data or external resources are required.
- Fixed seeds everywhere so results are deterministic.

NOT a performance test – metric values are not asserted.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from diff_benchmark.data.dataloaders import PreprocessedData
from diff_benchmark.data.generate_dataset import CustomDataset
from diff_benchmark.models.sklearn_models.dummy import (
    DummyClassifierModel,
    DummyRegressorModel,
)
from diff_benchmark.models.utils_models.trainer import SklearnTrainer
from diff_benchmark.utils.scores import compute_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 20
N_FEATURES = 50
N_SPLITS = 2


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return np.random.RandomState(SEED)


@pytest.fixture(scope="module")
def synthetic_features_df(rng):
    """DataFrame with subject_id + N_FEATURES numeric columns (20 samples)."""
    subject_ids = [f"sub_{i:03d}" for i in range(N_SAMPLES)]
    data = rng.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
    df = pd.DataFrame(data, columns=[str(c) for c in range(N_FEATURES)])
    df.insert(0, "subject_id", subject_ids)
    return df


@pytest.fixture(scope="module")
def synthetic_regression_targets(rng):
    """Continuous targets (age-like) – used for regression tests."""
    return rng.uniform(20.0, 80.0, size=N_SAMPLES).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_binary_targets(rng):
    """Balanced binary labels – used for classification tests."""
    labels = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2), dtype=np.float32)
    rng.shuffle(labels)
    return labels


@pytest.fixture(scope="module")
def gender_array(rng):
    """Gender labels (0/1) for stratification inside PreprocessedData."""
    genders = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2), dtype=np.int64)
    rng.shuffle(genders)
    return genders


def _make_minimal_config(n_splits: int = N_SPLITS, seed: int = SEED) -> OmegaConf:
    """Build the minimal OmegaConf config required by PreprocessedData."""
    return OmegaConf.create(
        {
            "data": {
                "data_partition": {
                    "n_splits": n_splits,
                    "train_size": 1.0,
                    "val_size": 0.1,
                }
            },
            "random_state": seed,
        }
    )


# ---------------------------------------------------------------------------
# 1. CustomDataset
# ---------------------------------------------------------------------------


class TestCustomDataset:
    def test_creates_without_error(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        ds = CustomDataset(
            features=synthetic_features_df.copy(),
            targets=synthetic_regression_targets,
            gender=gender_array,
        )
        assert len(ds) == N_SAMPLES

    def test_getitem_returns_three_tensors(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        ds = CustomDataset(
            features=synthetic_features_df.copy(),
            targets=synthetic_regression_targets,
            gender=gender_array,
        )
        features, target, gender = ds[0]
        assert isinstance(features, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(gender, torch.Tensor)

    def test_feature_shape(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        ds = CustomDataset(
            features=synthetic_features_df.copy(),
            targets=synthetic_regression_targets,
            gender=gender_array,
        )
        features, _, _ = ds[0]
        assert features.shape == (N_FEATURES,)

    def test_subject_ids_stored(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        ds = CustomDataset(
            features=synthetic_features_df.copy(),
            targets=synthetic_regression_targets,
            gender=gender_array,
        )
        assert hasattr(ds, "_subject_ids")
        assert len(ds._subject_ids) == N_SAMPLES


# ---------------------------------------------------------------------------
# 2. PreprocessedData + fold splitting
# ---------------------------------------------------------------------------


class TestPreprocessedData:
    def test_creates_without_error(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        cfg = _make_minimal_config()
        features_np = synthetic_features_df.drop(columns=["subject_id"]).to_numpy(
            dtype=np.float32
        )
        _ = PreprocessedData(
            features=features_np,
            targets=synthetic_regression_targets,
            genders=gender_array,
            config=cfg,
        )

    def test_fold_count(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        cfg = _make_minimal_config()
        features_np = synthetic_features_df.drop(columns=["subject_id"]).to_numpy(
            dtype=np.float32
        )
        preprocessed = PreprocessedData(
            features=features_np,
            targets=synthetic_regression_targets,
            genders=gender_array,
            config=cfg,
        )
        indices = preprocessed.get_fold_indices()
        assert len(indices) == N_SPLITS

    def test_train_test_disjoint(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        cfg = _make_minimal_config()
        features_np = synthetic_features_df.drop(columns=["subject_id"]).to_numpy(
            dtype=np.float32
        )
        preprocessed = PreprocessedData(
            features=features_np,
            targets=synthetic_regression_targets,
            genders=gender_array,
            config=cfg,
        )
        for train_idx, test_idx in preprocessed.get_fold_indices():
            assert (
                len(set(train_idx) & set(test_idx)) == 0
            ), "Train/test overlap detected"

    def test_get_specs_keys(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        cfg = _make_minimal_config()
        features_np = synthetic_features_df.drop(columns=["subject_id"]).to_numpy(
            dtype=np.float32
        )
        preprocessed = PreprocessedData(
            features=features_np,
            targets=synthetic_regression_targets,
            genders=gender_array,
            config=cfg,
        )
        specs = preprocessed.get_specs()
        assert specs.num_samples == N_SAMPLES
        assert specs.num_features == N_FEATURES


# ---------------------------------------------------------------------------
# 3. compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_regression_keys(self, rng):
        y_true = rng.uniform(20, 80, 20).astype(np.float32)
        y_pred = y_true + rng.normal(0, 5, 20).astype(np.float32)
        result = compute_metrics(y_true, y_pred, prediction_task="regression")
        for key in (
            "r2",
            "mae",
            "rmse",
            "mape",
            "explained_variance",
            "pearson_correlation",
        ):
            assert key in result, f"Missing regression metric key: {key!r}"

    def test_classification_keys(self, rng):
        y_true = np.array([0] * 10 + [1] * 10, dtype=np.float32)
        y_pred = y_true.copy()
        y_pred[:3] = 1 - y_pred[:3]  # introduce a few errors
        result = compute_metrics(
            y_true, y_pred, prediction_task="binary_classification"
        )
        for key in ("accuracy", "accuracy_weighted", "precision", "recall", "f1"):
            assert key in result, f"Missing classification metric key: {key!r}"

    def test_invalid_task_raises(self, rng):
        y = rng.randn(10)
        with pytest.raises(ValueError):
            compute_metrics(y, y, prediction_task="unknown_task")


# ---------------------------------------------------------------------------
# 4. Full end-to-end: preprocessing → fold split → train → predict → metrics
# ---------------------------------------------------------------------------


def _build_dataset_and_preprocessed(features_df, targets, gender_array):
    """Helper: build a CustomDataset and PreprocessedData from raw arrays."""
    dataset = CustomDataset(
        features=features_df.copy(),
        targets=targets,
        gender=gender_array,
    )
    cfg = _make_minimal_config()
    features_np = features_df.drop(columns=["subject_id"]).to_numpy(dtype=np.float32)
    preprocessed = PreprocessedData(
        features=features_np,
        targets=targets,
        genders=gender_array,
        config=cfg,
    )
    return dataset, preprocessed


class TestEndToEndRegression:
    """Run the full loop: data → folds → DummyRegressor → metrics."""

    def test_runs_without_error(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        dataset, preprocessed = _build_dataset_and_preprocessed(
            synthetic_features_df, synthetic_regression_targets, gender_array
        )
        indices = preprocessed.get_fold_indices()

        for fold_idx, (train_idx, test_idx) in enumerate(indices):
            train_loader, test_loader = preprocessed.get_dataloader_fold(
                dataset, fold_idx, indices, num_workers=0, batch_size=16
            )
            model = DummyRegressorModel()
            trainer = SklearnTrainer(model=model)
            trainer.set_fold(fold_idx)

            trainer.fit(train_loader)
            y_pred = trainer.predict(test_loader)

            y_test = synthetic_regression_targets[test_idx].squeeze()
            metrics = compute_metrics(y_test, y_pred, prediction_task="regression")

            assert isinstance(metrics, dict)
            for key in ("r2", "mae", "rmse", "mape", "explained_variance"):
                assert key in metrics, f"Fold {fold_idx}: missing key {key!r}"

    def test_prediction_shape(
        self, synthetic_features_df, synthetic_regression_targets, gender_array
    ):
        dataset, preprocessed = _build_dataset_and_preprocessed(
            synthetic_features_df, synthetic_regression_targets, gender_array
        )
        indices = preprocessed.get_fold_indices()
        train_loader, test_loader = preprocessed.get_dataloader_fold(
            dataset, 0, indices, num_workers=0, batch_size=16
        )
        model = DummyRegressorModel()
        trainer = SklearnTrainer(model=model)
        trainer.fit(train_loader)
        y_pred = trainer.predict(test_loader)

        _, test_idx = indices[0]
        assert y_pred.shape[0] == len(test_idx)


class TestEndToEndClassification:
    """Run the full loop: data → folds → DummyClassifier → metrics."""

    def test_runs_without_error(
        self, synthetic_features_df, synthetic_binary_targets, gender_array
    ):
        dataset, preprocessed = _build_dataset_and_preprocessed(
            synthetic_features_df, synthetic_binary_targets, gender_array
        )
        indices = preprocessed.get_fold_indices()

        for fold_idx, (train_idx, test_idx) in enumerate(indices):
            train_loader, test_loader = preprocessed.get_dataloader_fold(
                dataset, fold_idx, indices, num_workers=0, batch_size=16
            )
            model = DummyClassifierModel()
            trainer = SklearnTrainer(model=model)
            trainer.set_fold(fold_idx)

            trainer.fit(train_loader)
            y_pred = trainer.predict(test_loader)

            y_test = synthetic_binary_targets[test_idx].squeeze()
            metrics = compute_metrics(
                y_test, y_pred, prediction_task="binary_classification"
            )

            assert isinstance(metrics, dict)
            for key in ("accuracy", "precision", "recall", "f1"):
                assert key in metrics, f"Fold {fold_idx}: missing key {key!r}"

    def test_prediction_shape(
        self, synthetic_features_df, synthetic_binary_targets, gender_array
    ):
        dataset, preprocessed = _build_dataset_and_preprocessed(
            synthetic_features_df, synthetic_binary_targets, gender_array
        )
        indices = preprocessed.get_fold_indices()
        train_loader, test_loader = preprocessed.get_dataloader_fold(
            dataset, 0, indices, num_workers=0, batch_size=16
        )
        model = DummyClassifierModel()
        trainer = SklearnTrainer(model=model)
        trainer.fit(train_loader)
        y_pred = trainer.predict(test_loader)

        _, test_idx = indices[0]
        assert y_pred.shape[0] == len(test_idx)

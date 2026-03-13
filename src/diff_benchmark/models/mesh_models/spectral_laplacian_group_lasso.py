"""Sklearn bridge: frozen spectral Laplacian embeddings + Group Lasso head.

This module exposes :class:`SpectralLaplacianGroupLassoModel`, a mesh model that:

1. Uses :class:`~diff_benchmark.models.mesh_models.spectral_laplacian_model.SpectralLaplacianAdditiveModel`
   as a **frozen** feature extractor.
2. Flattens per-parcel embeddings ``(B, P, E) -> (B, P*E)``.
3. Fits a Group Lasso head with parcel-aligned groups (one group per parcel).
4. Tunes hyper-parameters with :class:`sklearn.model_selection.GridSearchCV`.

This gives a clean sklearn path for cross-validation and hyperparameter search
while keeping the graph embedding computation in the spectral mesh backbone.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from celer import GroupLasso
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from diff_benchmark.models.mesh_models.spectral_laplacian_model import (
    SpectralLaplacianAdditiveModel,
)
from diff_benchmark.models.utils_models.trainer import SklearnModel

_GLOBAL_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


class SpectralLaplacianEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Convert mesh dicts into flattened spectral Laplacian embeddings.

    Embeddings are cached by ``(model-config, sample-content)`` so repeated
    GridSearchCV folds/parameter combinations can reuse features instead of
    recomputing them.
    """

    def __init__(
        self,
        in_features: int = 1,
        n_spectral_components: int = 16,
        parcel_ids: Optional[List[int]] = None,
        device: str = "cpu",
        cache_embeddings: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.in_features = in_features
        self.n_spectral_components = n_spectral_components
        # Keep parity with SpectralLaplacianAdditiveModel:
        # parcel 0 is background / medial wall and is excluded.
        if parcel_ids is None:
            self.parcel_ids = None
        else:
            self.parcel_ids = [int(p) for p in parcel_ids if int(p) != 0]
        self.device = device
        self.cache_embeddings = cache_embeddings
        self.cache_dir = cache_dir

    def _cache_namespace(self) -> str:
        payload = {
            "version": 1,
            "in_features": int(self.in_features),
            "n_spectral_components": int(self.n_spectral_components),
            "parcel_ids": self.parcel_ids,
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous().numpy()
        return np.ascontiguousarray(np.asarray(value))

    @classmethod
    def _tensor_digest(cls, value) -> str:
        arr = cls._to_numpy(value)
        h = hashlib.md5()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
        return h.hexdigest()

    @classmethod
    def _sample_digest(cls, sample: dict) -> str:
        # Always compute a fresh content hash — no object-identity shortcut.
        # Using Python id() or data_ptr() as a memoisation key is unsafe:
        # the runtime may reuse the same address for a different object after
        # the original is garbage-collected, causing a stale digest to be
        # returned and the wrong embedding to be fetched from the cache.
        required = ("node_features", "parcel_labels", "edge_index")
        h = hashlib.md5()
        for key in required:
            h.update(cls._tensor_digest(sample[key]).encode("utf-8"))
        return h.hexdigest()

    def _cache_key(self, sample: dict) -> str:
        return f"{self.cache_namespace_}_{self._sample_digest(sample)}"

    def _cache_file(self, cache_key: str) -> Optional[Path]:
        if getattr(self, "cache_dir_", None) is None:
            return None
        return self.cache_dir_ / f"{cache_key}.npy"

    def _load_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        cached = _GLOBAL_EMBEDDING_CACHE.get(cache_key)
        if cached is not None:
            return cached

        cache_file = self._cache_file(cache_key)
        if cache_file is not None and cache_file.exists():
            try:
                arr = np.load(cache_file, allow_pickle=False)
                if arr.ndim == 2:
                    arr = np.asarray(arr, dtype=np.float32)
                    _GLOBAL_EMBEDDING_CACHE[cache_key] = arr
                    return arr
            except Exception:
                pass
        return None

    def _store_cached_embedding(self, cache_key: str, embedding: np.ndarray) -> None:
        embedding = np.asarray(embedding, dtype=np.float32)
        _GLOBAL_EMBEDDING_CACHE[cache_key] = embedding

        cache_file = self._cache_file(cache_key)
        if cache_file is None or cache_file.exists():
            return

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{cache_key}_",
            suffix=".npy",
            dir=str(cache_file.parent),
        )
        os.close(fd)
        try:
            np.save(tmp_name, embedding, allow_pickle=False)
            os.replace(tmp_name, cache_file)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def _postprocess_batch_embeddings(
        self, emb: torch.Tensor
    ) -> tuple[np.ndarray, list[int]]:
        if emb.dim() != 3:
            raise ValueError(f"Expected embeddings (B, P, E), got {tuple(emb.shape)}")

        _, n_parcels, _ = emb.shape
        parcel_ids = list(
            self.backbone_._parcel_ids
            if getattr(self.backbone_, "_parcel_ids", None) is not None
            else range(n_parcels)
        )

        keep_idx = [i for i, pid in enumerate(parcel_ids) if int(pid) != 0]
        if len(keep_idx) != n_parcels:
            emb = emb[:, keep_idx, :]
            parcel_ids = [parcel_ids[i] for i in keep_idx]

        return emb.contiguous().numpy(), parcel_ids

    def _resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def _build_backbone(self) -> SpectralLaplacianAdditiveModel:
        backbone = SpectralLaplacianAdditiveModel(
            in_features=self.in_features,
            n_spectral_components=self.n_spectral_components,
            parcel_ids=self.parcel_ids,
        )
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        return backbone

    def fit(self, X, y=None):
        self.device_ = self._resolved_device()
        self.backbone_ = self._build_backbone().to(self.device_)
        self.cache_namespace_ = self._cache_namespace()

        if self.cache_embeddings and self.cache_dir:
            self.cache_dir_ = Path(self.cache_dir).expanduser() / self.cache_namespace_
            self.cache_dir_.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir_ = None

        # Eagerly initialise parcel IDs so they are available even when every
        # sample is a cache hit and backbone.forward() is never called during
        # transform().  _maybe_init_from_batch only reads parcel_labels from
        # the first sample — it does not run a model forward pass.
        if isinstance(X, list) and len(X) > 0:
            self.backbone_._maybe_init_from_batch(X)

        # Fit does not learn parameters, but we run one pass to infer output shape.
        _ = self.transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "backbone_")
        if not isinstance(X, list):
            raise TypeError(
                "SpectralLaplacianEmbeddingTransformer expects a list of mesh dicts."
            )

        if len(X) == 0:
            n_features = int(
                getattr(self, "n_parcels_", 0) * getattr(self, "parcel_embed_dim_", 0)
            )
            return np.zeros((0, n_features), dtype=np.float32)

        if not self.cache_embeddings:
            with torch.no_grad():
                emb = self.backbone_(X).detach().cpu()  # (B, P, E)
            emb_np, parcel_ids = self._postprocess_batch_embeddings(emb)
        else:
            sample_embeddings: list[Optional[np.ndarray]] = [None] * len(X)
            missing_idx: list[int] = []
            missing_samples: list[dict] = []
            missing_keys: list[str] = []

            for idx, sample in enumerate(X):
                cache_key = self._cache_key(sample)
                cached = self._load_cached_embedding(cache_key)
                if cached is None:
                    missing_idx.append(idx)
                    missing_samples.append(sample)
                    missing_keys.append(cache_key)
                else:
                    sample_embeddings[idx] = cached

            parcel_ids = list(
                self.backbone_._parcel_ids
                if getattr(self.backbone_, "_parcel_ids", None) is not None
                else []
            )
            parcel_ids = [pid for pid in parcel_ids if int(pid) != 0]

            if missing_samples:
                with torch.no_grad():
                    missing_emb = self.backbone_(missing_samples).detach().cpu()
                missing_emb_np, parcel_ids = self._postprocess_batch_embeddings(
                    missing_emb
                )
                for local_i, global_i in enumerate(missing_idx):
                    sample_emb = np.asarray(missing_emb_np[local_i], dtype=np.float32)
                    sample_embeddings[global_i] = sample_emb
                    self._store_cached_embedding(missing_keys[local_i], sample_emb)

            # All samples must be resolved either from cache or new compute.
            unresolved = [i for i, val in enumerate(sample_embeddings) if val is None]
            if unresolved:
                raise RuntimeError(
                    f"Failed to resolve embeddings for sample indices: {unresolved[:5]}"
                )

            emb_np = np.stack(sample_embeddings, axis=0)

        n_subjects, n_parcels, parcel_embed_dim = emb_np.shape
        self.n_parcels_ = int(n_parcels)
        self.parcel_embed_dim_ = int(parcel_embed_dim)
        self.parcel_ids_ = parcel_ids if parcel_ids else list(range(n_parcels))
        return emb_np.reshape(n_subjects, n_parcels * parcel_embed_dim)


class SpectralGroupLassoRegressor(BaseEstimator, RegressorMixin):
    """Group Lasso regressor with parcel-group structure from spectral embeddings."""

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    @staticmethod
    def _groups_from_transformer(transformer) -> list[list[int]]:
        n_parcels = int(transformer.n_parcels_)
        parcel_embed_dim = int(transformer.parcel_embed_dim_)
        groups: list[list[int]] = []
        start = 0
        for _ in range(n_parcels):
            end = start + parcel_embed_dim
            groups.append(list(range(start, end)))
            start = end
        return groups

    def fit(self, X: np.ndarray, y: np.ndarray):
        transformer = getattr(self, "_transformer", None)
        if transformer is not None:
            try:
                check_is_fitted(transformer, ("n_parcels_", "parcel_embed_dim_"))
                self.groups_ = self._groups_from_transformer(transformer)
            except Exception:
                self.groups_ = [[i] for i in range(X.shape[1])]
        else:
            self.groups_ = [[i] for i in range(X.shape[1])]

        self.estimator_ = GroupLasso(
            groups=self.groups_,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        # For classification pipelines: keep shape fixed, zero-out inactive columns.
        check_is_fitted(self, "estimator_")
        coef = self.estimator_.coef_
        mask = coef != 0.0
        return X * mask[np.newaxis, :]

    @property
    def coef_(self) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.coef_

    @property
    def intercept_(self) -> float:
        check_is_fitted(self, "estimator_")
        return self.estimator_.intercept_


class _SpectralGroupLassoPipeline(Pipeline):
    """Pipeline that injects the fitted spectral transformer into Group Lasso."""

    def fit(self, X, y=None, **params):  # type: ignore[override]
        feature_transformer = self.named_steps.get("spectral")
        Xt = X

        for name, step in self.steps:
            if step is None or step == "passthrough":
                continue

            if isinstance(step, SpectralGroupLassoRegressor) and feature_transformer is not None:
                step._transformer = feature_transformer

            is_last = name == self.steps[-1][0]
            if is_last:
                step.fit(Xt, y)
            else:
                if hasattr(step, "fit_transform"):
                    Xt = step.fit_transform(Xt, y)
                else:
                    step.fit(Xt, y)
                    Xt = step.transform(Xt)

        return self

    def predict(self, X, **predict_params):  # type: ignore[override]
        Xt = X
        for _, transformer in self.steps[:-1]:
            if transformer is None or transformer == "passthrough":
                continue
            Xt = transformer.transform(Xt)
        return self.steps[-1][1].predict(Xt)


class SpectralLaplacianGroupLassoModel(SklearnModel):
    """Frozen spectral Laplacian backbone + Group Lasso head with GridSearchCV."""

    data_type: str = "mesh"

    def _build_model(self, **kwargs):
        self.embedding_cache_warm_start = kwargs.get("embedding_cache_warm_start", True)
        prediction_task = kwargs.get("prediction_task", "regression")
        in_features = kwargs.get("in_features", 1)
        n_spectral_components = kwargs.get("n_spectral_components", 16)
        n_spectral_components_grid = kwargs.get(
            "n_spectral_components_grid", [16, 32, 64]
        )
        alpha_grid = np.logspace(-4, -2, 3)
        cv = kwargs.get("cv", 5)
        n_jobs = kwargs.get("n_jobs", 1)
        verbose = kwargs.get("verbose", 3)
        device = kwargs.get("device", "cpu")
        random_state = kwargs.get("random_state", 42)
        cache_embeddings = kwargs.get("cache_embeddings", True)
        cache_dir = kwargs.get(
            "embedding_cache_dir",
            os.path.join(tempfile.gettempdir(), "diff_benchmark", "spectral_embeddings"),
        )

        spectral = SpectralLaplacianEmbeddingTransformer(
            in_features=in_features,
            n_spectral_components=n_spectral_components,
            parcel_ids=kwargs.get("parcel_ids"),
            device=device,
            cache_embeddings=cache_embeddings,
            cache_dir=cache_dir,
        )
        group_lasso = SpectralGroupLassoRegressor(
            max_iter=kwargs.get("group_lasso_max_iter", 100),
            tol=kwargs.get("group_lasso_tol", 1e-4),
            fit_intercept=kwargs.get("group_lasso_fit_intercept", True),
        )

        if prediction_task == "regression":
            pipeline = _SpectralGroupLassoPipeline(
                [("spectral", spectral), ("group_lasso", group_lasso)]
            )
            param_grid = {
                "spectral__n_spectral_components": n_spectral_components_grid,
                "group_lasso__alpha": alpha_grid,
            }
            scoring = "neg_mean_absolute_error"

        elif prediction_task == "binary_classification":
            pipeline = _SpectralGroupLassoPipeline(
                [
                    ("spectral", spectral),
                    ("group_lasso", group_lasso),
                    ("classifier", LogisticRegression(max_iter=5000, random_state=random_state)),
                ]
            )
            param_grid = {
                "spectral__n_spectral_components": n_spectral_components_grid,
                "group_lasso__alpha": alpha_grid,
                "classifier__C": kwargs.get("classifier_c_grid", np.logspace(-2, 0, 3)),
            }
            scoring = "balanced_accuracy"
        else:
            raise ValueError(f"Unknown prediction_task '{prediction_task}'")

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def fit(self, X, y: np.ndarray):
        if self.embedding_cache_warm_start:
            self._warm_start_embedding_cache(X)
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def _warm_start_embedding_cache(self, X) -> None:
        if not isinstance(X, list) or len(X) == 0:
            return
        if not isinstance(getattr(self, "model", None), GridSearchCV):
            return

        pipeline = self.model.estimator
        spectral = getattr(pipeline, "named_steps", {}).get("spectral")
        if not isinstance(spectral, SpectralLaplacianEmbeddingTransformer):
            return
        if not spectral.cache_embeddings:
            return

        grid_values: set[int] = {int(spectral.n_spectral_components)}
        param_grid = self.model.param_grid
        if isinstance(param_grid, dict):
            values = param_grid.get("spectral__n_spectral_components", [])
            grid_values.update(int(v) for v in values)
        elif isinstance(param_grid, list):
            for grid in param_grid:
                values = grid.get("spectral__n_spectral_components", [])
                grid_values.update(int(v) for v in values)

        for n_components in sorted(grid_values):
            warm = SpectralLaplacianEmbeddingTransformer(
                in_features=spectral.in_features,
                n_spectral_components=n_components,
                parcel_ids=spectral.parcel_ids,
                device=spectral.device,
                cache_embeddings=spectral.cache_embeddings,
                cache_dir=spectral.cache_dir,
            )
            warm.fit(X)

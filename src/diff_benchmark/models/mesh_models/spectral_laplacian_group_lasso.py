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


class SpectralLaplacianEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Convert mesh dicts into flattened spectral Laplacian embeddings."""

    def __init__(
        self,
        in_features: int = 1,
        n_spectral_components: int = 16,
        parcel_ids: Optional[List[int]] = None,
        device: str = "cpu",
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
        # Fit does not learn parameters, but we run one pass to infer output shape.
        _ = self.transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "backbone_")
        if not isinstance(X, list):
            raise TypeError(
                "SpectralLaplacianEmbeddingTransformer expects a list of mesh dicts."
            )

        with torch.no_grad():
            emb = self.backbone_(X).detach().cpu()  # (B, P, E)

        if emb.dim() != 3:
            raise ValueError(f"Expected embeddings (B, P, E), got {tuple(emb.shape)}")

        n_subjects, n_parcels, parcel_embed_dim = emb.shape
        parcel_ids = list(
            self.backbone_._parcel_ids
            if getattr(self.backbone_, "_parcel_ids", None) is not None
            else range(n_parcels)
        )

        # Defensive parity with backbone behavior: drop parcel 0 if present.
        keep_idx = [i for i, pid in enumerate(parcel_ids) if int(pid) != 0]
        if len(keep_idx) != n_parcels:
            emb = emb[:, keep_idx, :]
            parcel_ids = [parcel_ids[i] for i in keep_idx]
            n_subjects, n_parcels, parcel_embed_dim = emb.shape

        self.n_parcels_ = int(n_parcels)
        self.parcel_embed_dim_ = int(parcel_embed_dim)
        self.parcel_ids_ = parcel_ids
        return emb.reshape(n_subjects, n_parcels * parcel_embed_dim).numpy()


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
        prediction_task = kwargs.get("prediction_task", "regression")
        in_features = kwargs.get("in_features", 1)
        n_spectral_components = kwargs.get("n_spectral_components", 16)
        n_spectral_components_grid = [16, 32, 64]
        alpha_grid = np.logspace(-3, 3, 7)
        cv = kwargs.get("cv", 5)
        n_jobs = kwargs.get("n_jobs", 1)
        verbose = kwargs.get("verbose", 3)
        device = kwargs.get("device", "cpu")
        random_state = kwargs.get("random_state", 42)

        spectral = SpectralLaplacianEmbeddingTransformer(
            in_features=in_features,
            n_spectral_components=n_spectral_components,
            parcel_ids=kwargs.get("parcel_ids"),
            device=device,
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
                "classifier__C": kwargs.get("classifier_c_grid", np.logspace(-3, 3, 7)),
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
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

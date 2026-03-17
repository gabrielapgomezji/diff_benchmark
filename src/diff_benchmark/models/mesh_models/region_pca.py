"""Region-PCA model for surface-mesh data.

This module provides :class:`RegionPCATransformer` and
:class:`RegionPCAModel`.

Pipeline
--------
::

    mesh batch (list of dicts)
          │
          ▼  RegionPCATransformer
          For each subject:
            For each parcel r  (label > 0):
              region_nodes = node_features[parcel_labels == r]  (M_r, F)
              PCA(n_components=k).fit_transform(region_nodes)
              → first k components (or projected variance)
              → flatten → region feature vector of length k*F (or k)
            Concatenate all regions → (n_regions * n_components,)
          │
          ▼  (n_subjects, n_regions * n_components)
          │
          ▼  Group Lasso (GridSearchCV over alpha, n_components)
          │
          ▼  predictions (n_subjects,)

The transformer follows the standard sklearn API (``fit`` / ``transform`` /
``fit_transform``) so it can be embedded in a :class:`sklearn.pipeline.Pipeline`
and its hyper-parameters can be tuned by :class:`sklearn.model_selection.GridSearchCV`.

Notes
-----
- Label ``0`` is treated as the medial-wall / background and is always skipped.
- If a parcel has fewer nodes than ``n_components``, the number of components
  is automatically clamped to ``min(n_components, n_nodes)`` for that parcel.
- Region indices (parcel → node mask) are cached after the first ``fit`` call
  so that repeated ``transform`` calls on data with the same atlas are fast.
- The transformer projects each parcel's node-feature matrix onto its top-k
  PCA components and then uses the **explained variance of those components**
  (a scalar per component) as the feature vector for that region. This gives a
  fixed-length vector per region regardless of the number of nodes or features,
  while remaining informative about the spectral structure of the parcel.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from skglm import GeneralizedLinearEstimator
from skglm.datafits import Quadratic
from skglm.penalties import WeightedGroupL2
from skglm.solvers import AndersonCD

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from diff_benchmark.models.utils_models.trainer import SklearnModel


def _rss_mb() -> float | None:
    """Return the current process RSS in MB, or ``None`` if unavailable."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RegionPCATransformer
# ---------------------------------------------------------------------------


class RegionPCATransformer(BaseEstimator, TransformerMixin):
    """Convert a list of mesh dicts into a fixed-length feature matrix.

    For every subject a feature vector is produced by:

    1. Grouping vertices by ``parcel_labels`` (label 0 is skipped).
    2. Standardising node features inside each parcel (fit on training only).
    3. Fitting PCA inside each parcel on the **training** set and projecting
       all subjects onto the retained components.
    4. Using the **explained variance** of each component as the scalar
       feature for that (parcel, component) pair.
    5. Concatenating the per-parcel feature vectors into one vector per
       subject of length ``n_regions * n_components``.

    Parameters
    ----------
    n_components:
        Number of PCA components to retain per parcel.  If a parcel has
        fewer vertices than ``n_components``, the number of components is
        silently clamped to the parcel size.

    Attributes
    ----------
    pca_per_region_ : dict[int, IncrementalPCA]
        Fitted PCA objects keyed by parcel label.  Set after ``fit``.
    scaler_per_region_ : dict[int, StandardScaler]
        Fitted standard scalers keyed by parcel label. Set after ``fit``.
    region_order_ : list[int]
        Sorted list of parcel labels seen during ``fit``; defines the
        column order in the output matrix.
    n_features_out_ : int
        Total number of output features (``n_regions * n_components``
        approximately, accounting for small parcels).
    """

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_arrays(
        mesh: Dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(node_features, parcel_labels)`` as numpy arrays."""
        nf = mesh["node_features"]
        pl = mesh["parcel_labels"]
        if hasattr(nf, "numpy"):
            nf = nf.numpy()
        if hasattr(pl, "numpy"):
            pl = pl.numpy()
        # Ensure 2-D feature matrix  (N, F)
        if nf.ndim == 1:
            nf = nf[:, np.newaxis]
        return nf.astype(np.float32, copy=False), pl.astype(np.int32, copy=False)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: List[Dict], y=None) -> "RegionPCATransformer":
        """Fit one PCA per parcel using incremental updates (one subject at a time).

        Uses :class:`~sklearn.decomposition.IncrementalPCA` so that at any
        point only a single subject's node features are held in RAM, rather
        than the full ``(n_subjects × nodes_per_parcel, F)`` matrix.

        Parameters
        ----------
        X:
            List of mesh dicts, each with keys ``"node_features"`` and
            ``"parcel_labels"``.
        y:
            Ignored; present for sklearn API compatibility.

        Returns
        -------
        self
        """
        # ----------------------------------------------------------------
        # Pass 1 — lightweight scan: determine parcel labels, clamp k.
        # No node features are stored; only per-parcel min-node counts.
        # ----------------------------------------------------------------
        parcel_min_nodes: dict[int, int] = {}
        n_features: int = 0

        for mesh in X:
            nf, pl = self._extract_arrays(mesh)
            if n_features == 0:
                n_features = nf.shape[1]
            for label in np.unique(pl):
                if label == 0:
                    continue
                n_nodes = int((pl == label).sum())
                label = int(label)
                if label not in parcel_min_nodes or n_nodes < parcel_min_nodes[label]:
                    parcel_min_nodes[label] = n_nodes

        # ----------------------------------------------------------------
        # Initialise one StandardScaler + IncrementalPCA per parcel.
        # ----------------------------------------------------------------
        self.scaler_per_region_: dict[int, StandardScaler] = {}
        self.pca_per_region_: dict[int, IncrementalPCA] = {}
        self.n_components_per_region_: dict[int, int] = {}

        for label, min_nodes in parcel_min_nodes.items():
            k = min(self.n_components, min_nodes, n_features)
            self.scaler_per_region_[label] = StandardScaler(copy=False)
            self.pca_per_region_[label] = IncrementalPCA(n_components=k)
            self.n_components_per_region_[label] = k

        # ----------------------------------------------------------------
        # Pass 2 — fit per-parcel scalers incrementally.
        # ----------------------------------------------------------------
        n_subjects = len(X)
        n_parcels = len(self.pca_per_region_)
        approx_mb = (
            next(iter(X))["node_features"].shape[0] * n_features * 4 / 1e6
            if hasattr(next(iter(X))["node_features"], "shape")
            else float("nan")
        )
        for subject_idx, mesh in enumerate(X):
            nf, pl = self._extract_arrays(mesh)
            skipped = 0
            for label, scaler in self.scaler_per_region_.items():
                mask = pl == label
                if mask.sum() == 0:
                    skipped += 1
                    continue
                scaler.partial_fit(nf[mask])
            mem_mb = _rss_mb()

        # ----------------------------------------------------------------
        # Pass 3 — incremental PCA fit: one subject at a time per parcel.
        # Only one subject's mesh is in RAM at each step.
        # ----------------------------------------------------------------
        for subject_idx, mesh in enumerate(X):
            nf, pl = self._extract_arrays(mesh)
            skipped = 0
            for label, pca in self.pca_per_region_.items():
                mask = pl == label
                if mask.sum() < pca.n_components:
                    # Too few nodes in this subject to update this parcel.
                    skipped += 1
                    continue
                region_nodes = nf[mask]
                scaler = self.scaler_per_region_[label]
                region_nodes = scaler.transform(region_nodes)
                pca.partial_fit(region_nodes)
            mem_mb = _rss_mb()

        self.region_order_: list[int] = sorted(self.pca_per_region_.keys())
        self.n_features_out_: int = sum(
            self.n_components_per_region_[r] for r in self.region_order_
        )
        return self

    def transform(self, X: List[Dict], y=None) -> np.ndarray:
        """Project each subject's mesh onto the fitted per-parcel PCAs.

        For each subject and each parcel the **explained variance** of the
        top-k components is used as the feature representation (a k-dimensional
        vector per parcel).

        Parameters
        ----------
        X:
            List of mesh dicts (same format as ``fit``).
        y:
            Ignored.

        Returns
        -------
        np.ndarray of shape ``(n_subjects, n_features_out_)``
        """
        out = np.zeros((len(X), self.n_features_out_), dtype=np.float32)

        for i, mesh in enumerate(X):
            nf, pl = self._extract_arrays(mesh)
            col = 0
            for label in self.region_order_:
                k = self.n_components_per_region_[label]
                pca = self.pca_per_region_[label]
                mask = pl == label
                if mask.sum() == 0:
                    # Parcel absent in this subject — fill with zeros
                    col += k
                    continue
                region_nodes = nf[mask]  # (M_r, F)
                scaler = self.scaler_per_region_[label]
                region_nodes = scaler.transform(region_nodes)
                # Project onto PCA components → (M_r, k)
                projected = pca.transform(region_nodes)
                # Explained variance of each component for this subject
                # = variance of projected scores along each PC axis
                ev = (
                    projected.var(axis=0).astype(np.float32, copy=False)
                    if projected.shape[0] > 1
                    else np.abs(projected[0]).astype(np.float32, copy=False)
                )
                # ev shape: (k,)
                out[i, col : col + k] = ev
                col += k

        return out


# ---------------------------------------------------------------------------
# Group Lasso on Region-PCA features
# ---------------------------------------------------------------------------


class GroupLassoRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible Group Lasso regressor on Region-PCA features.

    A fitted :class:`RegionPCATransformer` is injected into ``self._transformer``
    by :class:`_RegionPCAPipeline` before this estimator is fit so that group
    boundaries align with region-wise PCA blocks.
    """

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
        groups: list[list[int]] = []
        col = 0
        for label in transformer.region_order_:
            k = int(transformer.n_components_per_region_[label])
            groups.append(list(range(col, col + k)))
            col += k
        return groups

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GroupLassoRegressor":
        transformer = getattr(self, "_transformer", None)
        if transformer is not None:
            try:
                check_is_fitted(transformer)
                self.groups_ = self._groups_from_transformer(transformer)
            except Exception:
                self.groups_ = [[i] for i in range(X.shape[1])]
        else:
            self.groups_ = [[i] for i in range(X.shape[1])]

        grp_indices = np.asarray(
            [idx for group in self.groups_ for idx in group],
            dtype=np.int32,
        )
        grp_sizes = np.asarray([len(group) for group in self.groups_], dtype=np.int32)
        grp_ptr = np.zeros(len(grp_sizes) + 1, dtype=np.int32)
        grp_ptr[1:] = np.cumsum(grp_sizes)

        penalty = WeightedGroupL2(
            alpha=self.alpha,
            weights=np.ones(len(self.groups_), dtype=float),
            grp_ptr=grp_ptr,
            grp_indices=grp_indices,
        )
        solver = AndersonCD(
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        self.estimator_ = GeneralizedLinearEstimator(
            datafit=Quadratic(),
            penalty=penalty,
            solver=solver,
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Zero out inactive features (used before LogisticRegression)."""
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


class _RegionPCAPipeline(Pipeline):
    """Pipeline that injects the fitted RegionPCATransformer into GroupLasso."""

    def fit(self, X, y=None, **params):  # type: ignore[override]
        region_pca = self.named_steps.get("region_pca")
        Xt = X
        for name, step in self.steps:
            if step is None or step == "passthrough":
                continue

            if isinstance(step, GroupLassoRegressor) and region_pca is not None:
                step._transformer = region_pca

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


# ---------------------------------------------------------------------------
# RegionPCAModel
# ---------------------------------------------------------------------------


class RegionPCAModel(SklearnModel):
    """Region-PCA + Group Lasso model for surface-mesh data.

    Extends :class:`~diff_benchmark.models.utils_models.trainer.SklearnModel`
    to accept ``data_type = "mesh"`` inputs.  The mesh-to-array conversion
    is handled transparently by :class:`RegionPCATransformer` inside the
    sklearn ``Pipeline``.

    Parameters forwarded to ``_build_model``
    -----------------------------------------
    prediction_task : str
        ``"regression"`` or ``"binary_classification"``.
    n_jobs : int
        Passed to :class:`~sklearn.model_selection.GridSearchCV`.
        Defaults to ``1`` so that per-subject progress prints from
        :class:`RegionPCATransformer` is visible in the main process.
        Set to ``-1`` to re-enable full parallelism once debugging is done.
    random_state : int
        Random seed (default: 42).

    Pipeline
    --------
    Regression: ``RegionPCATransformer`` → ``GroupLassoRegressor``
    Classification: ``RegionPCATransformer`` → ``GroupLassoRegressor`` → ``LogisticRegression``

    Grid search
    -----------
    ``region_pca__n_components``: [3, 5, 10]
    ``group_lasso__alpha``:       log-spaced grid [1e-5 … 1e5]
    """

    data_type: str = "mesh"

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", "regression")
        self.output_dim = 1
        # Set n_jobs=1 to keep transformer prints visible in the main process.
        # Switch back to -1 once debugging is complete.
        # (Worker processes may buffer or drop stdout in some schedulers.)
        #
        # OOM investigation is complete.
        n_jobs = kwargs.get("n_jobs", -1)
        verbose = kwargs.get("verbose", 1)

        if self.prediction_task == "regression":
            pipeline = _RegionPCAPipeline(
                [
                    ("region_pca", RegionPCATransformer()),
                    ("group_lasso", GroupLassoRegressor()),
                ]
            )
            param_grid = {
                "region_pca__n_components": [3, 5, 10],
                "group_lasso__alpha": np.logspace(-5, 5, 10),
            }
            scoring = "neg_mean_absolute_error"

        elif self.prediction_task == "binary_classification":
            pipeline = _RegionPCAPipeline(
                [
                    ("region_pca", RegionPCATransformer()),
                    ("group_lasso", GroupLassoRegressor()),
                    ("classifier", LogisticRegression(max_iter=5000)),
                ]
            )
            param_grid = {
                "region_pca__n_components": [3, 5, 10],
                "group_lasso__alpha": np.logspace(-5, -2, 5), #np.logspace(-5, 5, 10),
                "classifier__C": np.logspace(-2, 2, 5), #np.logspace(-5, 5, 10),
            }
            scoring = "balanced_accuracy"

        else:
            raise ValueError(
                f"Unknown prediction_task '{self.prediction_task}'."
            )
        
        print(f"Jobs={n_jobs}, param_grid={param_grid}")

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=n_jobs,
            pre_dispatch="n_jobs",
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Override fit / predict to handle mesh lists from the DataLoader
    # ------------------------------------------------------------------

    def fit(self, X, y: np.ndarray):
        """Fit the pipeline to *X*.

        Parameters
        ----------
        X:
            Either a **list of mesh dicts** (``data_type="mesh"`` path, coming
            directly from :class:`~diff_benchmark.models.utils_models.trainer.SklearnTrainer`)
            or a pre-converted ``np.ndarray``.
        y:
            Regression targets of shape ``(n_subjects,)``.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Return predictions for *X*.

        Parameters
        ----------
        X:
            List of mesh dicts or ``np.ndarray`` (same as ``fit``).
        """
        return self.model.predict(X)

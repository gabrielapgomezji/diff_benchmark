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
          ▼  Ridge regression  (GridSearchCV over alpha, n_components)
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

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
    2. Fitting PCA inside each parcel on the **training** set and projecting
       all subjects onto the retained components.
    3. Using the **explained variance** of each component as the scalar
       feature for that (parcel, component) pair.
    4. Concatenating the per-parcel feature vectors into one vector per
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
        return nf.astype(np.float64), pl.astype(np.int64)

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
        # Initialise one IncrementalPCA per parcel with the correct k.
        # ----------------------------------------------------------------
        self.pca_per_region_: dict[int, IncrementalPCA] = {}
        self.n_components_per_region_: dict[int, int] = {}

        for label, min_nodes in parcel_min_nodes.items():
            k = min(self.n_components, min_nodes, n_features)
            self.pca_per_region_[label] = IncrementalPCA(n_components=k)
            self.n_components_per_region_[label] = k

        # ----------------------------------------------------------------
        # Pass 2 — incremental fit: one subject at a time per parcel.
        # Only one subject's mesh is in RAM at each step.
        # ----------------------------------------------------------------
        n_subjects = len(X)
        n_parcels = len(self.pca_per_region_)
        logger.info(
            "IncrementalPCA fit starting: %d subjects, %d parcels, "
            "n_components=%d, n_features=%d — approx. %.1f MB/subject.",
            n_subjects, n_parcels, self.n_components, n_features,
            # rough upper bound: full node-feature matrix for one subject
            next(iter(X))["node_features"].shape[0] * n_features * 8 / 1e6
            if hasattr(next(iter(X))["node_features"], "shape")
            else float("nan"),
        )
        for subject_idx, mesh in enumerate(X):
            nf, pl = self._extract_arrays(mesh)
            skipped = 0
            for label, pca in self.pca_per_region_.items():
                mask = pl == label
                if mask.sum() < pca.n_components:
                    # Too few nodes in this subject to update this parcel.
                    skipped += 1
                    continue
                pca.partial_fit(nf[mask])
            mem_mb = _rss_mb()
            logger.info(
                "  subject %d/%d done — nf shape %s, parcels skipped: %d, "
                "process RSS: %s",
                subject_idx + 1, n_subjects, nf.shape, skipped,
                f"{mem_mb:.1f} MB" if mem_mb is not None else "n/a",
            )

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
        out = np.zeros((len(X), self.n_features_out_), dtype=np.float64)

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
                # Project onto PCA components → (M_r, k)
                projected = pca.transform(region_nodes)
                # Explained variance of each component for this subject
                # = variance of projected scores along each PC axis
                ev = projected.var(axis=0) if projected.shape[0] > 1 else np.abs(projected[0])
                # ev shape: (k,)
                out[i, col : col + k] = ev
                col += k

        return out


# ---------------------------------------------------------------------------
# RegionPCAModel
# ---------------------------------------------------------------------------


class RegionPCAModel(SklearnModel):
    """Region-PCA + Ridge regression model for surface-mesh data.

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
        Defaults to ``1`` so that per-subject logging from
        :class:`RegionPCATransformer` is visible in the main process.
        Set to ``-1`` to re-enable full parallelism once debugging is done.
    random_state : int
        Random seed (default: 42).

    Pipeline
    --------
    ``RegionPCATransformer`` → ``Ridge``

    Grid search
    -----------
    ``region_pca__n_components``: [1, 2, 3, 5]
    ``model__alpha``:             log-spaced grid [1e-3 … 1e3]
    """

    data_type: str = "mesh"

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", "regression")
        self.output_dim = 1
        # Set n_jobs=1 to keep transformer logs visible (joblib worker
        # processes swallow logging output). Switch back to -1 once the
        # OOM investigation is complete.
        n_jobs = kwargs.get("n_jobs", 1)

        if self.prediction_task == "regression":
            estimator = Ridge()
            param_grid = {
                "region_pca__n_components": [1, 2, 3, 5],
                "model__alpha": np.logspace(-3, 3, 13),
            }
            scoring = "neg_mean_absolute_error"
        elif self.prediction_task == "binary_classification":
            estimator = LogisticRegression(max_iter=5000)
            param_grid = {
                "region_pca__n_components": [1, 2, 3, 5],
                "model__C": np.logspace(-3, 3, 3), #13),
            }
            scoring = "balanced_accuracy"
            
        else:
            raise ValueError(
                f"Unknown prediction_task '{self.prediction_task}'."
            )

        pipeline = Pipeline(
            [
                ("region_pca", RegionPCATransformer()),
                ("model", estimator),
            ]
        )

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=n_jobs,
            verbose=3,
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

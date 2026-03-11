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

from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from diff_benchmark.models.utils_models.trainer import SklearnModel


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
    pca_per_region_ : dict[int, PCA]
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
        """Fit one PCA per parcel on the concatenated training subjects.

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
        # 1. Collect all vertices per parcel across training subjects
        # ----------------------------------------------------------------
        parcel_nodes: dict[int, list[np.ndarray]] = {}

        for mesh in X:
            nf, pl = self._extract_arrays(mesh)
            unique_labels = np.unique(pl)
            for label in unique_labels:
                if label == 0:
                    continue  # medial wall / background
                mask = pl == label
                parcel_nodes.setdefault(int(label), []).append(nf[mask])

        # ----------------------------------------------------------------
        # 2. Fit PCA per parcel
        # ----------------------------------------------------------------
        self.pca_per_region_: dict[int, PCA] = {}
        self.n_components_per_region_: dict[int, int] = {}

        for label, node_list in parcel_nodes.items():
            region_matrix = np.concatenate(node_list, axis=0)  # (M_total, F)
            k = min(self.n_components, region_matrix.shape[0], region_matrix.shape[1])
            pca = PCA(n_components=k)
            pca.fit(region_matrix)
            self.pca_per_region_[label] = pca
            self.n_components_per_region_[label] = k

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
        ``"regression"`` or ``"binary_classification"``.  Only regression is
        supported; a ``ValueError`` is raised for classification tasks.
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
            n_jobs=-1,
            verbose=1,
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

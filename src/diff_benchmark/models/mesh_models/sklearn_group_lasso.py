"""sklearn-compatible Group Lasso model for region-level mesh features.

This module provides :class:`GroupLassoRegressor` and
:class:`RegionGroupLassoModel`.

Pipeline
--------
::

    mesh batch (list of dicts)
          │
          ▼  RegionFeatureTransformer
          (n_subjects, n_flat_region_features)
          │
          ▼  StandardScaler
          │
          ▼  GroupLassoRegressor  ← celer.GroupLasso with automatic group structure
          predictions (n_subjects,)

Group structure
---------------
After :class:`RegionFeatureTransformer` produces an
``(n_subjects, n_features)`` matrix the columns are laid out as contiguous
blocks — one block per region ``r`` — each of size
``n_nodes_in_region_r * n_node_features``.

``celer.GroupLasso`` accepts ``groups`` as a list-of-lists of feature indices,
so :class:`GroupLassoRegressor` reads ``region_order_`` and
``region_sizes_`` / ``n_node_features_`` from the fitted transformer (injected by the
custom :class:`_GroupLassoPipeline`) and constructs the group specification
automatically.

``GridSearchCV`` tunes Group Lasso's regularisation strength
(``group_lasso__alpha``) by default.

Notes
-----
- ``celer.GroupLasso`` minimises

  .. math::

      \\frac{1}{2n} \\|y - Xw\\|^2 + \\alpha \\sum_g \\|w_g\\|_2

- Grid search covers ``group_lasso__alpha`` (and ``classifier__C`` for
  binary classification).
- ``data_type = "mesh"`` signals
  :class:`~diff_benchmark.models.utils_models.trainer.SklearnTrainer` to feed
  raw mesh lists instead of numpy arrays.
"""

from __future__ import annotations

import numpy as np
from celer import GroupLasso
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from diff_benchmark.models.utils_models.trainer import SklearnModel

# ---------------------------------------------------------------------------
# Region Transformer
# ---------------------------------------------------------------------------
class RegionFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Convert mesh dicts into a flat feature matrix while preserving
    region structure for Group Lasso.
    """

    def fit(self, X, y=None):

        mesh = X[0]

        pl = mesh["parcel_labels"]
        nf = mesh["node_features"]

        if hasattr(pl, "numpy"):
            pl = pl.numpy()

        if hasattr(nf, "numpy"):
            nf = nf.numpy()

        # number of node features per vertex
        self.n_node_features_ = nf.shape[1]

        self.region_order_ = sorted(np.unique(pl))
        self.region_order_ = [r for r in self.region_order_ if r != 0]

        self.region_sizes_ = {}

        for r in self.region_order_:
            mask = pl == r
            self.region_sizes_[r] = mask.sum()

        return self

    def transform(self, X):

        features = []

        for mesh in X:
            nf = mesh["node_features"]
            pl = mesh["parcel_labels"]

            if hasattr(nf, "numpy"):
                nf = nf.numpy()
            if hasattr(pl, "numpy"):
                pl = pl.numpy()

            subj_feat = []

            for r in self.region_order_:
                mask = pl == r
                region_nodes = nf[mask]           # (n_nodes, n_features)

                subj_feat.append(region_nodes.flatten())

            features.append(np.concatenate(subj_feat))

        return np.vstack(features)
    
# ---------------------------------------------------------------------------
# GroupLassoRegressor
# ---------------------------------------------------------------------------


class GroupLassoRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible Group Lasso regressor backed by ``celer.GroupLasso``.

    Group membership is read from a fitted
    :class:`RegionFeatureTransformer`
    that is injected into ``self._transformer`` by :class:`_GroupLassoPipeline`
    immediately before ``fit`` is called.  If no transformer is available the
    estimator falls back to one singleton group per feature (equivalent to Lasso).

    Parameters
    ----------
    alpha : float
        Regularisation strength (penalty multiplier for ``celer.GroupLasso``).
    max_iter : int
        Maximum number of subproblem iterations in the celer solver.
    tol : float
        Solver convergence tolerance (duality-gap based).
    fit_intercept : bool
        Whether to fit an intercept term.

    Attributes
    ----------
    groups_ : list[list[int]]
        Feature-index groups passed to ``celer.GroupLasso``.  Set after ``fit``.
    estimator_ : celer.GroupLasso
        Fitted celer estimator.  Set after ``fit``.
    coef_ : np.ndarray
        Coefficient vector, shape ``(n_features,)``.  Derived from ``estimator_``.
    intercept_ : float
        Intercept term.  Derived from ``estimator_``.
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _groups_from_transformer(transformer):

        groups = []
        col = 0

        for label in transformer.region_order_:
            size = transformer.region_sizes_[label]

            # nodes × features
            k = size * transformer.n_node_features_

            groups.append(list(range(col, col + k)))
            col += k

        return groups

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GroupLassoRegressor":
        """Fit the Group Lasso estimator.

        Parameters
        ----------
        X : np.ndarray of shape (n_subjects, n_features)
            Region-wise flattened feature matrix.
        y : np.ndarray of shape (n_subjects,)
            Regression targets.

        Returns
        -------
        self
        """
        transformer = getattr(self, "_transformer", None)
        if transformer is not None:
            try:
                check_is_fitted(transformer)
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
        """Return predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_subjects, n_features)

        Returns
        -------
        np.ndarray of shape (n_subjects,)
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the Group-Lasso-masked feature matrix.

        Used in the classification pipeline where this step sits between
        :class:`RegionFeatureTransformer` and ``LogisticRegression``.  Columns
        belonging to groups whose coefficient vector is entirely zero are
        **zeroed out** (not dropped), so the downstream estimator always
        receives a matrix of consistent shape.

        Parameters
        ----------
        X : np.ndarray of shape (n_subjects, n_features)

        Returns
        -------
        np.ndarray of shape (n_subjects, n_features)
            ``X`` with zeroed columns for inactive groups.
        """
        check_is_fitted(self, "estimator_")
        coef = self.estimator_.coef_          # (n_features,)
        mask = coef != 0.0                    # True for active features
        return X * mask[np.newaxis, :]        # broadcast: zero inactive cols

    @property
    def coef_(self) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.coef_

    @property
    def intercept_(self) -> float:
        check_is_fitted(self, "estimator_")
        return self.estimator_.intercept_


# ---------------------------------------------------------------------------
# _GroupLassoPipeline
# ---------------------------------------------------------------------------


class _GroupLassoPipeline(Pipeline):
    """Pipeline subclass that wires the fitted feature transformer into the
    Group Lasso regressor before calling the regressor's ``fit``.

    Standard :class:`sklearn.pipeline.Pipeline` passes the transformed ``X``
    to the final estimator's ``fit``, but does not expose the upstream
    transformer to it.  This subclass intercepts ``fit`` to inject the fitted
    :class:`RegionFeatureTransformer` instance into ``GroupLassoRegressor._transformer``
    so that the regressor can build the correct group structure.

    ``sklearn.base.clone`` preserves the subclass (it calls
    ``estimator.__class__(**estimator.get_params())``), so each cross-validation
    fold receives its own fresh :class:`_GroupLassoPipeline` that re-runs this
    injection after its own ``RegionFeatureTransformer.fit``.
    """

    def fit(self, X, y=None, **params):  # type: ignore[override]
        """Fit all pipeline steps in order.

        Whenever a :class:`GroupLassoRegressor` step is encountered, the
        already-fitted :class:`RegionFeatureTransformer` (first step) is injected
        into it so that it can build the correct group structure.

        For regression the pipeline is::

            RegionFeatureTransformer → StandardScaler → GroupLassoRegressor

        For classification the pipeline is::

            RegionFeatureTransformer → StandardScaler → GroupLassoRegressor → LogisticRegression

        In the classification case ``GroupLassoRegressor`` acts as a
        transformer: after fitting, its ``transform`` method is called to
        produce a sparse feature matrix (zeroed-out region columns removed)
        which is then passed to the downstream ``LogisticRegression``.
        """
        feature_transformer = self.named_steps.get("region_features")

        Xt = X
        for name, step in self.steps:
            if step is None or step == "passthrough":
                continue

            # Inject the fitted feature transformer before the Group Lasso fits.
            if isinstance(step, GroupLassoRegressor) and feature_transformer is not None:
                step._transformer = feature_transformer

            is_last = (name == self.steps[-1][0])

            if is_last:
                # Final step: always call fit (not fit_transform).
                step.fit(Xt, y)
            else:
                if hasattr(step, "fit_transform"):
                    Xt = step.fit_transform(Xt, y)
                else:
                    step.fit(Xt, y)
                    Xt = step.transform(Xt)

        return self

    def predict(self, X, **predict_params):  # type: ignore[override]
        """Run inference through the full pipeline."""
        Xt = X
        for _, transformer in self.steps[:-1]:
            if transformer is None or transformer == "passthrough":
                continue
            Xt = transformer.transform(Xt)
        return self.steps[-1][1].predict(Xt)


# ---------------------------------------------------------------------------
# RegionGroupLassoModel
# ---------------------------------------------------------------------------


class RegionGroupLassoModel(SklearnModel):
    """Region-feature + Group Lasso model for surface-mesh data.

    Combines :class:`RegionFeatureTransformer`
    with :class:`GroupLassoRegressor` in a single sklearn pipeline wrapped by
    ``GridSearchCV``.

    The mesh-to-array conversion is handled transparently by the transformer,
    and the Group Lasso penalty groups each region's flattened node features so
    that the solver can zero out entire brain regions jointly.

    Parameters forwarded to ``_build_model``
    -----------------------------------------
    prediction_task : str
        ``"regression"`` or ``"binary_classification"``.
    random_state : int
        Unused; present for interface consistency with other models.

    Pipeline
    --------
    Regression: ``RegionFeatureTransformer`` → ``StandardScaler`` → ``GroupLassoRegressor``
    Classification: ``RegionFeatureTransformer`` → ``StandardScaler`` → ``GroupLassoRegressor`` → ``LogisticRegression``

    Grid search
    -----------
    ``group_lasso__alpha``:       log-spaced grid [1e-3 … 1e3]
    """

    data_type: str = "mesh"

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", "regression")
        self.output_dim = 1
        cv = kwargs.get("cv", 5)
        n_jobs = kwargs.get("n_jobs", 1)
        verbose = kwargs.get("verbose", 1)
        reg_alpha_grid = kwargs.get("group_lasso_alpha_grid", np.logspace(-5, 5, 10))
        cls_alpha_grid = kwargs.get(
            "group_lasso_alpha_grid_classification", np.logspace(-5, 5, 10)
        )

        if self.prediction_task == "regression":

            pipeline = _GroupLassoPipeline(
                [
                    ("region_features", RegionFeatureTransformer()),
                    ("scaler", StandardScaler(copy=False)),
                    ("group_lasso", GroupLassoRegressor()),
                ]
            )

            param_grid = {
                "group_lasso__alpha": reg_alpha_grid,
            }

            scoring = "neg_mean_absolute_error"

        elif self.prediction_task == "binary_classification":

            pipeline = _GroupLassoPipeline(
                [
                    ("region_features", RegionFeatureTransformer()),
                    ("scaler", StandardScaler(copy=False)),
                    ("group_lasso", GroupLassoRegressor()),
                    ("classifier", LogisticRegression(max_iter=5000)),
                ]
            )

            param_grid = {
                "group_lasso__alpha": cls_alpha_grid,
                "classifier__C": np.logspace(-5, 5, 10),
            }

            scoring = "balanced_accuracy"

        else:
            raise ValueError(f"Unknown prediction_task '{self.prediction_task}'")

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Override fit / predict to accept mesh lists
    # ------------------------------------------------------------------

    def fit(self, X, y: np.ndarray):
        """Fit the pipeline to a list of mesh dicts.

        Parameters
        ----------
        X : list[dict]
            List of mesh dicts with keys ``"node_features"`` and
            ``"parcel_labels"``.
        y : np.ndarray of shape (n_subjects,)
            Regression targets.

        Returns
        -------
        self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Return predictions for *X*.

        Parameters
        ----------
        X : list[dict]
            List of mesh dicts (same format as ``fit``).

        Returns
        -------
        np.ndarray of shape (n_subjects,)
        """
        return self.model.predict(X)

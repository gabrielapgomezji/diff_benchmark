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

from collections.abc import Sequence
import os
import cvxpy as cp
import numpy as np
from skglm.datafits import Quadratic
try:
    from skglm.datafits import QuadraticGroup
except ImportError:  # skglm compatibility fallback
    QuadraticGroup = None
from skglm.penalties import WeightedGroupL2
try:
    from skglm.solvers import GroupBCD
except ImportError:  # skglm compatibility fallback
    GroupBCD = None
from skglm.solvers import AndersonCD
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from diff_benchmark.models.mesh_models.region_feature_extractor import RegionFeatureExtractor
from diff_benchmark.models.mesh_models.skglm_compat import CompatGeneralizedLinearEstimator
from diff_benchmark.models.utils_models.trainer import SklearnModel


def _mem_profile_enabled() -> bool:
    return str(os.getenv("DIFF_BENCHMARK_MEM_PROFILE", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _describe_array(x) -> str:
    if isinstance(x, np.ndarray):
        mib = x.nbytes / (1024.0 * 1024.0)
        return f"ndarray shape={x.shape} dtype={x.dtype} size_mib={mib:.2f}"
    return f"type={type(x).__name__}"


def _describe_mesh_list(x) -> str:
    if not isinstance(x, list):
        return f"type={type(x).__name__}"
    if not x:
        return "list(len=0)"
    first = x[0]
    if not isinstance(first, dict):
        return f"list(len={len(x)}, first_type={type(first).__name__})"
    key_bits = []
    for key, value in first.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            key_bits.append(f"{key}:shape={tuple(shape)}")
        else:
            key_bits.append(f"{key}:type={type(value).__name__}")
    return f"mesh_list len={len(x)} first_keys=[{', '.join(key_bits)}]"

# ---------------------------------------------------------------------------
# Region Transformer
# ---------------------------------------------------------------------------
RegionFeatureTransformer = RegionFeatureExtractor
    
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
            k = transformer.region_feature_widths_[label]

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
        if _mem_profile_enabled():
            print(f"[MEM] GroupLassoRegressor.fit X: {_describe_array(X)}")

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
        if GroupBCD is not None and QuadraticGroup is not None:
            solver = GroupBCD(
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
            )
            datafit = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
        else:
            solver = AndersonCD(
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
            )
            datafit = Quadratic()
        self.estimator_ = CompatGeneralizedLinearEstimator(
            datafit=datafit,
            penalty=penalty,
            solver=solver,
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


class FusedSparseGroupLassoRegressor(GroupLassoRegressor):
    r"""Composite-penalty regressor with group, sparse and fused terms.

    Objective minimized:

    .. math::

        \frac{1}{2n}\|y - Xw - b\|_2^2
        + \lambda_1 \sum_r \|w_r\|_2
        + \lambda_2 \|w\|_1
        + \lambda_3 \sum_{(r,s)\in E} \alpha_{rs}\|w_r - w_s\|_1

    where ``w_r`` denotes coefficients of region/group ``r`` and ``E`` is an
    adjacency graph over regions.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1e-3,
        lambda3: float = 1e-2,
        fused_edges: Sequence[tuple[int, int]] | None = None,
        fused_edge_weights: Sequence[float] | None = None,
        fused_graph_mode: str = "chain",
        fused_k_neighbors: int = 3,
        fused_use_distance_weights: bool = True,
        fit_intercept: bool = True,
        solver: str = "SCS",
        max_iter: int = 5000,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(
            alpha=lambda1,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
        )
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.fused_edges = fused_edges
        self.fused_edge_weights = fused_edge_weights
        self.fused_graph_mode = fused_graph_mode
        self.fused_k_neighbors = fused_k_neighbors
        self.fused_use_distance_weights = fused_use_distance_weights
        self.solver = solver

    @staticmethod
    def _slice_from_group(group: list[int]) -> slice:
        return slice(group[0], group[-1] + 1)

    @staticmethod
    def _fused_pair_expr(
        w: cp.Variable,
        g_r: list[int],
        g_s: list[int],
    ):
        """Return L1 distance expression between possibly different-width groups.

        For unequal widths this implements a zero-padding interpretation:
        ``||pad(w_r) - pad(w_s)||_1``.
        """
        wr = w[FusedSparseGroupLassoRegressor._slice_from_group(g_r)]
        ws = w[FusedSparseGroupLassoRegressor._slice_from_group(g_s)]
        nr = len(g_r)
        ns = len(g_s)
        nmin = min(nr, ns)

        expr = 0
        if nmin > 0:
            expr += cp.norm1(wr[:nmin] - ws[:nmin])
        if nr > nmin:
            expr += cp.norm1(wr[nmin:])
        if ns > nmin:
            expr += cp.norm1(ws[nmin:])
        return expr

    @staticmethod
    def _default_chain_edges(n_groups: int) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(n_groups - 1)]

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def _distance_graph_from_meshes(
        self,
        meshes,
        transformer,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        if meshes is None or len(meshes) == 0:
            raise ValueError("No mesh data available for distance graph construction.")

        region_labels = [int(r) for r in transformer.region_order_]
        n_groups = len(region_labels)
        if n_groups <= 1:
            return [], []

        centroid_sum = {r: np.zeros(3, dtype=float) for r in region_labels}
        centroid_count = {r: 0 for r in region_labels}

        for mesh in meshes:
            if "vertices" not in mesh or "parcel_labels" not in mesh:
                raise ValueError(
                    "Distance-based fused graph requires mesh keys 'vertices' and "
                    "'parcel_labels'."
                )
            vertices = self._to_numpy(mesh["vertices"])
            parcel_labels = self._to_numpy(mesh["parcel_labels"]).reshape(-1)

            for r in region_labels:
                mask = parcel_labels == r
                if np.any(mask):
                    centroid_sum[r] += vertices[mask].mean(axis=0)
                    centroid_count[r] += 1

        centroids = np.zeros((n_groups, 3), dtype=float)
        for i, r in enumerate(region_labels):
            if centroid_count[r] == 0:
                raise ValueError(
                    f"Region {r} not found in training meshes; cannot build "
                    "distance-based fused graph."
                )
            centroids[i] = centroid_sum[r] / centroid_count[r]

        dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)

        k = int(self.fused_k_neighbors)
        if k < 1:
            raise ValueError(f"fused_k_neighbors must be >= 1, got {k}.")
        k = min(k, n_groups - 1)

        edge_set: set[tuple[int, int]] = set()
        for i in range(n_groups):
            nbr_idx = np.argpartition(dmat[i], kth=k - 1)[:k]
            for j in nbr_idx.tolist():
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                edge_set.add((a, b))

        edges = sorted(edge_set)
        if self.fused_use_distance_weights:
            eps = 1e-8
            weights = [1.0 / (float(dmat[i, j]) + eps) for i, j in edges]
        else:
            weights = [1.0] * len(edges)
        return edges, weights

    def _resolve_edges_and_weights(
        self,
        transformer,
        n_groups: int,
        raw_meshes=None,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        if self.fused_graph_mode not in {"chain", "distance", "manual"}:
            raise ValueError(
                "fused_graph_mode must be one of {'chain', 'distance', 'manual'}, "
                f"got '{self.fused_graph_mode}'."
            )

        label_to_pos = {}
        region_labels: list[int] | None = None
        if transformer is not None and hasattr(transformer, "region_order_"):
            region_labels = [int(x) for x in transformer.region_order_]
            label_to_pos = {lab: i for i, lab in enumerate(region_labels)}

        if self.fused_graph_mode == "distance" and self.fused_edges is None:
            return self._distance_graph_from_meshes(raw_meshes, transformer)

        if self.fused_graph_mode == "manual" and self.fused_edges is None:
            raise ValueError(
                "fused_graph_mode='manual' requires fused_edges to be provided."
            )

        if self.fused_edges is None:
            edges = self._default_chain_edges(n_groups)
        else:
            edges = []
            for a, b in self.fused_edges:
                ai = label_to_pos.get(int(a), int(a))
                bi = label_to_pos.get(int(b), int(b))
                if not (0 <= ai < n_groups and 0 <= bi < n_groups):
                    raise ValueError(
                        "fused_edges contains invalid region/group indices: "
                        f"({a}, {b}) -> ({ai}, {bi}) with n_groups={n_groups}."
                    )
                if ai == bi:
                    continue
                edges.append((ai, bi))

        if self.fused_edge_weights is None:
            weights = [1.0] * len(edges)
        else:
            weights = [float(v) for v in self.fused_edge_weights]
            if len(weights) != len(edges):
                raise ValueError(
                    "fused_edge_weights must match fused_edges length; got "
                    f"{len(weights)} vs {len(edges)}."
                )
        return edges, weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FusedSparseGroupLassoRegressor":
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape

        if y.shape[0] != n_samples:
            raise ValueError(
                f"X has {n_samples} rows but y has {y.shape[0]} entries."
            )
        if self.lambda1 < 0 or self.lambda2 < 0 or self.lambda3 < 0:
            raise ValueError("lambda1/lambda2/lambda3 must be non-negative.")

        transformer = getattr(self, "_transformer", None)
        if transformer is not None:
            try:
                check_is_fitted(transformer)
                groups = self._groups_from_transformer(transformer)
            except Exception:
                groups = [[i] for i in range(n_features)]
        else:
            groups = [[i] for i in range(n_features)]

        covered = sum(len(g) for g in groups)
        if covered != n_features:
            raise ValueError(
                "Constructed region groups do not cover all features: "
                f"covered={covered}, n_features={n_features}."
            )

        raw_meshes = getattr(self, "_raw_meshes", None)
        edges, edge_weights = self._resolve_edges_and_weights(
            transformer,
            len(groups),
            raw_meshes=raw_meshes,
        )

        w = cp.Variable(n_features)
        b = cp.Variable() if self.fit_intercept else 0.0

        resid = y - (X @ w + b)
        data_term = 0.5 / n_samples * cp.sum_squares(resid)
        group_term = cp.sum([cp.norm2(w[self._slice_from_group(g)]) for g in groups])
        sparse_term = cp.norm1(w)
        fused_term = cp.sum(
            [
                alpha_rs * self._fused_pair_expr(w, groups[i], groups[j])
                for (i, j), alpha_rs in zip(edges, edge_weights)
            ]
        )

        objective = (
            data_term
            + self.lambda1 * group_term
            + self.lambda2 * sparse_term
            + self.lambda3 * fused_term
        )
        problem = cp.Problem(cp.Minimize(objective))

        solve_kwargs = {"solver": self.solver, "warm_start": True}
        if self.solver.upper() == "SCS":
            solve_kwargs.update({"max_iters": self.max_iter, "eps": self.tol})
        else:
            solve_kwargs.update({"max_iter": self.max_iter})

        problem.solve(**solve_kwargs)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(
                "FusedSparseGroupLassoRegressor optimisation failed with status "
                f"'{problem.status}'."
            )

        if w.value is None:
            raise RuntimeError("Solver did not return coefficient values.")

        self.groups_ = groups
        self.fused_edges_ = edges
        self.fused_edge_weights_ = edge_weights
        self.coef_value_ = np.asarray(w.value, dtype=float).reshape(-1)
        if self.fit_intercept:
            if b.value is None:
                raise RuntimeError("Solver did not return intercept value.")
            self.intercept_value_ = float(b.value)
        else:
            self.intercept_value_ = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_value_")
        X = np.asarray(X)
        return X @ self.coef_value_ + self.intercept_value_

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_value_")
        mask = self.coef_value_ != 0.0
        return X * mask[np.newaxis, :]

    @property
    def coef_(self) -> np.ndarray:
        check_is_fitted(self, "coef_value_")
        return self.coef_value_

    @property
    def intercept_(self) -> float:
        check_is_fitted(self, "intercept_value_")
        return self.intercept_value_


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
        mem_debug = _mem_profile_enabled()

        if mem_debug:
            print(f"[MEM] Pipeline.fit input: {_describe_mesh_list(X)}")

        Xt = X
        for name, step in self.steps:
            if step is None or step == "passthrough":
                continue

            # Inject the fitted feature transformer before the Group Lasso fits.
            if isinstance(
                step,
                (GroupLassoRegressor, FusedSparseGroupLassoRegressor),
            ) and feature_transformer is not None:
                step._transformer = feature_transformer
            if isinstance(step, FusedSparseGroupLassoRegressor):
                # Raw meshes are needed for optional distance-based graph wiring.
                step._raw_meshes = X

            is_last = (name == self.steps[-1][0])

            if is_last:
                # Final step: always call fit (not fit_transform).
                if mem_debug:
                    print(
                        f"[MEM] Step '{name}' fit input: "
                        f"{_describe_array(Xt) if isinstance(Xt, np.ndarray) else _describe_mesh_list(Xt)}"
                    )
                step.fit(Xt, y)
            else:
                if hasattr(step, "fit_transform"):
                    Xt = step.fit_transform(Xt, y)
                else:
                    step.fit(Xt, y)
                    Xt = step.transform(Xt)
                if mem_debug:
                    print(
                        f"[MEM] Step '{name}' output: "
                        f"{_describe_array(Xt) if isinstance(Xt, np.ndarray) else _describe_mesh_list(Xt)}"
                    )

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
        region_representation = kwargs.get("region_representation", "flatten")
        representation_cfg = kwargs.get(region_representation, {})
        if representation_cfg is None:
            representation_cfg = {}
        if not isinstance(representation_cfg, dict):
            raise ValueError(
                f"Expected kwargs['{region_representation}'] to be a dict, "
                f"got {type(representation_cfg).__name__}."
            )

        pca_n_components = kwargs.get(
            "pca_n_components",
            representation_cfg.get(
                "pca_n_components",
                representation_cfg.get("n_components_per_region", 3),
            ),
        )
        # Keep CV in-process for mesh-list inputs to avoid heavy loky serialization.
        n_jobs = kwargs["n_jobs"] #kwargs.get("n_jobs", 1)
        print(f"Building Group Lasso model with n_jobs={n_jobs} (CV parallelism disabled for mesh data).")
        verbose = 3 #kwargs.get("verbose", 3)
        
        alpha_grid = kwargs.get("group_lasso_alpha_grid", np.logspace(-3, 5, 10))
        rep_alpha = representation_cfg.get("group_lasso_alpha_grid", None)
        if rep_alpha is not None:
            alpha_grid = rep_alpha
        
        cls_C_grid = kwargs.get("classifier__C", np.logspace(-5, 5, 10))

        rep_C = representation_cfg.get("classifier__C", None)
        if rep_C is not None:
            cls_C_grid = rep_C
        
        if self.prediction_task == "regression":

            pipeline = _GroupLassoPipeline(
                [
                    (
                        "region_features",
                        RegionFeatureTransformer(
                            region_representation=region_representation,
                            pca_n_components=pca_n_components,
                        ),
                    ),
                    ("scaler", StandardScaler(copy=False)),
                    ("group_lasso", GroupLassoRegressor()),
                ]
            )

            param_grid = {
                "group_lasso__alpha": alpha_grid,
            }

            scoring = "neg_mean_absolute_error"

        elif self.prediction_task == "binary_classification":

            pipeline = _GroupLassoPipeline(
                [
                    (
                        "region_features",
                        RegionFeatureTransformer(
                            region_representation=region_representation,
                            pca_n_components=pca_n_components,
                        ),
                    ),
                    ("scaler", StandardScaler(copy=False)),
                    ("group_lasso", GroupLassoRegressor()),
                    ("classifier", LogisticRegression(max_iter=5000)),
                ]
            )

            param_grid = {
                "group_lasso__alpha": alpha_grid,
                "classifier__C": cls_C_grid,
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
            # pre_dispatch=1,
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


class RegionFusedSparseGroupLassoModel(SklearnModel):
    """Region-feature model with fused sparse group-lasso regularisation.

    Reuses the same mesh-feature extraction and sklearn pipeline structure as
    :class:`RegionGroupLassoModel`, but the regressor minimises a composite
    objective combining group sparsity, element-wise sparsity and graph-fusion.
    """

    data_type: str = "mesh"

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", "regression")
        self.output_dim = 1
        cv = kwargs.get("cv", 5)

        region_representation = kwargs.get("region_representation", "flatten")
        representation_cfg = kwargs.get(region_representation, {})
        if representation_cfg is None:
            representation_cfg = {}
        if not isinstance(representation_cfg, dict):
            raise ValueError(
                f"Expected kwargs['{region_representation}'] to be a dict, "
                f"got {type(representation_cfg).__name__}."
            )

        pca_n_components = kwargs.get(
            "pca_n_components",
            representation_cfg.get(
                "pca_n_components",
                representation_cfg.get("n_components_per_region", 3),
            ),
        )
        n_jobs = kwargs.get("n_jobs", 1)
        verbose = 3 #kwargs.get("verbose", 1)
        
        lambda1_grid = kwargs.get("fused_group_lambda1_grid", np.logspace(-4, 2, 7))
        lambda2_grid = kwargs.get("fused_group_lambda2_grid", np.logspace(-6, 0, 7))
        lambda3_grid = kwargs.get("fused_group_lambda3_grid", np.logspace(-6, 0, 7))

        rep_l1 = representation_cfg.get("fused_group_lambda1_grid", None)
        rep_l2 = representation_cfg.get("fused_group_lambda2_grid", None)
        rep_l3 = representation_cfg.get("fused_group_lambda3_grid", None)

        if rep_l1 is not None:
            lambda1_grid = rep_l1
        if rep_l2 is not None:
            lambda2_grid = rep_l2
        if rep_l3 is not None:
            lambda3_grid = rep_l3
            
        cls_C_grid = kwargs.get("classifier__C", np.logspace(-5, 5, 10))

        rep_C = representation_cfg.get("classifier__C", None)
        if rep_C is not None:
            cls_C_grid = rep_C

        fused_edges = kwargs.get(
            "fused_edges",
            representation_cfg.get("fused_edges", None),
        )
        fused_edge_weights = kwargs.get(
            "fused_edge_weights",
            representation_cfg.get("fused_edge_weights", None),
        )
        fused_graph_mode = kwargs.get(
            "fused_graph_mode",
            representation_cfg.get("fused_graph_mode", "chain"),
        )
        fused_k_neighbors = kwargs.get(
            "fused_k_neighbors",
            representation_cfg.get("fused_k_neighbors", 3),
        )
        fused_use_distance_weights = kwargs.get(
            "fused_use_distance_weights",
            representation_cfg.get("fused_use_distance_weights", True),
        )

        base_regressor = FusedSparseGroupLassoRegressor(
            fused_edges=fused_edges,
            fused_edge_weights=fused_edge_weights,
            fused_graph_mode=fused_graph_mode,
            fused_k_neighbors=fused_k_neighbors,
            fused_use_distance_weights=fused_use_distance_weights,
            fit_intercept=kwargs.get("fit_intercept", True),
            solver=kwargs.get("fused_solver", "SCS"),
            max_iter=kwargs.get("max_iter", 5000),
            tol=kwargs.get("tol", 1e-4),
        )

        if self.prediction_task == "regression":
            pipeline = _GroupLassoPipeline(
                [
                    (
                        "region_features",
                        RegionFeatureTransformer(
                            region_representation=region_representation,
                            pca_n_components=pca_n_components,
                        ),
                    ),
                    ("scaler", StandardScaler(copy=False)),
                    ("fused_group_lasso", base_regressor),
                ]
            )

            param_grid = {
                "fused_group_lasso__lambda1": lambda1_grid,
                "fused_group_lasso__lambda2": lambda2_grid,
                "fused_group_lasso__lambda3": lambda3_grid,
            }
            scoring = "neg_mean_absolute_error"

        elif self.prediction_task == "binary_classification":
            pipeline = _GroupLassoPipeline(
                [
                    (
                        "region_features",
                        RegionFeatureTransformer(
                            region_representation=region_representation,
                            pca_n_components=pca_n_components,
                        ),
                    ),
                    ("scaler", StandardScaler(copy=False)),
                    ("fused_group_lasso", base_regressor),
                    ("classifier", LogisticRegression(max_iter=5000)),
                ]
            )

            param_grid = {
                "fused_group_lasso__lambda1": lambda1_grid,
                "fused_group_lasso__lambda2": lambda2_grid,
                "fused_group_lasso__lambda3": lambda3_grid,
                "classifier__C": cls_C_grid,
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

    def fit(self, X, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

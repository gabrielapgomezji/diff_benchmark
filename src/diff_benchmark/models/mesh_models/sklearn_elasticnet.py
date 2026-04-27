from __future__ import annotations

import os
import numpy as np
from skglm import GeneralizedLinearEstimator
from skglm.datafits import QuadraticGroup
from skglm.penalties import WeightedL1GroupL2
from skglm.solvers import GroupBCD
from skglm.utils.data import grp_converter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from diff_benchmark.models.mesh_models.region_feature_extractor import RegionFeatureExtractor
from diff_benchmark.models.utils_models.trainer import SklearnModel


def _mem_profile_enabled() -> bool:
    return str(os.getenv("DIFF_BENCHMARK_MEM_PROFILE", "1")).lower() in {
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

    
class GroupElasticNetRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible Sparse Group Elastic Net regressor backed by skglm.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if _mem_profile_enabled():
            print(f"[MEM] GroupElasticNetRegressor.fit X: {_describe_array(X)}")

        transformer = getattr(self, "_transformer", None)
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError(
                f"l1_ratio must be in [0, 1], got {self.l1_ratio}."
            )

        if transformer is None:
            raise RuntimeError(
                "GroupElasticNetRegressor requires a fitted RegionFeatureExtractor "
                "to define region groups; _transformer was not set."
            )

        try:
            if not hasattr(transformer, "region_order_"):
                raise AttributeError("Missing required attribute 'region_order_'.")
            if not hasattr(transformer, "region_feature_widths_"):
                raise AttributeError(
                    "Missing required attribute 'region_feature_widths_'."
                )
            groups = []
            col = 0
            for label in transformer.region_order_:
                width = transformer.region_feature_widths_[label]
                if width <= 0:
                    raise ValueError(
                        f"Region '{label}' has non-positive feature width {width}."
                    )
                groups.append(list(range(col, col + width)))
                col += width
        except Exception as exc:
            raise RuntimeError(
                "Failed to construct region-based groups from RegionFeatureExtractor."
            ) from exc

        n_features = X.shape[1]
        if col != n_features:
            raise ValueError(
                "Mismatch between constructed grouped features and X columns: "
                f"groups cover {col} features but X has {n_features}."
            )

        grp_indices, grp_ptr = grp_converter(groups, n_features)
        grp_ptr = np.asarray(grp_ptr, dtype=np.int32)
        grp_indices = np.asarray(grp_indices, dtype=np.int32)

        self.groups_ = groups
        self.grp_ptr_ = grp_ptr
        self.grp_indices_ = grp_indices

        feature_strength = self.alpha * self.l1_ratio
        group_strength = self.alpha * (1.0 - self.l1_ratio)

        weights_features = np.full(n_features, feature_strength, dtype=np.float64)
        weights_groups = np.full(len(groups), group_strength, dtype=np.float64)

        penalty = WeightedL1GroupL2(
            alpha=1.0,
            weights_groups=weights_groups,
            weights_features=weights_features,
            grp_ptr=grp_ptr,
            grp_indices=grp_indices,
        )

        if self.fit_intercept:
            self._x_offset_ = X.mean(axis=0)
            self._y_offset_ = float(y.mean())
            X_fit = X - self._x_offset_
            y_fit = y - self._y_offset_
        else:
            self._x_offset_ = np.zeros(n_features, dtype=X.dtype)
            self._y_offset_ = 0.0
            X_fit = X
            y_fit = y

        solver = GroupBCD(
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=False,
            ws_strategy="fixpoint",
        )

        self.estimator_ = GeneralizedLinearEstimator(
            datafit=QuadraticGroup(grp_ptr, grp_indices),
            penalty=penalty,
            solver=solver,
        )
        self.estimator_.fit(X_fit, y_fit)

        self.intercept_value_ = (
            self._y_offset_ - float(self._x_offset_ @ self.estimator_.coef_)
        )

        coef = self.estimator_.coef_
        self.zero_group_mask_ = np.array(
            [np.all(coef[group] == 0.0) for group in self.groups_],
            dtype=bool,
        )
        self.partial_zero_active_group_mask_ = np.array(
            [
                np.any(coef[group] == 0.0) and np.any(coef[group] != 0.0)
                for group in self.groups_
            ],
            dtype=bool,
        )

        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return np.asarray(X) @ self.estimator_.coef_ + self.intercept_value_

    def transform(self, X):
        check_is_fitted(self, "estimator_")
        coef = self.estimator_.coef_
        mask = coef != 0.0
        return X * mask[np.newaxis, :]

    @property
    def coef_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.coef_

    @property
    def intercept_(self):
        check_is_fitted(self, "intercept_value_")
        return self.intercept_value_


class _GroupElasticNetPipeline(Pipeline):

    def fit(self, X, y=None, **params):  # type: ignore[override]
        feature_transformer = self.named_steps.get("region_features")
        mem_debug = _mem_profile_enabled()

        if mem_debug:
            print(f"[MEM] Pipeline.fit input: {_describe_mesh_list(X)}")

        Xt = X
        for name, step in self.steps:
            if step is None or step == "passthrough":
                continue

            if isinstance(step, GroupElasticNetRegressor) and feature_transformer is not None:
                step._transformer = feature_transformer

            is_last = name == self.steps[-1][0]

            if is_last:
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
        Xt = X
        for _, transformer in self.steps[:-1]:
            if transformer is None or transformer == "passthrough":
                continue
            Xt = transformer.transform(Xt)
        return self.steps[-1][1].predict(Xt)


class RegionElasticNetModel(SklearnModel):

    data_type: str = "mesh"

    def _build_model(self, **kwargs):
        self.prediction_task = kwargs.get("prediction_task", "regression")
        self.output_dim = 1

        cv = kwargs.get("cv", 5)
        region_representation = kwargs.get("region_representation", "flatten")
        if region_representation not in RegionFeatureExtractor.VALID_REPRESENTATIONS:
            raise ValueError(
                "region_representation must be one of "
                "['flatten', 'pca', 'mean_std', 'summary_stats', 'percentiles']"
            )
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
        verbose = kwargs.get("verbose", 1)

        alpha_grid = kwargs.get("elasticnet_alpha_grid", np.logspace(-5, 5, 10))
        l1_ratio_grid = kwargs.get(
            "elasticnet_l1_ratio_grid",
            [0.1, 0.3, 0.5, 0.7, 0.9],
        )
        cls_C_grid = kwargs.get("classifier__C", kwargs.get("classifier_C_grid", np.logspace(-5, 5, 10)))

        rep_alpha = representation_cfg.get("elasticnet_alpha_grid", None)
        rep_l1 = representation_cfg.get("elasticnet_l1_ratio_grid", None)
        rep_cls_C = representation_cfg.get("classifier__C", representation_cfg.get("classifier_C_grid", None))

        if rep_alpha is not None:
            alpha_grid = rep_alpha
        if rep_l1 is not None:
            l1_ratio_grid = rep_l1
        if rep_cls_C is not None:
            cls_C_grid = rep_cls_C

        if self.prediction_task == "regression":

            pipeline = _GroupElasticNetPipeline(
                [
                    (
                        "region_features",
                        RegionFeatureExtractor(
                            region_representation=region_representation,
                            pca_n_components=pca_n_components,
                        ),
                    ),
                    ("scaler", StandardScaler(copy=False)),
                    ("group_elastic_net", GroupElasticNetRegressor()),
                ]
            )

            param_grid = {
                "group_elastic_net__alpha": alpha_grid,
                "group_elastic_net__l1_ratio": l1_ratio_grid,
            }

            scoring = "neg_mean_absolute_error"

        elif self.prediction_task == "binary_classification":

            pipeline = _GroupElasticNetPipeline(
                [
                    (
                        "region_features",
                        RegionFeatureExtractor(
                            region_representation=region_representation,
                            pca_n_components=pca_n_components,
                        ),
                    ),
                    ("scaler", StandardScaler(copy=False)),
                    ("group_elastic_net", GroupElasticNetRegressor()),
                    ("classifier", LogisticRegression(max_iter=5000)),
                ]
            )

            param_grid = {
                "group_elastic_net__alpha": alpha_grid,
                "group_elastic_net__l1_ratio": l1_ratio_grid,
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
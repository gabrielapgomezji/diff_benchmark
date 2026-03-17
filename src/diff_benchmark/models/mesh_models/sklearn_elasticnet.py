from __future__ import annotations

import numpy as np
from skglm import GeneralizedLinearEstimator
try:
    from skglm.penalties import SparseGroupL1
except ImportError:  # skglm<=0.5 compatibility
    SparseGroupL1 = None
    from skglm.datafits import Quadratic
    from skglm.penalties import WeightedL1GroupL2
    from skglm.solvers import AndersonCD
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from diff_benchmark.models.mesh_models.region_feature_extractor import RegionFeatureExtractor
from diff_benchmark.models.utils_models.trainer import SklearnModel

    
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
        transformer = getattr(self, "_transformer", None)
        if transformer is not None:
            try:
                check_is_fitted(transformer)
                groups = []
                col = 0
                for label in transformer.region_order_:
                    width = transformer.region_feature_widths_[label]
                    groups.append(list(range(col, col + width)))
                    col += width
                self.groups_ = groups
            except Exception:
                self.groups_ = [[i] for i in range(X.shape[1])]
        else:
            self.groups_ = [[i] for i in range(X.shape[1])]

        if SparseGroupL1 is not None:
            penalty = SparseGroupL1(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                groups=self.groups_,
            )
        else:
            grp_indices = np.asarray(
                [idx for group in self.groups_ for idx in group],
                dtype=np.int32,
            )
            grp_sizes = np.asarray([len(group) for group in self.groups_], dtype=np.int32)
            grp_ptr = np.zeros(len(grp_sizes) + 1, dtype=np.int32)
            grp_ptr[1:] = np.cumsum(grp_sizes)

            weights_groups = np.full(
                len(self.groups_),
                fill_value=(1.0 - self.l1_ratio),
                dtype=float,
            )
            weights_features = np.full(
                X.shape[1],
                fill_value=self.l1_ratio,
                dtype=float,
            )

            penalty = WeightedL1GroupL2(
                alpha=self.alpha,
                weights_groups=weights_groups,
                weights_features=weights_features,
                grp_ptr=grp_ptr,
                grp_indices=grp_indices,
            )

        try:
            self.estimator_ = GeneralizedLinearEstimator(
                penalty=penalty,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        except TypeError:
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

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

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
        check_is_fitted(self, "estimator_")
        return self.estimator_.intercept_


class _GroupElasticNetPipeline(Pipeline):

    def fit(self, X, y=None, **params):  # type: ignore[override]
        feature_transformer = self.named_steps.get("region_features")

        Xt = X
        for name, step in self.steps:
            if step is None or step == "passthrough":
                continue

            if isinstance(step, GroupElasticNetRegressor) and feature_transformer is not None:
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

        reg_alpha_grid = kwargs.get("elasticnet_alpha_grid", np.logspace(-5, 5, 10))
        reg_l1_ratio_grid = kwargs.get(
            "elasticnet_l1_ratio_grid",
            [0.1, 0.3, 0.5, 0.7, 0.9],
        )
        cls_alpha_grid = kwargs.get(
            "elasticnet_alpha_grid_classification",
            np.logspace(-5, 5, 5),
        )
        cls_l1_ratio_grid = kwargs.get(
            "elasticnet_l1_ratio_grid_classification",
            [0.3, 0.5, 0.7],
        )
        cls_C_grid = kwargs.get("classifier_C_grid", np.logspace(-5, 5, 10))

        rep_reg_alpha = representation_cfg.get("elasticnet_alpha_grid", None)
        rep_reg_l1 = representation_cfg.get("elasticnet_l1_ratio_grid", None)
        rep_cls_alpha = representation_cfg.get(
            "elasticnet_alpha_grid_classification",
            None,
        )
        rep_cls_l1 = representation_cfg.get(
            "elasticnet_l1_ratio_grid_classification",
            None,
        )
        rep_cls_C = representation_cfg.get("classifier_C_grid", None)

        if rep_reg_alpha is not None:
            reg_alpha_grid = rep_reg_alpha
        if rep_reg_l1 is not None:
            reg_l1_ratio_grid = rep_reg_l1
        if rep_cls_alpha is not None:
            cls_alpha_grid = rep_cls_alpha
        elif rep_reg_alpha is not None:
            cls_alpha_grid = rep_reg_alpha
        if rep_cls_l1 is not None:
            cls_l1_ratio_grid = rep_cls_l1
        elif rep_reg_l1 is not None:
            cls_l1_ratio_grid = rep_reg_l1
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
                "group_elastic_net__alpha": reg_alpha_grid,
                "group_elastic_net__l1_ratio": reg_l1_ratio_grid,
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
                "group_elastic_net__alpha": cls_alpha_grid,
                "group_elastic_net__l1_ratio": cls_l1_ratio_grid,
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
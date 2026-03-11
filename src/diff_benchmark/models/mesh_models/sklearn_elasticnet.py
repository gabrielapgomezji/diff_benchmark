from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from diff_benchmark.models.utils_models.trainer import SklearnModel

from celer import ElasticNet

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
    
    
class ElasticNetRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible Elastic Net regressor backed by celer.ElasticNet.
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

        self.estimator_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )

        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    @property
    def coef_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.coef_

    @property
    def intercept_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.intercept_
    
class RegionElasticNetModel(SklearnModel):

    data_type: str = "mesh"

    def _build_model(self, **kwargs):

        self.prediction_task = kwargs.get("prediction_task", "regression")

        if self.prediction_task == "regression":

            pipeline = Pipeline(
                [
                    ("region_features", RegionFeatureTransformer()),
                    ("elastic_net", ElasticNetRegressor()),
                ]
            )

            param_grid = {
                "elastic_net__alpha": np.logspace(-4, 2, 10),
                "elastic_net__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            }

            scoring = "neg_mean_absolute_error"

        elif self.prediction_task == "binary_classification":

            pipeline = Pipeline(
                [
                    ("region_features", RegionFeatureTransformer()),
                    ("elastic_net", ElasticNetRegressor()),
                    ("classifier", LogisticRegression(max_iter=5000)),
                ]
            )

            param_grid = {
                "elastic_net__alpha": np.logspace(-4, 2, 8),
                "elastic_net__l1_ratio": [0.3, 0.5, 0.7],
                "classifier__C": np.logspace(-3, 3, 7),
            }

            scoring = "balanced_accuracy"

        else:
            raise ValueError(f"Unknown prediction_task '{self.prediction_task}'")

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
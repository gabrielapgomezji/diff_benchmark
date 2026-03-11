from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diff_benchmark.models.utils_models.trainer import SklearnModel


class DummyRegressorModel(SklearnModel):
    """Dummy regressor baseline (predicts mean)."""

    data_type = "array"
    prediction_task = "regression"
    output_dim = 1

    def _build_model(self, **kwargs):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", DummyRegressor(strategy="mean")),
            ]
        )
        return pipeline


class DummyClassifierModel(SklearnModel):
    """Dummy classifier baseline (predicts most frequent class)."""

    data_type = "array"
    prediction_task = "binary_classification"
    output_dim = 1  # for single-label classification

    def _build_model(self, **kwargs):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        )
        return pipeline

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diff_benchmark.models.utils_models.trainer import SklearnModel


class DummyRegressorModel(SklearnModel):
    """
    Unified dummy regressor compatible with the diff_benchmark pipeline.
    """

    data_type = "array"  # so trainer knows it's array data
    prediction_task = "regression"
    output_dim = 1

    def _build_model(self, **kwargs):
        # Wrap in a simple pipeline (optional: scaling)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", DummyRegressor(strategy="mean")),
            ]
        )
        return pipeline


class DummyClassifierModel(SklearnModel):
    """
    Unified dummy classifier compatible with the diff_benchmark pipeline.
    """

    data_type = "array"
    prediction_task = "classification"
    #  prediction_task = "binary_classification"
    output_dim = 1  # for single-label classification

    def _build_model(self, **kwargs):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        )
        return pipeline

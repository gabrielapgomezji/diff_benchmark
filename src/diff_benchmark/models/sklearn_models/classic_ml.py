from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# from sklearn.svm import SVC, SVR
from sklearn.svm import LinearSVC, LinearSVR

from diff_benchmark.models.utils_models.trainer import SklearnModel


# ---------------------------------------------------------------------------
# Private helpers – shared across model classes
# ---------------------------------------------------------------------------


def _rf_head_and_scoring(prediction_task: str, random_state: int = 42):
    """Return the appropriate Random Forest estimator and GridSearchCV scoring string."""
    if prediction_task == "binary_classification":
        return RandomForestClassifier(random_state=random_state), "balanced_accuracy"
    return RandomForestRegressor(random_state=random_state), "neg_mean_absolute_error"


def _svm_head_and_scoring(prediction_task: str):
    """Return the appropriate SVM estimator and GridSearchCV scoring string."""
    if prediction_task == "binary_classification":
        return LinearSVC(), "balanced_accuracy"
    return LinearSVR(), "neg_mean_absolute_error"


class PCARandomForestModel(SklearnModel):
    """
    PCARandomForestModel combines PCA for dimensionality reduction
    with a Random Forest classifier for classification tasks.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        random_state = kwargs.get("random_state", 42)
        rf_head, scoring = _rf_head_and_scoring(self.prediction_task, random_state)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("rf", rf_head),
            ]
        )

        param_grid = {
            "pca__n_components": [10, 20, 30, 50, 60, 75, 100, 400], 
            "rf__n_estimators": [200], 
            "rf__max_depth": [5, 10, 15], 
            "rf__max_features": ["sqrt", 0.3],
            "rf__min_samples_leaf": [5, 10, 20], 
        }

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )


class RandomForestModel(SklearnModel):
    """
    RandomForestModel uses a Random Forest classifier or regressor
    depending on the prediction task.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        random_state = kwargs.get("random_state", 42)
        rf_head, scoring = _rf_head_and_scoring(self.prediction_task, random_state)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", rf_head),
            ]
        )

        param_grid = {
            "rf__n_estimators": [200], 
            "rf__max_depth": [5, 10, 15], 
            "rf__max_features": ["sqrt", 0.3],
            "rf__min_samples_leaf": [5, 10, 20],
        }

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )


class PCASVMModel(SklearnModel):
    """
    PCASVMModel combines PCA for dimensionality reduction
    with a Support Vector Machine classifier or regressor.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        svm_head, scoring = _svm_head_and_scoring(self.prediction_task)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("svm", svm_head),
            ]
        )

        param_grid = {
            "pca__n_components": [50],
            "svm__C": [0.1, 1], 
        }

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )


class SVMModel(SklearnModel):
    """
    SVMModel uses a Support Vector Machine classifier or regressor
    depending on the prediction task.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        svm_head, scoring = _svm_head_and_scoring(self.prediction_task)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", svm_head),
            ]
        )

        param_grid = {
            "svm__C": [0.1, 1],
        }

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )


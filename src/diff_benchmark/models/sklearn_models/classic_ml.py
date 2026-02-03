from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from diff_benchmark.models.utils_models.trainer import SklearnModel


class PCARandomForestModel(SklearnModel):
    """
    PCARandomForestModel combines PCA for dimensionality reduction
    with a Random Forest classifier for classification tasks.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        # Define pipeline: standardization -> PCA -> RandomForest
        # Choose RF head depending on task
        if self.prediction_task == "binary_classification":
            rf_head = RandomForestClassifier(random_state=42)
            scoring = "accuracy"
        else:  # if self.prediction_task == "regression":
            rf_head = RandomForestRegressor(random_state=42)
            scoring = "neg_mean_squared_error"  # scikit-learn convention

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("rf", rf_head),
            ]
        )

        # Define hyperparameter grid
        param_grid = {
            "pca__n_components": [50, 100, 400],
            "rf__n_estimators": [100, 200, 500],
            "rf__max_depth": [None, 10, 30],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
        }

        # Define GridSearchCV
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
        # Define pipeline: standardization -> PCA -> RandomForest
        # Choose RF head depending on task
        if self.prediction_task == "binary_classification":
            rf_head = RandomForestClassifier(random_state=42)
            scoring = "accuracy"
        else:  # if self.prediction_task == "regression":
            rf_head = RandomForestRegressor(random_state=42)
            scoring = "neg_mean_squared_error"  # scikit-learn convention

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", rf_head),
            ]
        )

        # Define hyperparameter grid
        param_grid = {
            "rf__n_estimators": [100, 200, 500],
            "rf__max_depth": [None, 10, 30],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
        }

        # Define GridSearchCV
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
    with a Support Vector Machine classifier.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        if self.prediction_task == "binary_classification":
            svm_head = SVC(probability=True)
            scoring = "accuracy"
            svm_gamma = ["scale", "auto"]

        else:  # regression
            svm_head = SVR()
            scoring = "neg_mean_squared_error"
            svm_gamma = ["scale"]

        # Define pipeline: scaling -> PCA -> SVM
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("svm", svm_head),
            ]
        )

        # Define hyperparameter grid
        param_grid = {
            "pca__n_components": [50, 100, 400],
            "svm__C": [0.1, 1, 10, 100],
            "svm__kernel": ["linear", "rbf"],
            "svm__gamma": svm_gamma,
        }

        # Define GridSearchCV
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
        if self.prediction_task == "binary_classification":
            svm_head = SVC(probability=True)
            scoring = "accuracy"
            svm_gamma = ["scale", "auto"]

        else:  # regression
            svm_head = SVR()
            scoring = "neg_mean_squared_error"
            svm_gamma = ["scale"]

        # Define pipeline: scaling -> PCA -> SVM
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", svm_head),
            ]
        )

        # Define hyperparameter grid
        param_grid = {
            "svm__C": [0.1, 1, 10, 100],
            "svm__kernel": ["linear", "rbf"],
            "svm__gamma": svm_gamma,
        }

        # Define GridSearchCV
        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
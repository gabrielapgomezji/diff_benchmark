import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from torch.utils.data import DataLoader

from diff_benchmark.models.base import NumpyAbstractModel


class PCARandomForestModel(NumpyAbstractModel):
    """
    PCARandomForestModel combines PCA for dimensionality reduction
    with a Random Forest classifier for classification tasks.
    """

    data_type = "array"

    def __init__(self, **kwargs):
        self.prediction_task = kwargs.get("prediction_task", None)
        # Define pipeline: standardization -> PCA -> RandomForest
        # Choose RF head depending on task
        if self.prediction_task == "classification":
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
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

    def fit(self, dataloader: DataLoader):
        """Fit PCA + RandomForest using grid search.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with training data.
        """
        assert self.prediction_task in [
            "classification",
            "regression",
        ], f"prediction_task must be set before calling fit(). Got {self.prediction_task}"

        features, targets = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        self.model.fit(features_reshaped, targets.flatten())
        # print("Best params found:", self.model.best_params_)

    def predict(self, dataloader: DataLoader):
        """Predict using the best pipeline from grid search.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with data to predict."""
        features, _ = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        return self.model.predict(features_reshaped)


class PCASVMModel(NumpyAbstractModel):
    """
    PCASVMModel combines PCA for dimensionality reduction
    with a Support Vector Machine classifier.
    """

    data_type = "array"

    def __init__(self, **kwargs):
        self.prediction_task = kwargs.get("prediction_task", None)
        if self.prediction_task == "classification":
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
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

    def fit(self, dataloader: DataLoader):
        """Fit PCA + SVM using grid search.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with training data.
        """
        features, targets = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        self.model.fit(features_reshaped, targets.flatten())
        # print("Best params found:", self.model.best_params_)

    def predict(self, dataloader: DataLoader):
        """Predict class labels using the trained SVM model.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with data to predict.
        """
        features, _ = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        return self.model.predict(features_reshaped)

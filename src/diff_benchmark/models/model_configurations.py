import hashlib
import json

from diff_benchmark.models.cca import CanonicalCorrelationRegressor
from diff_benchmark.models.cnn import ResNet3SliceModel
from diff_benchmark.models import cnn_with_base
from diff_benchmark.models.cnn_medicalnet import ResNet3DModel
from diff_benchmark.models.dummy import DummyClassifier
from diff_benchmark.models.logistic_regression import (
    LogisticRegressionModel,
    PCALogisticRegressionModel,
)
from diff_benchmark.models.classic_ml import PCARandomForestModel, PCASVMModel

# from diff_benchmark.models.mlp import MLPClassifier


def make_run_id(name, params):
    """
    Generates a unique run identifier based on the provided name and parameters.
    Parameters:
        name (str): The name associated with the run.
        params (dict): A dictionary of parameters that will be used to generate the run ID.
    Returns:
        str: A unique run ID formatted as 'name_hash', where 'hash' is the first 8 characters
             of the MD5 hash of the sorted parameters.
    """

    # Sort params to keep consistency
    params_str = json.dumps(params, sort_keys=True)
    # Hash to avoid overly long filenames
    run_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"{name}_{run_hash}"


def get_model(name: str, config: dict):
    """
    Get a machine learning model instance based on the specified name and configuration.
    Parameters:
        name (str): The name of the model to retrieve. Supported names include:
                    - "cca": Canonical Correlation Regressor
                    - "dummy_classifier": Dummy Classifier
                    - "mlp_classifier": Multi-Layer Perceptron Classifier
                    - "logistic_regression": PCA Logistic Regression Model
        config (dict): A dictionary containing configuration parameters for the model.
                       The required keys depend on the model name:
                       - For "cca": "n_components"
                       - For "mlp_classifier": "input_dim", "hidden_layers", "output_dim",
                         "learning_rate", "dropout_rate", "epochs"
                       - For "logistic_regression": "n_components"
    Returns:
        An instance of the specified model.
    Raises:
        ValueError: If the provided model name is not recognized.
    """

    name = name.lower()

    if name == "cca":
        return CanonicalCorrelationRegressor(n_components=config["n_components"])

    if name == "dummy_classifier":
        return DummyClassifier()

    # if name == "mlp_classifier":
    #     return MLPClassifier(
    #         input_dim=config["input_dim"],
    #         hidden_layers=config["hidden_layers"],
    #         output_dim=config["output_dim"],
    #         learning_rate=config["learning_rate"],
    #         dropout_rate=config["dropout_rate"],
    #         epochs=config["epochs"],
    #     )

    if name == "pca_logistic":
        return PCALogisticRegressionModel()

    if name == "logistic_regression":
        return LogisticRegressionModel()
    
    if name == "pca_forest":
        return PCARandomForestModel()
    
    if name == "pca_svm":
        return PCASVMModel()
    if name == "2dcnn":
        # return ResNet3SliceModel(input_slices=config.get("input_slices", 145), num_classes=config.get("num_classes", 2), device=config.get("device", "cuda"))
        return ResNet3SliceModel(**config)
    
    if name == "2dcnn_lite":
        return cnn_with_base.ResNet3SliceModel(**config)

    if name == "3dcnn_medicalnet":
        return ResNet3DModel(**config)

    # elif name == "other_model":
    #     return OtherModelClass(param1=config["param1"], ...)

    raise ValueError(f"Unknown model name: {name}")

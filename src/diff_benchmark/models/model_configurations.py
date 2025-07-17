from diff_benchmark.models.cca import CanonicalCorrelationRegressor
from diff_benchmark.models.dummy import DummyClassifier
from diff_benchmark.models.logistic_regression import PCALogisticRegressionModel
from diff_benchmark.models.mlp import MLPClassifier


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

    elif name == "dummy_classifier":
        return DummyClassifier()

    elif name == "mlp_classifier":
        return MLPClassifier(
            input_dim=config["input_dim"],
            hidden_layers=config["hidden_layers"],
            output_dim=config["output_dim"],
            learning_rate=config["learning_rate"],
            dropout_rate=config["dropout_rate"],
            epochs=config["epochs"],
        )

    elif name == "logistic_regression":
        return PCALogisticRegressionModel(n_components=config["n_components"])
    # elif name == "other_model":
    #     return OtherModelClass(param1=config["param1"], ...)

    else:
        raise ValueError(f"Unknown model name: {name}")

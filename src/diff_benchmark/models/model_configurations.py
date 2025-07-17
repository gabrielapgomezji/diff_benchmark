from diff_benchmark.models.cca import CanonicalCorrelationRegressor
from diff_benchmark.models.dummy import DummyClassifier


def get_model(name: str, config: dict):
    name = name.lower()

    if name == "cca":
        return CanonicalCorrelationRegressor(n_components=config["n_components"])
    
    elif name == "dummy_classifier":
        return DummyClassifier()

    elif name == "mlp_classifier":
        from diff_benchmark.models.mlp import MLPClassifier
        return MLPClassifier(
            input_dim=config["input_dim"],
            hidden_layers=config["hidden_layers"],
            output_dim=config["output_dim"],
            learning_rate=config["learning_rate"],
            dropout_rate=config["dropout_rate"],
            epochs=config["epochs"],
        )
    
    elif name == "logistic_regression":
        from diff_benchmark.models.logistic_regression import PCALogisticRegressionModel
        return PCALogisticRegressionModel(n_components=config["n_components"])
    # elif name == "other_model":
    #     return OtherModelClass(param1=config["param1"], ...)

    else:
        raise ValueError(f"Unknown model name: {name}")

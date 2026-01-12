import hashlib
import json

from torch import nn

from diff_benchmark.models.sklearn_models.classic_ml import (
    PCARandomForestModel,
    PCASVMModel,
)
from diff_benchmark.models.sklearn_models.dummy import (
    DummyClassifierModel,
    DummyRegressorModel,
)
from diff_benchmark.models.sklearn_models.logistic_regression import (
    LinearModel,
    PCALinearModel,
)
from diff_benchmark.models.torch_models.cnn import ResNet3SliceMultihead
from diff_benchmark.models.torch_models.medicalnet import MedicalNet
from diff_benchmark.models.utils_models.prediction_head import build_prediction_head
from diff_benchmark.models.utils_models.trainer import (
    LightningTrainer,
    SklearnTrainer,
    TorchTrainer,
)


def make_run_id(name: str, params: dict) -> str:
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
    try:
        params_str = json.dumps(params, sort_keys=True)
    except TypeError:
        # fallback: convert everything to string
        params_str = repr(params)
    # Hash to avoid overly long filenames
    run_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"{name}_{run_hash}"


class TaskModel(nn.Module):
    """
    A composite neural network model that combines a backbone and head architecture.
    This class serves as a container for two-stage models where a backbone network
    extracts features and a head network produces the final output. It provides a
    unified interface for forward passes and exposes the backbone's data type.
    Attributes:
        backbone (nn.Module): The feature extraction network that processes input data.
        head (nn.Module): The output network that processes backbone features to produce
                          the final model output.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
        if hasattr(backbone, "mean"):
            self.mean = backbone.mean
        if hasattr(backbone, "std"):
            self.std = backbone.std

    @property
    def data_type(self) -> str:
        """
        Get the data type of the model's backbone.
        Returns:
            str: The data type used by the backbone model (e.g., 'float32', 'float16', etc.).
        """

        return self.backbone.data_type

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input tensor to the model.
        Returns:
            Output tensor from the model head applied to backbone features.
        """

        feats = self.backbone(x)
        return self.head(feats)


def create_model(
    model_name: str,
    model_kwargs: dict = {},
):
    """Creates a model instance based on the specified type.
    Args:
        model (str): The type of model to create (e.g., "torch", "lightning").
        model_kwargs (dict): Additional keyword arguments for the model.

    Returns:
        nn.Module: Configured model instance for the specified type.
    """
    if model_name == "dummy_classifier":
        backbone = DummyClassifierModel(**model_kwargs)
        return backbone

    if model_name == "dummy_regressor":
        backbone = DummyRegressorModel(**model_kwargs)
        return backbone

    if model_name == "linear":
        backbone = LinearModel(**model_kwargs)
        return backbone

    if model_name == "pca_linear":
        backbone = PCALinearModel(**model_kwargs)
        return backbone

    if model_name == "pca_forest":
        backbone = PCARandomForestModel(**model_kwargs)
        return backbone

    if model_name == "pca_svm":
        backbone = PCASVMModel(**model_kwargs)
        return backbone

    if model_name == "2dcnn":
        backbone = ResNet3SliceMultihead(**model_kwargs)
        head = build_prediction_head(
            embedding_dim=backbone.out_dim,
            prediction_task=model_kwargs["prediction_task"],
            # num_classes=model_kwargs["num_classes"],
            dropout=model_kwargs["dropout"],
        )
        return TaskModel(backbone, head)

    if model_name == "medicalnet":
        backbone = MedicalNet(**model_kwargs)
        head = build_prediction_head(
            embedding_dim=backbone.out_dim,
            prediction_task=model_kwargs["prediction_task"],
            # num_classes=model_kwargs["num_classes"],
            dropout=model_kwargs["dropout"],
        )
        return TaskModel(backbone, head)

    else:
        raise ValueError(f"Unknown model type: {model_name}")


def create_backend_trainer(
    model,
    backend_kwargs: dict,
):
    """
    Create a backend trainer based on the specified backend type.
    Parameters
    ----------
    model : object
        The machine learning model to be trained.
    backend : str
        The backend framework to use for training. Supported options are:
        - "sklearn": scikit-learn based trainer
        - "torch": PyTorch based trainer
        - "lightning": PyTorch Lightning based trainer
        Case-insensitive.
    backend_kwargs : dict
        Additional keyword arguments to pass to the selected trainer class.
    Returns
    -------
    SklearnTrainer | TorchTrainer | LightningTrainer
        An instance of the appropriate trainer class based on the backend parameter.
    Raises
    ------
    ValueError
        If the backend string does not match any of the supported backends.
    """
    backend = backend_kwargs["backend"].lower()
    
    if backend == "sklearn":
        return SklearnTrainer(model=model, **backend_kwargs)

    if backend == "torch":
        return TorchTrainer(model=model, **backend_kwargs)

    if backend == "lightning":
        return LightningTrainer(model=model, **backend_kwargs)

    raise ValueError(f"Unknown backend: {backend}")


def create_trainer(
    model_name: str,
    model_kwargs: dict = {},
    backend_kwargs: dict = {},
):
    """Creates a Trainer for a specific model and backend.
    Args:
        model_name (str): The type of model to create (e.g., "2dcnn", "medicalnet").
        model_kwargs (dict): Additional keyword arguments for the model.
        backend (str): The backend type (e.g., "torch", "lightning").
        backend_kwargs (dict): Additional keyword arguments for the backend trainer.
    Returns:
        Trainer: Configured Trainer instance for the specified model and backend.
    """
    model = create_model(model_name, model_kwargs)
    trainer = create_backend_trainer(model, backend_kwargs)
    return trainer


def get_model(name: str, config: dict) -> object:
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

    return create_trainer(
        model_name=name,
        model_kwargs={**config["backbone"]},
        backend_kwargs={**config["backend"]},
    )

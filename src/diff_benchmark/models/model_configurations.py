from omegaconf import OmegaConf
from torch import nn

from diff_benchmark.models.sklearn_models.classic_ml import (
    PCARandomForestModel,
    PCASVMModel,
    RandomForestModel,
    SVMModel,
)
from diff_benchmark.models.sklearn_models.dummy import (
    DummyClassifierModel,
    DummyRegressorModel,
)
from diff_benchmark.models.sklearn_models.logistic_regression import (
    LassoModel,
    LinearModel,
    PCALinearModel,
)
from diff_benchmark.models.deep_models.cnn import ResNet3SliceMultihead
from diff_benchmark.models.deep_models.curia import CuriaBackbone
from diff_benchmark.models.deep_models.dinov2 import DinoViTBackbone
from diff_benchmark.models.deep_models.medicalnet import MedicalNet
from diff_benchmark.models.deep_models.vit import GoogleViTBackbone
from diff_benchmark.models.utils_models.prediction_head import build_prediction_head
from diff_benchmark.models.utils_models.trainer import (
    LightningTrainer,
    SklearnTrainer,
    TorchTrainer,
)


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
        if hasattr(backbone, "collate_with_augmentation"):
            self.collate_fn = backbone.collate_with_augmentation
        else:
            self.collate_fn = None

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
    model_kwargs: dict | None = None,
    pred_head: dict | None = None,
):
    """Creates a model instance based on the specified type.
    Args:
        model_name (str): The type of model to create (e.g., "forest", "2dcnn", "vit").
        model_kwargs (dict | None): Additional keyword arguments forwarded to the model constructor.
        pred_head (dict | None): Prediction head configuration (prediction_task, num_classes, etc.).
    Returns:
        nn.Module | SklearnModel: Configured model instance.
    Raises:
        ValueError: If model_name is not recognised.
    """
    model_kwargs = model_kwargs or {}
    pred_head = pred_head or {}
    if model_name == "dummy_classifier":
        backbone = DummyClassifierModel(**model_kwargs)
        return backbone

    if model_name == "dummy_regressor":
        backbone = DummyRegressorModel(**model_kwargs)
        return backbone

    if model_name == "linear":
        model_kwargs["prediction_task"] = pred_head["prediction_task"]
        backbone = LinearModel(**model_kwargs)
        return backbone

    if model_name == "pca_linear":
        model_kwargs["prediction_task"] = pred_head["prediction_task"]
        backbone = PCALinearModel(**model_kwargs)
        return backbone

    if model_name == "forest":
        model_kwargs["prediction_task"] = pred_head["prediction_task"]
        backbone = RandomForestModel(**model_kwargs)
        return backbone

    if model_name == "svm":
        model_kwargs["prediction_task"] = pred_head["prediction_task"]
        backbone = SVMModel(**model_kwargs)
        return backbone

    if model_name == "pca_forest":
        model_kwargs["prediction_task"] = pred_head["prediction_task"]
        backbone = PCARandomForestModel(**model_kwargs)
        return backbone

    if model_name == "pca_svm":
        model_kwargs["prediction_task"] = pred_head["prediction_task"]
        backbone = PCASVMModel(**model_kwargs)
        return backbone

    if model_name == "lasso":
        model_kwargs["prediction_task"] = pred_head["prediction_task"]
        backbone = LassoModel(**model_kwargs)
        return backbone

    if model_name == "2dcnn":
        backbone = ResNet3SliceMultihead(**model_kwargs)
        head = build_prediction_head(
            embedding_dim=backbone.out_dim,
            **pred_head,
        )
        return TaskModel(backbone, head)

    if model_name == "medicalnet":
        backbone = MedicalNet(**model_kwargs)
        head = build_prediction_head(
            embedding_dim=backbone.out_dim,
            **pred_head,
        )
        return TaskModel(backbone, head)
    if model_name == "dinov2":
        backbone = DinoViTBackbone(**model_kwargs)
        head = build_prediction_head(
            embedding_dim=backbone.embedding_dim,
            **pred_head,
        )
        return TaskModel(backbone, head)

    if model_name == "vit":
        backbone = GoogleViTBackbone(**model_kwargs)
        head = build_prediction_head(
            embedding_dim=backbone.embedding_dim,
            **pred_head,
        )
        return TaskModel(backbone, head)

    if model_name == "curia":
        backbone = CuriaBackbone(**model_kwargs)
        head = build_prediction_head(
            embedding_dim=backbone.embedding_dim,
            **pred_head,
        )
        return TaskModel(backbone, head)

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
    model_kwargs: dict | None = None,
    pred_head: dict | None = None,
    backend_kwargs: dict | None = None,
):
    """Creates a Trainer for a specific model and backend.
    Args:
        model_name (str): The type of model to create (e.g., "2dcnn", "medicalnet").
        model_kwargs (dict | None): Additional keyword arguments for the model.
        pred_head (dict | None): Prediction head configuration.
        backend_kwargs (dict | None): Additional keyword arguments for the backend trainer.
    Returns:
        BaseTrainer: Configured Trainer instance for the specified model and backend.
    """
    model_kwargs = model_kwargs or {}
    pred_head = pred_head or {}
    backend_kwargs = backend_kwargs or {}
    model = create_model(model_name, model_kwargs, pred_head)
    backend_kwargs["prediction_task"] = pred_head["prediction_task"]
    trainer = create_backend_trainer(model, backend_kwargs)
    return trainer


def get_model(name: str, config: dict) -> object:
    """
    Assemble and return a trainer for the named model using a resolved config dict.

    Pulls backend, data-partition, and random-state settings out of the config,
    injects them into the backend kwargs, then delegates to :func:`create_trainer`.

    Parameters:
        name (str): Model name, e.g. ``"linear"``, ``"forest"``, ``"2dcnn"``.
        config (dict): Fully-resolved config dictionary produced by OmegaConf.
    Returns:
        BaseTrainer: Configured trainer wrapping the model.
    Raises:
        ValueError: If *name* is not a recognised model identifier.
    """

    name = name.lower()
    config["backend"]["run_id"] = config["runtime"]["run_id"]
    config["backend"]["val_ratio"] = config["data"]["data_partition"]["val_size"]
    config["backend"]["seed"] = config["random_state"]
    return create_trainer(
        model_name=name,
        model_kwargs={**config["model"]["backbone"], "random_state": config["random_state"]},
        pred_head={**config["pred_head"]},
        backend_kwargs={**config["backend"]},
    )

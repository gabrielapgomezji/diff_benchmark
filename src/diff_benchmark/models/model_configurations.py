import torch
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
from diff_benchmark.models.sklearn_models.linear import (
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
from diff_benchmark.models.mesh_models.simple_mesh_model import SimpleMeshModel
from diff_benchmark.models.mesh_models.group_lasso import MeshGroupLassoModel
from diff_benchmark.models.mesh_models.spectral_laplacian_model import SpectralLaplacianAdditiveModel
from diff_benchmark.models.mesh_models.region_pca import RegionPCAModel
from diff_benchmark.models.mesh_models.sklearn_group_lasso import RegionGroupLassoModel
from diff_benchmark.models.mesh_models.sklearn_elasticnet import RegionElasticNetModel 
from diff_benchmark.models.mesh_models.spectral_laplacian_group_lasso import SpectralLaplacianGroupLassoModel
from diff_benchmark.models.mesh_models.sw_weisfeiler_leman import SWWeisfeilerLemanModel
from diff_benchmark.models.utils_models.additive_parcel_head import build_additive_parcel_head as build_additive_head
from diff_benchmark.models.utils_models.additive_parcel_head import build_new_parcel_head

class TaskModel(nn.Module):
    """Backbone + prediction head assembled into a single forward pass."""

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
        """Data type string from the backbone (e.g. ``'images'``, ``'array'``)."""
        return self.backbone.data_type

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def regularization_loss(self) -> torch.Tensor:
        """Forward the head's group regularisation penalty (if any)."""
        if hasattr(self.head, "regularization_loss"):
            return self.head.regularization_loss()
        return torch.tensor(0.0)


def create_model(
    model_name: str,
    model_kwargs: dict | None = None,
    pred_head: dict | None = None,
):
    """Instantiate a model from its name and config dicts.

    For sklearn-based models the prediction task is forwarded via
    ``model_kwargs["prediction_task"]``.  For deep models a
    :class:`TaskModel` wrapping backbone + head is returned.

    Args:
        model_name: Model identifier, e.g. ``"forest"``, ``"2dcnn"``,
            ``"vit"``.
        model_kwargs: Extra keyword arguments forwarded to the model
            constructor.
        pred_head: Prediction-head config (``prediction_task``,
            ``num_classes``, etc.).

    Returns:
        Configured model or :class:`TaskModel` instance.

    Raises:
        ValueError: If *model_name* is not recognised.
    """
    model_kwargs = model_kwargs or {}
    pred_head = pred_head or {}

    # --- Dummy baselines (no prediction_task needed) ---
    if model_name == "dummy_classifier":
        return DummyClassifierModel(**model_kwargs)

    if model_name == "dummy_regressor":
        return DummyRegressorModel(**model_kwargs)

    # --- sklearn models (prediction_task forwarded via kwargs) ---
    _sklearn_models: dict[str, type] = {
        "linear": LinearModel,
        "pca_linear": PCALinearModel,
        "forest": RandomForestModel,
        "svm": SVMModel,
        "pca_forest": PCARandomForestModel,
        "pca_svm": PCASVMModel,
        "lasso": LassoModel,
        "region_pca": RegionPCAModel,
        "region_group_lasso": RegionGroupLassoModel,
        "region_elasticnet": RegionElasticNetModel,
        "spectral_laplacian_group_lasso": SpectralLaplacianGroupLassoModel,
    }
    if model_name in _sklearn_models:
        model_kwargs["prediction_task"] = pred_head["prediction_task"]
        return _sklearn_models[model_name](**model_kwargs)

    # --- Mesh group-lasso (deep/torch model, data_type="mesh") ---
    if model_name == "group_lasso":
        backbone = MeshGroupLassoModel(**model_kwargs)
        head = build_additive_head(
            embed_dim=backbone.parcel_embed_dim,
            reg_type="group_lasso",
            lambda1=model_kwargs.get("lambda_gl", 1e-3),
            **pred_head,
        )
        return TaskModel(backbone, head)

    # --- Deep models (backbone + head) ---
    if model_name == "2dcnn":
        backbone = ResNet3SliceMultihead(**model_kwargs)
        head = build_prediction_head(embedding_dim=backbone.out_dim, **pred_head)
        return TaskModel(backbone, head)
    
    if model_name == "simple_mesh":
        backbone = SimpleMeshModel(**model_kwargs)
        # regression_head is Linear(hidden_dim → 1); use hidden_dim as embedding_dim
        head = build_prediction_head(embedding_dim=backbone.hidden_dim, **pred_head)
        return TaskModel(backbone, head)

    if model_name == "spectral_laplacian":
        backbone = SpectralLaplacianAdditiveModel(**model_kwargs)
        # AdditiveParcelHead: one weight vector per parcel, optional group regularisation.
        # pred_head may carry reg_type / lambda1 / lambda2 in addition to prediction_task.
        # head = build_additive_head(
        #     embed_dim=backbone.parcel_embed_dim,
        #     **pred_head,
        # )
        head = build_new_parcel_head(embed_dim=backbone.parcel_embed_dim, head_type="attention", **pred_head)
        return TaskModel(backbone, head)

    if model_name == "medicalnet":
        backbone = MedicalNet(**model_kwargs)
        head = build_prediction_head(embedding_dim=backbone.out_dim, **pred_head)
        return TaskModel(backbone, head)

    if model_name == "dinov2":
        backbone = DinoViTBackbone(**model_kwargs)
        head = build_prediction_head(embedding_dim=backbone.embedding_dim, **pred_head)
        return TaskModel(backbone, head)

    if model_name == "vit":
        backbone = GoogleViTBackbone(**model_kwargs)
        head = build_prediction_head(embedding_dim=backbone.embedding_dim, **pred_head)
        return TaskModel(backbone, head)

    if model_name == "curia":
        backbone = CuriaBackbone(**model_kwargs)
        head = build_prediction_head(embedding_dim=backbone.embedding_dim, **pred_head)
        return TaskModel(backbone, head)
        
    if model_name == "sw_weisfeiler_leman":
        ## Not a good idea to do Attention for a 1D distribution
        ## Maybe greater dim would be ok (e.g. with true Weisfeiler-Leman embedding)
        backbone = SWWeisfeilerLemanModel(**model_kwargs)
        head = build_new_parcel_head(embed_dim=backbone._parcel_embed_dim, head_type="attention_proba", **pred_head)
        return TaskModel(backbone, head)
    
    if model_name == "1d_distr_kernel":
        backbone = SWWeisfeilerLemanModel(**model_kwargs)
        # head = build_new_parcel_head(embed_dim=backbone._parcel_embed_dim, head_type="kernel_regression", **pred_head)
        # return TaskModel(backbone, head)
        print("???", backbone._parcel_embed_dim)
        return

    raise ValueError(f"Unknown model type: {model_name}")


def create_backend_trainer(
    model,
    backend_kwargs: dict,
):
    """Create a backend trainer for the given model.

    Args:
        model: The model to be trained.
        backend_kwargs (dict): Keyword arguments forwarded to the trainer; must contain
            ``backend`` (``"sklearn"``, ``"torch"``, or ``"lightning"``).

    Returns:
        SklearnTrainer | TorchTrainer | LightningTrainer: Configured trainer instance.

    Raises:
        ValueError: If ``backend_kwargs["backend"]`` is not a supported backend name.
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
    """Assemble and return a trainer for the named model using a resolved config dict.

    Args:
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
        model_kwargs={
            **config["model"]["backbone"],
            "random_state": config["random_state"],
            "n_jobs": config["cluster"]["conf"]["n_jobs"],
        },
        pred_head={**config["pred_head"]},
        backend_kwargs={**config["backend"]},
    )

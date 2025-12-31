import hashlib
import json
import torch.nn as nn


from diff_benchmark.models.classic_ml import PCARandomForestModel, PCASVMModel
from diff_benchmark.models.medicalnet import MedicalNet #ResNet3DModelLite, ResNet3DModel, 
from diff_benchmark.models.cnn import ResNet3SliceBackbone #CNNTorchTrainModel, ResNet3SliceModel, ResNet3SliceMultihead

from diff_benchmark.models.dummy import DummyClassifier, DummyRegressor
from diff_benchmark.models.logistic_regression import (
    LinearModel,
    PCALinearModel,
)

from diff_benchmark.models.base import TorchPipeline, LightningModel, NumpyAbstractModel


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
    params_str = json.dumps(params, sort_keys=True)
    # Hash to avoid overly long filenames
    run_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"{name}_{run_hash}"








# def create_model(
#     model_name: str,
#     model_kwargs: dict = {},
# ):
    
#     """Creates a model instance based on the specified type.
#     Args:
#         model (str): The type of model to create (e.g., "torch", "lightning").
#         model_kwargs (dict): Additional keyword arguments for the model.
    
#     Returns:
#         nn.Module: Configured model instance for the specified type.
#     """
    
    
#     if model_name == "pca":
#         return PCARandomForestModel(**model_kwargs)
    
#     elif model_name == "2dcnn":
#         return ResNet3SliceMultihead(**model_kwargs)
    
#     elif model_name == "medicalnet":
#         return MedicalNet(**model_kwargs)

#     raise ValueError(f"Unknown model type: {model_name}")



# def create_backend_trainer(
#     model: nn.Module,
#     backend: str,
#     backend_kwargs: dict = {},
# ):
#     """Creates a Trainer for a specific backend model.
#     Args:
#         model (nn.Module): The model to be trained.
#         backend (str): The backend type (e.g., "torch", "lightning").
#         backend_kwargs (dict): Additional keyword arguments for the backend trainer.
    
#     Returns:
#         Trainer: Configured Trainer instance for the specified backend.
#     """
#     if backend == "torch":
#         model.std = 0.5
#         model.mean = 0.5
#         return TorchPipeline(model=model, **backend_kwargs)
    
#     elif backend == "lightning":
#         return LightningModel(model=model, **backend_kwargs)
    
#     elif backend == "sklearn":
#         return NumpyAbstractModel(model=model, **backend_kwargs)


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
    
    if model_name in {"2dcnn", "2dcnn_torch", "2dcnn_lite"}:
        return ResNet3SliceBackbone(**model_kwargs)
    
    elif model_name in {"medicalnet", "medicalnet_lite"}:
        return MedicalNet(**model_kwargs)

    raise ValueError(f"Unknown model type: {model_name}")


from diff_benchmark.models.utils_models.trainer import TorchTrainer, LightningTrainer
def create_backend_trainer(
    model,
    backend: str,
    backend_kwargs: dict,
):
    backend = backend.lower()

    if backend == "torch":
        return TorchTrainer(model=model, **backend_kwargs)

    if backend == "lightning":
        return LightningTrainer(model=model, **backend_kwargs)

    raise ValueError(f"Unknown backend: {backend}")


def create_trainer(
    model_name: str,
    model_kwargs: dict = {},
    backend: str = "lightning",
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
    trainer = create_backend_trainer(model, backend, backend_kwargs)
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
    backend = "lightning" #"torch" #
    if backend == "lightning":
        config["trainer_kwargs"] = {
                "max_epochs": config.get("epochs", 100),
                "accelerator": "gpu",
                "devices": 1,
                "log_every_n_steps": 10,
            }
    return create_trainer(model_name=name,
    model_kwargs={**config}, #model_kwargs,
    backend=backend,
    backend_kwargs={**config}, #backend_kwargs,
    )

    if name == "dummy_classifier":
        return DummyClassifier()

    if name == "dummy_regressor":
        return DummyRegressor()

    if name == "pca_linear":
        return PCALinearModel(**config)

    if name == "linear":
        return LinearModel(**config)

    if name == "pca_forest":
        return PCARandomForestModel(**config)

    if name == "pca_svm":
        return PCASVMModel(**config)

    if name in {"2dcnn", "2dcnn_torch", "2dcnn_lite"}:
        backbone = ResNet3SliceBackbone(
            input_slices=config["input_slices"],
            num_classes=config["num_classes"],
            freeze_backbone=config.get("freeze_backbone", True),
            dropout=config.get("dropout", 0.5),
            pretrained=config.get("pretrained", False),
            trainable_blocks=config.get("trainable_blocks", None),
            prediction_task=config.get("prediction_task", None),
        )

        backend = "lightning" #"torch" # 
        # For torch training
        # backend_kwargs = {
        #     "epochs": config.get("epochs", 100),
        #     "learning_rate": config.get("learning_rate", 1e-4),
        #     "weight_decay": config.get("weight_decay", 1e-4),
        #     "prediction_task": config.get("prediction_task", "regression"),
        # }
        # For lightning training
        backend_kwargs = {
            "learning_rate": config.get("learning_rate", 1e-4),
            "weight_decay": config.get("weight_decay", 1e-4),
            "prediction_task": config.get("prediction_task", "regression"),
            "trainer_kwargs": {
                "max_epochs": config.get("epochs", 100),
                "accelerator": "gpu",
                "devices": 1,
                "log_every_n_steps": 10,
            },
        }

        return create_trainer(
            model=backbone,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )
    if name in {"medicalnet", "medicalnet_lite"}:
        backbone = MedicalNet(
                depth = 50,
                num_classes=2,
                prediction_task="classification",
                shortcut_type="B",
                no_cuda=False,
            )
        backend_kwargs = {
            "epochs": config.get("epochs", 100),
            "learning_rate": config.get("learning_rate", 1e-4),
            "weight_decay": config.get("weight_decay", 1e-4),
            "prediction_task": config.get("prediction_task", "regression"),
        }

        # For Lightning backend
        # backend_kwargs = {
        #     "learning_rate": 1e-5,
        #     "weight_decay": 1e-4,
        #     "scheduler_type": "plateau",
        #     "optimizer_type": "adamw",
        #     "prediction_task": "classification",
        #     "run_id": "lightning_run_01",
        #     "fold_idx": 0,
        #    "trainer_kwargs": {
        #         "max_epochs": config.get("epochs", 100),
        #         "accelerator": "gpu",
        #         "devices": 1,
        #         "log_every_n_steps": 10,
        #     },
        # }
        
        backend = "torch"
        

        return create_trainer(
            model=backbone,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )
    # if name == "2dcnn_torch":
    #     return CNNTorchTrainModel(**config)
    
    return create_trainer(
        model_name=name,
        model_kwargs=config,
        backend="lightning",
        backend_kwargs={},
    )

    if name == "2dcnn":
        return create_backend_trainer(
            model=CNNTorchTrainModel(**config),
            backend="lightning",
            backend_kwargs={},
        )

    if name == "2dcnn_lite":
        return ResNet3SliceModel(**config)

    if name == "medicalnet":
        return ResNet3DModel(**config)
    
    if name == "medicalnet_lite":
        return ResNet3DModelLite(**config)

    raise ValueError(f"Unknown model name: {name}")

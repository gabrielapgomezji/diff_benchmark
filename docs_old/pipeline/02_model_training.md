# Model Training

> **Pipeline position:** Step 3 — runs after preprocessing and optional feature caching.  
> **← [Back to index](index.md)**

---

## Overview

The training component implements a unified trainer interface spanning sklearn-style pipelines and pure PyTorch training backend. All trainers expose the same `fit` / `predict` API, keeping `run.py` entirely backend-agnostic.

**Inputs:**
- A PyTorch `DataLoader` wrapping a `CustomDataset` (or `CachedFeatureDataset`)
- Model instance (sklearn estimator or `nn.Module`)

**Outputs:**
- Fitted model state
- Prediction arrays (`np.ndarray`) for train and test splits

---

| Name | Type | Description |
|---|---|---|
| `linear` | sklearn | Logistic / linear regression |
| `pca_linear` | sklearn | PCA + linear |
| `lasso` | sklearn | Lasso regression |
| `forest` | sklearn | Random forest |
| `pca_forest` | sklearn | PCA + random forest |
| `svm` | sklearn | SVM |
| `pca_svm` | sklearn | PCA + SVM |
| `dummy_classifier` | sklearn | Majority-class baseline |
| `dummy_regressor` | sklearn | Mean-prediction baseline |
| `dinov2` | deep | DINOv2 ViT backbone |
| `curia` | deep | CURIA backbone |
| `medicalnet` | deep | MedicalNet 3D backbone |

---

## Configuration

Parameters for deep trainers are set in the `backend` Hydra config group (`configs/backend/`).

| Parameter | Trainer | Default | Description |
|---|---|---|---|
| `epochs` | `TorchTrainer`, `LightningTrainer` | 5 | Number of training epochs |
| `learning_rate` | All deep | `1e-4` | Initial learning rate |
| `weight_decay` | All deep | `1e-4` | L2 regularisation coefficient |
| `debug` | All deep | `True` | Enables `TorchDebugLogger` |

** Any parameters specific to each model

---

## Usage Example

```python
from diff_benchmark.models.model_configurations import get_model
from omegaconf import OmegaConf

cfg = OmegaConf.load("exp_outputs/experiments/exp_myrun/config.yaml")

model = get_model(cfg.model.name, OmegaConf.to_container(cfg, resolve=True))

model.set_fold(0)
model.fit(train_loader)

predictions = model.predict(test_loader)  # np.ndarray, shape (N,)
```



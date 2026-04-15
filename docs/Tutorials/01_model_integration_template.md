# Tutorial: Add Your Own Model to DiffBench

This template explains how to plug in a new model:

- sklearn-style models (classical ML)
- deep models (PyTorch backbone + prediction head)

---

## 1) How model creation works in this benchmark

All model creation is centralized in:

- `src/diff_benchmark/models/model_configurations.py`

High-level flow:

1. Hydra loads config groups (`model`, `pred_head`, `backend`, etc.).
2. `get_model(...)` builds `model_kwargs`, `pred_head`, `backend_kwargs`.
3. `create_model(...)` instantiates either:
   - a sklearn model, or
   - a deep `TaskModel(backbone, head)`.
4. `create_backend_trainer(...)` wraps it in:
   - `SklearnTrainer` or `TorchTrainer`.
5. The run loop calls `trainer.fit(train_loader)` then `trainer.predict(test_loader)`.

---

## 2) Required data contract (input, intermediate representation, output)

There are two different contracts:

1. Backbone to head contract (depends on head design).
2. Final trainer output contract (standardized for benchmark metrics).

### 2.1 Backbone input: Dataloader

Your model uses dataloaders with tuple structure/representation like:

- `(x, y, gender)` for image/array pipelines
- `(x, y, ...)` for mesh pipelines

### 2.2 Backbone output to head input (For Deep models only)

This step does depend on the head architecture.

| Model family | Typical backbone output | Head expected input |
|---|---|---|
| Standard deep models (`build_prediction_head`) | `(B, E)` | `(B, E)` |
| Mesh additive heads (`build_new_parcel_head`) | `(B, P, E)` | `(B, P, E)` |

Where:

- `B` = batch size
- `E` = embedding dimension
- `P` = number of parcels/regions

For mesh models, since we want a parcel representation:

1. Backbone produces one embedding per region: `(B, P, E)`.
2. Parcel head computes per-parcel contributions.
3. Head aggregates parcel contributions to one prediction tensor per sample.

Examples in codebase:

- `SpectralLaplacianAdditiveModel` outputs parcel embeddings and is paired with parcel heads.
- `RegionConstrainedPointNetPP` returns `(B, P, E)` for additive parcel heads.

### 2.3 Final trainer output contract

Regardless of internal head design, `trainer.predict(...)` is expected to return:

- `np.ndarray` shape `(N,)`
- class labels for `binary_classification`
- continuous values for `regression`

For deep models, the head forward output consumed by the trainer should be:

- classification logits `(B, C)`
- regression `(B, 1)` (trainer squeezes dim 1)

The benchmark output format is standardized, but the intermediate tensor dimensions does depend on the head type.

---

## 3) Model type 1: Add a sklearn model

### 3.1 Create model class

Create a file in:

- `src/diff_benchmark/models/sklearn_models/`

Use this template:

```python
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge

from diff_benchmark.models.utils_models.trainer import SklearnModel


class MySklearnModel(SklearnModel):
    """Example sklearn model compatible with DiffBench."""

    # Optional: override when your model expects mesh objects instead of arrays
    # data_type = "mesh"

    def _build_model(self, **kwargs) -> BaseEstimator:
        prediction_task = kwargs.get("prediction_task", "binary_classification")
        random_state = kwargs.get("random_state", 42)

        if prediction_task == "binary_classification":
            estimator = LogisticRegression(max_iter=2000, random_state=random_state)
        else:
            estimator = Ridge(alpha=1.0)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("estimator", estimator),
            ]
        )
        return pipeline
```

Notes:

- In most cases, no custom `fit` or `predict` is needed because `SklearnModel` already provides both.
- If your model works with mesh dict lists instead of flat arrays, set `data_type = "mesh"` and handle that internally.

### 3.2 Register in model factory

Edit:

- `src/diff_benchmark/models/model_configurations.py`

Add import and registry entry in `_sklearn_models`:

```python
from diff_benchmark.models.sklearn_models.my_model import MySklearnModel

_sklearn_models = {
    # ...
    "my_sklearn": MySklearnModel,
}
```

### 3.3 Add Hydra config

Create:

- `src/diff_benchmark/configs/model/my_sklearn.yaml`

Template:

```yaml
name: my_sklearn

backbone:
  # put kwargs needed by _build_model here
  # example:
  # alpha: 1.0
```

### 3.4 Run

```bash
diffbenchmark-run model=my_sklearn backend=sklearn pred_head=binary_classification
```

---

## 4) Model type 2: Add a deep model

### 4.1 Create backbone module

Create a file in:

- `src/diff_benchmark/models/deep_models/` or `src/diff_benchmark/models/mesh_models`

Template:

```python
import torch
from torch import nn


class MyBackbone(nn.Module):
    data_type = "images"  # or "array" / "mesh" according to your dataset representation

    def __init__(self, in_dim: int = 1024, out_dim: int = 256, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
        self.out_dim = out_dim

        # Optional normalization metadata used by data pipeline
        self.mean = 0.5
        self.std = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return embeddings shaped (B, out_dim)
        return self.net(x)
```

Optional advanced hook:

- implement `collate_with_augmentation(self, batch, transform=None)` when you need custom collation/augmentation.

### 4.2 Pick an existing head or add your own

In `src/diff_benchmark/models/utils_models/` there are already implemented heads. You have two options when wiring a deep model. 

Option A: use existing heads.

- Vector embeddings `(B, E)`:
  - use `build_prediction_head(...)`
- Parcel embeddings `(B, P, E)`:
  - use `build_new_parcel_head(...)`
  - supported `head_type`: `simple`, `additive`, `attention`, `transformer`, `gam`, `moe`

Option B: implement a custom head (recommended when your algorithm needs a different aggregation rule).

Custom parcel head template:

```python
import torch
from torch import nn


class MyParcelHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.parcel_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (B, P, E)
        if x.dim() != 3:
            raise ValueError(f"Expected (B, P, E), got {tuple(x.shape)}")
        contrib = self.parcel_mlp(x)    # (B, P, C)
        return contrib.sum(dim=1)       # (B, C)
```

### 4.3 Register in model factory

Edit:

- `src/diff_benchmark/models/model_configurations.py`

Template registration with an existing head:

```python
from diff_benchmark.models.deep_models.my_backbone import MyBackbone

if model_name == "my_deep":
    backbone = MyBackbone(**model_kwargs)
    head = build_prediction_head(embedding_dim=backbone.out_dim, **pred_head)
    return TaskModel(backbone, head)
```

Template registration with a parcel head:

```python
if model_name == "my_mesh_deep":
    backbone = MyMeshBackbone(**model_kwargs)  # returns (B, P, E)
    head = build_new_parcel_head(embed_dim=backbone.out_dim, **pred_head)
    return TaskModel(backbone, head)
```

Use the embedding attribute your backbone exposes, for example:

- `backbone.out_dim` (PointNet-style backbones)
- `backbone.parcel_embed_dim` (spectral-laplacian-style backbones)

Template registration with your custom head:

```python
if model_name == "my_mesh_custom_head":
    backbone = MyMeshBackbone(**model_kwargs)  # returns (B, P, E)
    output_dim = 2 if pred_head["prediction_task"] == "binary_classification" else 1
    head = MyParcelHead(embed_dim=backbone.out_dim, output_dim=output_dim)
    return TaskModel(backbone, head)
```

Why this pattern matters:

- `TaskModel` standardizes backbone plus head behavior.
- `TorchTrainer` expect this forward contract.

### 4.4 Add Hydra config

Create:

- `src/diff_benchmark/configs/model/my_deep.yaml`

Template:

```yaml
name: my_deep

backbone:
  in_dim: 1024
  out_dim: 256
```

### 4.5 Choose backend and prediction head

Typical combinations:

- deep training with PyTorch loop: `backend=torch`
- task type from prediction head:
  - `pred_head=binary_classification`
  - `pred_head=regression`

Example `pred_head` for an existing parcel head:

```yaml
pred_head:
    prediction_task: binary_classification
    head_type: attention
    head:
        attention:
            hidden_dim: 64
```

Example `pred_head` for additive regularized parcel head:

```yaml
prediction_task: binary_classification
head_type: additive
bias: true

# Per-head parameter blocks. Only head.<head_type>.* is forwarded.
head:
  simple: {}

  additive:
    reg_type: group_lasso    # or group_elastic_net / none
    lambda1: 0.00001
    lambda2: 0.000001
```

If you use a fully custom head instantiated directly in `create_model(...)`, keep `pred_head` minimal and pass only task-level settings your branch needs.

### 4.6 Run

```bash
diffbenchmark-run model=my_deep backend=torch pred_head=regression
```

---

## 5) Common pitfalls

- Forgetting `model_kwargs["prediction_task"]` for sklearn models.
- Returning wrong tensor shape for deep classification/regression.
- Missing `out_dim` or `embedding_dim` attribute needed to build prediction head.
- Registering model name in YAML but not in `create_model(...)`.
- Using `backend=sklearn` with a deep model (or the opposite).
- Returning `(B, P, C)` from a parcel head without reducing over parcels.

---

## 6) Minimal quick-start commands

```bash
# sklearn path
diffbenchmark-run model=my_sklearn backend=sklearn pred_head=binary_classification

# deep path
diffbenchmark-run model=my_deep backend=torch pred_head=regression
```

If these run end-to-end, your model is correctly wired into the benchmark.

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class SimpleMeshModel(nn.Module):
    """MLP on node features + global mean pooling for mesh-based regression.

    The model expects a batch in the form of a list of dictionaries, where
    each dictionary represents a mesh sample and contains:

        {
            "node_features": FloatTensor (N, F),
            "vertices": FloatTensor (N, 3),
            "parcel_labels": LongTensor (N,),
            "edge_index": LongTensor (2, E),
        }

    Only ``node_features`` are used in this simple demonstration model.
    """

    data_type: str = "mesh"
    collate_fn: Optional[object] = None

    def __init__(
        self,
        in_features: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _stack_node_features(
        self, batch: List[Dict[str, torch.Tensor]], device: torch.device
    ) -> torch.Tensor:
        """Extract and stack node features from a batch of mesh dicts.

        Args:
            batch: List of mesh dictionaries.
            device: Device where tensors should be moved.

        Returns:
            Tensor of shape (B, N, F)
        """
        node_tensors = [sample["node_features"].to(device) for sample in batch]
        return torch.stack(node_tensors, dim=0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Union[List[Dict], torch.Tensor]) -> torch.Tensor:
        """Compute graph embeddings from a mesh batch.

        Args:
            x: Either

                • List of mesh dictionaries (batch format from safe_collate)
                • Tensor (B, N, F) or (N, F)

        Returns:
            Tensor (B, hidden_dim) graph embeddings.
        """

        device = next(self.parameters()).device

        if isinstance(x, list):
            # Convert list of mesh dicts → tensor (B, N, F)
            x = self._stack_node_features(x, device)

        elif isinstance(x, torch.Tensor):
            x = x.to(device)

        else:
            raise TypeError(
                f"Unsupported input type {type(x)}. Expected list of dicts or torch.Tensor."
            )

        # Handle single mesh input
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (N, F) → (1, N, F)

        # Node encoding
        node_embeddings = self.node_encoder(x)  # (B, N, hidden)

        # Global mean pooling across vertices
        graph_embedding = node_embeddings.mean(dim=1)  # (B, hidden)

        return graph_embedding

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SimpleMeshModel("
            f"in_features={self.in_features}, "
            f"hidden_dim={self.hidden_dim})"
        )

# """Simple demonstration model for surface-mesh data.

# This module provides :class:`SimpleMeshModel`, a lightweight MLP that operates
# on per-vertex node features produced by
# :class:`~diff_benchmark.preprocessing.brain_feature_extraction.MeshPipeline`.

# Architecture
# ------------
# ::

#     safe_collate loads both parquets → X_batch = [mesh_dict, mesh_dict, ...]
#          │
#          ▼  (inside forward)
#     stack node_features  →  (B, N, F)
#          │
#          ▼
#     Linear(F → hidden)  +  ReLU
#          │
#          ▼
#     Linear(hidden → hidden)  +  ReLU
#          │
#          ▼
#     Global mean-pooling across N vertices  →  (B, hidden)
#          │
#          ▼
#     [TaskModel.head]  →  scalar prediction

# Each mesh dict in the batch contains::

#     {
#         "node_features":  FloatTensor (N, F),
#         "vertices":       FloatTensor (N, 3),
#         "parcel_labels":  LongTensor  (N,),
#         "edge_index":     LongTensor  (2, E),
#     }

# The model does **not** require PyTorch Geometric.  ``forward`` uses only
# ``node_features`` from each dict; ``vertices``, ``parcel_labels``, and
# ``edge_index`` are available for more advanced graph models.

# Usage example
# -------------
# ::

#     from diff_benchmark.models.simple_mesh_model import SimpleMeshModel
#     import torch

#     model = SimpleMeshModel(in_features=1, hidden_dim=64)

#     # Batch as produced by safe_collate in mesh mode
#     X = [
#         {"node_features": torch.randn(64984, 1), "vertices": ..., ...},
#         {"node_features": torch.randn(64984, 1), "vertices": ..., ...},
#     ]
#     emb = model(X)   # (2, 64)
# """

# from __future__ import annotations

# from typing import Dict, List, Optional, Union

# import torch
# import torch.nn as nn


# class SimpleMeshModel(nn.Module):
#     """MLP on node features + global mean pooling for mesh-based regression.

#     ``forward`` accepts the mesh batch produced by
#     :meth:`~diff_benchmark.data.dataloaders.PreprocessedData.safe_collate` in
#     mesh mode: a **list of dicts** each containing ``"nodes_path"`` and
#     ``"edges_path"`` keys pointing to BIDS-named Parquet files.  The model
#     loads the node features from Parquet internally, so no graph data needs to
#     be pre-loaded in memory.

#     Args:
#         in_features: Number of input features per vertex (``F``).  Defaults to
#             ``1`` (single microstructure metric per vertex).
#         hidden_dim: Width of the hidden layers.  Defaults to ``128``.
#         dropout: Dropout probability applied after each hidden activation.
#             Set to ``0.0`` to disable.  Defaults to ``0.0``.

#     Attributes:
#         data_type: Always ``"mesh"`` — consumed by
#             :func:`~diff_benchmark.data.prepare_data.get_data_pipeline` to
#             select the correct preprocessing pipeline.
#         collate_fn: Always ``None``; the standard
#             :meth:`~diff_benchmark.data.dataloaders.PreprocessedData.safe_collate`
#             is used.
#     """

#     data_type: str = "mesh"
#     collate_fn: Optional[object] = None

#     def __init__(
#         self,
#         in_features: int = 1,
#         hidden_dim: int = 128,
#         dropout: float = 0.0,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         self.in_features = in_features
#         self.hidden_dim = hidden_dim

#         self.node_encoder = nn.Sequential(
#             nn.Linear(in_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------

#     def forward(self, x: Union[List[Dict], torch.Tensor]) -> torch.Tensor:
#         """Predict graph embedding from a batch of mesh dicts or a raw tensor.

#         Args:
#             x: Either:

#                 - A **list of mesh dicts** as produced by
#                   :meth:`~diff_benchmark.data.dataloaders.PreprocessedData.safe_collate`
#                   in mesh mode.  Each dict must contain the key
#                   ``"node_features"`` (FloatTensor ``(N, F)``).  The other keys
#                   (``"vertices"``, ``"parcel_labels"``, ``"edge_index"``) are
#                   available for more advanced models but are ignored here.
#                 - A **float tensor** ``(B, N, F)`` or ``(N, F)`` for direct use
#                   in tests or interactive exploration.

#         Returns:
#             Graph embeddings of shape ``(B, hidden_dim)``.  The final
#             prediction head is applied by
#             :class:`~diff_benchmark.models.model_configurations.TaskModel`.
#         """
#         breakpoint()
#         if isinstance(x, list):
#             # Stack node_features from each mesh dict → (B, N, F)
#             node_tensors = [d["node_features"] for d in x]
#             x = torch.stack(node_tensors, dim=0).to(next(self.parameters()).device)

#         if x.dim() == 2:
#             x = x.unsqueeze(0)   # (N, F) → (1, N, F)

#         # x: (B, N, F)
#         node_embeddings = self.node_encoder(x)         # (B, N, hidden)
#         graph_embedding = node_embeddings.mean(dim=1)  # (B, hidden) — global mean pool
#         return graph_embedding

#     # ------------------------------------------------------------------
#     # Repr
#     # ------------------------------------------------------------------

#     def __repr__(self) -> str:  # pragma: no cover
#         return (
#             f"SimpleMeshModel("
#             f"in_features={self.in_features}, "
#             f"hidden_dim={self.hidden_dim})"
#         )

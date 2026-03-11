"""Group-lasso model for surface-mesh data.

This module provides :class:`MeshGroupLassoModel`, a backbone ``nn.Module``
that pools per-vertex node features to **parcel-level means** and returns a
structured ``(B, P, F)`` tensor.

It follows the same backbone-only pattern as every other deep model in the
benchmark (e.g. :class:`~diff_benchmark.models.mesh_models.laplacian_model.SimpleMeshModel`).
The prediction head and group-lasso regularisation are handled by the existing
:class:`~diff_benchmark.models.utils_models.additive_parcel_head.AdditiveParcelHead`
(``reg_type="group_lasso"``), assembled via
:func:`~diff_benchmark.models.utils_models.additive_parcel_head.build_additive_parcel_head`
in ``model_configurations.py`` — exactly like ``"spectral_laplacian"``.

Architecture
------------
::

    mesh batch (list of dicts)
          │
          ▼  parcel mean-pooling (label 0 excluded)
    (B, P, F)   ← backbone output
          │
          ▼  AdditiveParcelHead(reg_type="group_lasso")
    (B, output_dim)

The ``parcel_embed_dim`` attribute exposed by the backbone equals ``F``
(``in_features``) so that ``build_additive_parcel_head`` can be called with
``embed_dim=backbone.parcel_embed_dim``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class MeshGroupLassoModel(nn.Module):
    """Parcel mean-pooling backbone for surface-mesh data.

    Aggregates per-vertex ``node_features`` to parcel-level means using
    ``parcel_labels`` from the mesh dict, producing a structured
    ``(B, P, F)`` tensor compatible with
    :class:`~diff_benchmark.models.utils_models.additive_parcel_head.AdditiveParcelHead`.

    The number of parcels ``P`` is inferred from the data at runtime; no
    atlas-specific configuration is required.

    The model expects a batch produced by
    :meth:`~diff_benchmark.data.dataloaders.PreprocessedData.safe_collate`
    in mesh mode: a **list of dicts**, each containing:

    .. code-block:: python

        {
            "node_features":  FloatTensor (N, F),
            "parcel_labels":  LongTensor  (N,),   # globally unique across LH+RH
            "hemisphere":     LongTensor  (N,),   # 0 = LH, 1 = RH
            "vertices":       FloatTensor (N, 3),   # unused
            "edge_index":     LongTensor  (2, E),   # unused
        }

    Label 0 is treated as the medial-wall / unlabelled region and excluded.
    Right-hemisphere parcel IDs are offset by the maximum LH label (e.g.
    labels 1–500 for LH, 501–1000 for RH with Schaefer-1000) so that the
    combined ``parcel_labels`` vector contains 1 000 globally unique IDs and
    pooling never conflates the two hemispheres.

    Parameters
    ----------
    in_features:
        Number of per-vertex scalar features ``F``.  Defaults to 1.
    """

    data_type: str = "mesh"
    collate_fn: Optional[object] = None

    def __init__(self, in_features: int = 1, **kwargs) -> None:
        super().__init__()

        self.in_features = in_features
        # Exposed so build_additive_parcel_head can be called as:
        #   build_additive_parcel_head(embed_dim=backbone.parcel_embed_dim, ...)
        self.parcel_embed_dim: int = in_features

        # Dummy parameter — keeps the module non-empty before any real
        # parameters are added by the head (needed by the optimiser).
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ------------------------------------------------------------------
    # Parcel mean-pooling
    # ------------------------------------------------------------------

    def _pool_parcels(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Aggregate per-vertex features to parcel means.

        Parameters
        ----------
        batch:
            List of mesh dicts (one per sample in the mini-batch).

        Returns
        -------
        FloatTensor of shape ``(B, P, F)``.
        """
        device = self._dummy.device
        pooled_samples = []

        for sample in batch:
            node_features = sample["node_features"].to(device)   # (N, F)
            parcel_labels = sample["parcel_labels"].to(device)   # (N,)

            # Skip label 0 (medial wall / unlabelled)
            unique_labels = torch.unique(parcel_labels)
            unique_labels = unique_labels[unique_labels > 0]
            unique_labels, _ = unique_labels.sort()

            parcel_means = []
            for label in unique_labels:
                mask = parcel_labels == label
                parcel_means.append(node_features[mask].mean(dim=0))  # (F,)

            pooled_samples.append(torch.stack(parcel_means, dim=0))  # (P, F)

        return torch.stack(pooled_samples, dim=0)   # (B, P, F)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Union[List[Dict[str, torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        """Return parcel-mean-pooled embeddings ``(B, P, F)``.

        Parameters
        ----------
        x:
            List of mesh dicts (standard mesh-pipeline format from safe_collate).

        Returns
        -------
        FloatTensor of shape ``(B, P, F)``.
        """
        if not isinstance(x, list):
            raise TypeError(
                "MeshGroupLassoModel expects a list of mesh dicts. "
                f"Got {type(x)}."
            )
        
        return self._pool_parcels(x)   # (B, P, F)
"""
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SWWeisfeilerLemanModel(nn.Module):
    """

    .. note::
        **Parcel id 0 (background / medial wall) is always excluded.**
        Background vertices remain in the mesh data but are never processed:
        no spectral basis is built for them, no embedding is produced, and
        downstream heads (group lasso, coefficient extraction, contribution
        visualisations) will not see a coefficient for the background.

    Parameters
    ----------
    in_features:
        Number of scalar features per vertex (``F``).
    n_spectral_components:
        Number of Laplacian eigenvectors (``k``) per parcel.
    parcel_ids:
        Optional list of integer parcel IDs.  Inferred lazily from the
        first batch if not provided.  Parcel 0 is always removed even if
        explicitly supplied here.
    """

    data_type: str = "mesh"
    collate_fn: Optional[object] = None

    def __init__(
        self,
        in_features: int = 1,
        n_spectral_components: int = 16,
        parcel_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.k = n_spectral_components

        # Fixed at construction time — used by downstream heads.
        self._parcel_embed_dim: int = n_spectral_components * in_features




    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: Union[List[Dict[str, torch.Tensor]], torch.Tensor]
    ) -> torch.Tensor:
        """

        Args:
            x: List of mesh dicts; each must contain ``node_features`` (N, F),
               ``parcel_labels`` (N,), and ``edge_index`` (2, E).

        Returns:
            FloatTensor ``(B, n_parcels, parcel_embed_dim)`` where the parcel
            axis is ordered by sorted parcel IDs.
        """
        if not isinstance(x, list):
            raise TypeError(
                "SpectralLaplacianAdditiveModel expects a list of mesh dicts. "
                f"Got {type(x)}."
            )

        device = self._dummy.device

        return 0 # (B, n_parcels, k*F)

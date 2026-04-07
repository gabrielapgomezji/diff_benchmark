"""
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

import ot

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class Distr1D_kernel(nn.Module):
    """

    .. note::
        **Parcel id 0 (background / medial wall) is always excluded.**
        Background vertices remain in the mesh data but are never processed:
        no spectral basis is built for them, no embedding is produced, and
        downstream heads (group lasso, coefficient extraction, contribution
        visualisations) will not see a coefficient for the background.

    Parameters
    ----------
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
        parcel_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_features = in_features

        # Fixed at construction time — used by downstream heads.
        self._parcel_embed_dim: int = in_features

        # Sorted list of parcel IDs; set lazily if not provided.
        # Parcel 0 (background / medial wall) is always excluded.
        self._parcel_ids: Optional[List[int]] = None
        if parcel_ids is not None:
            self._parcel_ids = sorted(int(p) for p in parcel_ids if int(p) != 0)

        # Dummy parameter so the module has a device and state_dict entry.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)


    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _maybe_init_from_batch(self, batch: List[Dict[str, torch.Tensor]]) -> None:
        """Lazy initialisation: infer parcel IDs from the first batch.

        Parcel 0 (background / medial wall) is always excluded.
        """
        if self._parcel_ids is not None:
            return

        parcel_labels = batch[0]["parcel_labels"]
        unique_ids = parcel_labels.unique().tolist()
        self._parcel_ids = sorted(int(p) for p in unique_ids if int(p) != 0)
        log.debug("Ignoring parcel 0 (background).")
        log.debug("Lazily initialised %d parcel IDs.", len(self._parcel_ids))



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
            FloatTensor ``(B, B, n_parcels)`` where the parcel
            axis is ordered by sorted parcel IDs.
        """
        if not isinstance(x, list):
            raise TypeError(
                "SpectralLaplacianAdditiveModel expects a list of mesh dicts. "
                f"Got {type(x)}."
            )

        device = self._dummy.device
        self._maybe_init_from_batch(x)

        B, n_parcels, F = len(x), len(self._parcel_ids), self.in_features

        matrix_Wasserstein_distance = torch.zeros(B, B, n_parcels, device=device)

        for b1, x1 in enumerate(x):
            for b2, x2 in enumerate(x):
                for p_idx, pid in enumerate(self._parcel_ids):
                    mask1 = x1["parcel_labels"] == pid
                    mask2 = x2["parcel_labels"] == pid
                    node_features1 = x1["node_features"][:, 0][mask1].to(device)   # (n1,)
                    node_features2 = x2["node_features"][:, 0][mask2].to(device)   # (n2,)
                    matrix_Wasserstein_distance[b1, b2, p_idx] = ot.wasserstein_1d(node_features1, node_features2)

        return matrix_Wasserstein_distance  # (B, B, n_parcels)

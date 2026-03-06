"""Spectral Laplacian backbone for parcel-based additive graph regression.

Architecture
------------
For every sample the mesh is decomposed into a fixed set of parcels (brain
regions identified by ``parcel_labels``).  For each parcel *p*:

1. Extract the induced subgraph (nodes whose ``parcel_labels == p`` and
   edges whose both endpoints are in *p*).
2. Build the **symmetrically-normalised Laplacian**
   :math:`L_p = I - D_p^{-1/2} A_p D_p^{-1/2}`.
3. Compute the :math:`k` eigenvectors corresponding to the :math:`k` smallest
   eigenvalues (**bottom-k spectral basis** :math:`U_k \\in \\mathbb{R}^{N_p \\times k}`).
4. Project the parcel node features onto this basis:
   :math:`Z_p = U_k^\\top X_p \\in \\mathbb{R}^{k \\times F}` (flattened to
   :math:`k \\cdot F`).

The backbone returns ``(B, n_parcels, k·F)`` — one spectral projection per
parcel per sample.  A downstream
:class:`~diff_benchmark.models.utils_models.additive_parcel_head.AdditiveParcelHead`
applies a **per-parcel linear layer** and optionally group-regularises the
weights, making each parcel's contribution to the scalar prediction directly
readable.

Because the mesh topology is preserved across samples, spectral bases
:math:`U_k` are cached after the first computation.

Usage
-----
::

    from diff_benchmark.models.mesh_models.spectral_laplacian_model import (
        SpectralLaplacianAdditiveModel,
    )
    from diff_benchmark.models.utils_models.additive_parcel_head import (
        AdditiveParcelHead,
    )

    backbone = SpectralLaplacianAdditiveModel(in_features=1, n_spectral_components=16)

    X = [
        {
            "node_features":  torch.randn(64984, 1),
            "parcel_labels":  torch.randint(0, 148, (64984,)),
            "edge_index":     edge_index,   # LongTensor (2, E)
        },
        ...
    ]

    emb = backbone(X)   # (B, n_parcels, k*F)

    head = AdditiveParcelHead(
        embed_dim=backbone.parcel_embed_dim,
        output_dim=1,
        reg_type="group_lasso",
        lambda1=1e-3,
    )
    pred = head(emb)    # (B, 1)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_normalized_laplacian(
    edge_index_local: torch.Tensor, n_nodes: int, device: torch.device
) -> torch.Tensor:
    """Build the dense symmetric-normalised Laplacian for a small subgraph.

    Args:
        edge_index_local: LongTensor (2, E_p) with **local** node indices in
            [0, n_nodes).
        n_nodes: Number of nodes in the subgraph.
        device: Compute device.

    Returns:
        Dense FloatTensor (n_nodes, n_nodes) — the normalised Laplacian.
    """
    # Degree vector
    deg = torch.zeros(n_nodes, device=device, dtype=torch.float32)
    if edge_index_local.numel() > 0:
        src = edge_index_local[0]
        deg.scatter_add_(0, src, torch.ones(src.shape[0], device=device))

    # Avoid division by zero for isolated nodes
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

    # Dense adjacency
    A = torch.zeros(n_nodes, n_nodes, device=device, dtype=torch.float32)
    if edge_index_local.numel() > 0:
        A[edge_index_local[0], edge_index_local[1]] = 1.0

    # L = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L = torch.eye(n_nodes, device=device) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def _spectral_basis(L: torch.Tensor, k: int) -> torch.Tensor:
    """Return the k eigenvectors of L corresponding to its k smallest eigenvalues.

    Uses ``torch.linalg.eigh`` (assumes symmetric matrix) for numerical
    stability.

    Args:
        L: Symmetric FloatTensor (n, n).
        k: Number of eigenvectors to keep.

    Returns:
        FloatTensor (n, k) — column-wise eigenvectors.
    """
    k_actual = min(k, L.shape[0])
    # eigh returns eigenvalues in ascending order
    _, eigvecs = torch.linalg.eigh(L)  # (n,), (n, n)
    return eigvecs[:, :k_actual]  # (n, k_actual)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SpectralLaplacianAdditiveModel(nn.Module):
    """Spectral backbone: returns per-parcel Laplacian projections ``(B, P, k·F)``.

    Each parcel's node features are projected onto the bottom-k eigenvectors
    of its subgraph Laplacian, yielding a fixed-size spectral embedding of
    length ``k·F`` regardless of parcel size.  A separate
    :class:`~diff_benchmark.models.utils_models.additive_parcel_head.AdditiveParcelHead`
    consumes this output and applies a linear, interpretable mapping.

    Parameters
    ----------
    in_features:
        Number of scalar features per vertex (``F``).
    n_spectral_components:
        Number of Laplacian eigenvectors (``k``) per parcel.
    parcel_ids:
        Optional list of integer parcel IDs.  Inferred lazily from the
        first batch if not provided.
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

        # Sorted list of parcel IDs; set lazily if not provided.
        self._parcel_ids: Optional[List[int]] = None
        if parcel_ids is not None:
            self._parcel_ids = sorted(int(p) for p in parcel_ids)

        # Spectral basis cache: parcel_id → FloatTensor (N_p, k)
        self._spectral_cache: Dict[int, torch.Tensor] = {}

        # Dummy parameter so the module has a device and state_dict entry.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parcel_embed_dim(self) -> int:
        """Dimension of each parcel's spectral embedding: ``k · F``."""
        return self._parcel_embed_dim

    @property
    def n_parcels(self) -> Optional[int]:
        """Number of parcels, or ``None`` if not yet initialised."""
        return len(self._parcel_ids) if self._parcel_ids is not None else None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _maybe_init_from_batch(self, batch: List[Dict[str, torch.Tensor]]) -> None:
        """Lazy initialisation: infer parcel IDs from the first batch."""
        if self._parcel_ids is not None:
            return
        parcel_labels = batch[0]["parcel_labels"]
        unique_ids = parcel_labels.unique().tolist()
        self._parcel_ids = sorted(int(p) for p in unique_ids)
        log.debug("Lazily initialised %d parcel IDs.", len(self._parcel_ids))

    # ------------------------------------------------------------------
    # Spectral basis (cached)
    # ------------------------------------------------------------------

    def _get_spectral_basis(
        self,
        parcel_mask: torch.Tensor,
        edge_index: torch.Tensor,
        parcel_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the (possibly cached) spectral basis for *parcel_id*.

        Args:
            parcel_mask: Boolean mask of shape (N,) — True for nodes in this
                parcel.
            edge_index: Global edge index LongTensor (2, E).
            parcel_id: Integer parcel identifier.
            device: Target device.

        Returns:
            FloatTensor (N_p, k) — the k smallest eigenvectors of L_p.
        """
        if parcel_id not in self._spectral_cache:
            # Build subgraph edge index with local (zero-based) node indices
            global_ids = parcel_mask.nonzero(as_tuple=False).squeeze(1)  # (N_p,)
            n_nodes = global_ids.shape[0]

            if edge_index.numel() > 0:
                # Keep only edges where both endpoints are in the parcel
                in_parcel = parcel_mask[edge_index[0]] & parcel_mask[edge_index[1]]
                local_edges = edge_index[:, in_parcel]  # (2, E_p)
                # Remap global → local indices
                remap = torch.full(
                    (parcel_mask.shape[0],), -1, dtype=torch.long, device=device
                )
                remap[global_ids] = torch.arange(n_nodes, device=device)
                local_edges = remap[local_edges]  # still (2, E_p)
            else:
                local_edges = torch.zeros((2, 0), dtype=torch.long, device=device)

            L = _build_normalized_laplacian(local_edges, n_nodes, device)
            basis = _spectral_basis(L, self.k)  # (N_p, k)
            self._spectral_cache[parcel_id] = basis.detach()
            log.debug(
                "Cached spectral basis for parcel %d  (N_p=%d, k=%d).",
                parcel_id,
                n_nodes,
                basis.shape[1],
            )

        return self._spectral_cache[parcel_id].to(device)

    # ------------------------------------------------------------------
    # Core per-sample computation
    # ------------------------------------------------------------------

    def _project_sample(
        self, sample: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[int, torch.Tensor]:
        """Compute spectral projections for all parcels in one sample.

        Args:
            sample: Mesh dict with keys ``node_features`` (N, F),
                ``parcel_labels`` (N,), ``edge_index`` (2, E).
            device: Compute device.

        Returns:
            Dict mapping each parcel ID to a FloatTensor of shape
            ``(parcel_embed_dim,)`` = ``(k·F,)`` (zero-padded for small parcels).
        """
        node_features = sample["node_features"].to(device)   # (N, F)
        parcel_labels = sample["parcel_labels"].to(device)   # (N,)
        edge_index    = sample["edge_index"].to(device)      # (2, E)

        if node_features.shape[1] != self.in_features:
            raise ValueError(
                f"Expected in_features={self.in_features}, "
                f"got {node_features.shape[1]}."
            )

        projections: Dict[int, torch.Tensor] = {}

        for pid in self._parcel_ids:  # type: ignore[union-attr]
            mask = parcel_labels == pid
            n_nodes_p = int(mask.sum().item())

            if n_nodes_p == 0:
                projections[pid] = torch.zeros(self._parcel_embed_dim, device=device)
                continue

            U_k = self._get_spectral_basis(mask, edge_index, pid, device)

            if U_k.shape[0] != n_nodes_p:
                log.warning(
                    "Parcel %d: cached basis size %d != current size %d — recomputing.",
                    pid, U_k.shape[0], n_nodes_p,
                )
                del self._spectral_cache[pid]
                U_k = self._get_spectral_basis(mask, edge_index, pid, device)

            X_p   = node_features[mask]   # (N_p, F)
            Z_p   = U_k.T @ X_p           # (k_actual, F)
            z_flat = Z_p.reshape(-1)      # (k_actual * F,)

            # Zero-pad if the parcel has fewer nodes than k
            if z_flat.shape[0] < self._parcel_embed_dim:
                pad = torch.zeros(
                    self._parcel_embed_dim - z_flat.shape[0], device=device
                )
                z_flat = torch.cat([z_flat, pad])

            projections[pid] = z_flat

        return projections

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: Union[List[Dict[str, torch.Tensor]], torch.Tensor]
    ) -> torch.Tensor:
        """Compute per-parcel spectral embeddings for a batch.

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
        self._maybe_init_from_batch(x)

        batch_tensors = []
        for sample in x:
            proj = self._project_sample(sample, device)
            parcel_stack = torch.stack(
                [proj[pid] for pid in self._parcel_ids], dim=0
            )  # (n_parcels, k*F)
            batch_tensors.append(parcel_stack)

        return torch.stack(batch_tensors, dim=0)  # (B, n_parcels, k*F)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def forward_contributions(
        self, x: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[int, torch.Tensor]]:
        """Return per-parcel spectral projections for the whole batch.

        Args:
            x: List of mesh dicts (same format as :meth:`forward`).

        Returns:
            List of length B.  Each element maps
            ``parcel_id (int) -> FloatTensor(parcel_embed_dim)``.
        """
        device = self._dummy.device
        self._maybe_init_from_batch(x)
        return [self._project_sample(sample, device) for sample in x]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._parcel_ids) if self._parcel_ids else "?"
        return (
            f"SpectralLaplacianAdditiveModel("
            f"in_features={self.in_features}, "
            f"k={self.k}, "
            f"parcel_embed_dim={self._parcel_embed_dim}, "
            f"n_parcels={n})"
        )

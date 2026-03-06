"""Linear additive prediction head for parcel-structured embeddings.

Given a backbone that produces per-parcel spectral embeddings
``(B, P, E)`` — B samples, P parcels, E-dimensional embedding — this head
computes the prediction as a **sum of independent linear contributions**,
one per parcel:

.. math::

    \\hat{y}_b = \\sum_{p=1}^{P} W_p \\, z_{b,p} + \\mathbf{b}

where :math:`W_p \\in \\mathbb{R}^{C \\times E}` is the weight matrix for
parcel *p*, :math:`z_{b,p} \\in \\mathbb{R}^{E}` is sample *b*'s embedding
for parcel *p*, and :math:`\\mathbf{b} \\in \\mathbb{R}^{C}` is a global bias.
``C = 1`` for regression and ``C = n_classes`` for classification.

The **number of parcels** need not be known at construction time: the weight
matrix is allocated lazily on the first forward call.

Regularisation
--------------
Because the parcel weights naturally form groups, the head supports two
structured penalties that encourage parcel-level sparsity (driving whole
parcels to zero rather than individual weights):

* **Group lasso** (``reg_type="group_lasso"``)

  .. math::  \\lambda_1 \\sum_p \\|W_p\\|_F

* **Group elastic net** (``reg_type="group_elastic_net"``)

  .. math::  \\lambda_1 \\sum_p \\|W_p\\|_F + \\lambda_2 \\|W\\|_F^2

The penalty is available via :meth:`regularization_loss` and is added
automatically to the training loss when the trainer detects the method on
the model (see :class:`~diff_benchmark.models.utils_models.trainer.TorchTrainer`).

Interpretability
----------------
:meth:`parcel_contributions` returns each parcel's individual output
``W_p z_{b,p} \\in \\mathbb{R}^{C}`` before summation, giving a direct,
additive decomposition of the prediction into parcel contributions.

Usage
-----
::

    from diff_benchmark.models.utils_models.additive_parcel_head import (
        AdditiveParcelHead,
        build_additive_parcel_head,
    )

    head = AdditiveParcelHead(embed_dim=16, output_dim=1,
                              reg_type="group_lasso", lambda1=1e-3)
    x = torch.randn(4, 148, 16)   # (B, P, E)
    pred = head(x)                # (4, 1)
    reg  = head.regularization_loss()
    loss = mse(pred.squeeze(), y) + reg
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

RegType = Literal["none", "group_lasso", "group_elastic_net"]


class AdditiveParcelHead(nn.Module):
    """Linear prediction head with per-parcel weights and group regularisation.

    Parameters
    ----------
    embed_dim:
        Dimensionality of each parcel's input embedding ``E``
        (= ``k * in_features`` from the backbone).
    output_dim:
        Number of output units: ``1`` for regression, ``n_classes`` for
        classification.
    reg_type:
        Regularisation strategy applied to the parcel weight groups.
        One of ``"none"``, ``"group_lasso"``, ``"group_elastic_net"``.
    lambda1:
        Coefficient for the group-norm penalty (group lasso / elastic net).
    lambda2:
        Coefficient for the squared Frobenius penalty (elastic net only).
    bias:
        Whether to include a global output bias ``b \\in \\mathbb{R}^C``.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        reg_type: RegType = "none",
        lambda1: float = 1e-3,
        lambda2: float = 1e-4,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.reg_type = reg_type
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Lazily allocated when n_parcels becomes known.
        self._W: Optional[nn.Parameter] = None   # (P, C, E)
        self._n_parcels: Optional[int] = None

        self._bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(output_dim)) if bias else None
        )

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _maybe_init_weights(self, n_parcels: int, device: torch.device) -> None:
        """Allocate weight matrix on first call once P is known."""
        if self._W is not None:
            return
        self._n_parcels = n_parcels
        W = torch.empty(n_parcels, self.output_dim, self.embed_dim)
        nn.init.xavier_uniform_(W.view(n_parcels * self.output_dim, self.embed_dim))
        self._W = nn.Parameter(W)
        if self._bias is not None:
            self._bias = self._bias.to(device)
        log.debug(
            "AdditiveParcelHead: allocated W (%d, %d, %d).",
            n_parcels, self.output_dim, self.embed_dim,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the additive linear prediction.

        Args:
            x: FloatTensor ``(B, P, E)`` — batch of per-parcel embeddings.

        Returns:
            FloatTensor ``(B, C)`` — raw logits or regression values.
        """
        if x.dim() != 3:
            raise ValueError(
                f"AdditiveParcelHead expects input of shape (B, P, E), got {x.shape}."
            )
        B, P, E = x.shape
        if E != self.embed_dim:
            raise ValueError(
                f"embed_dim mismatch: expected {self.embed_dim}, got {E}."
            )

        device = x.device
        self._maybe_init_weights(P, device)

        assert self._W is not None  # mypy / type checkers
        W = self._W.to(device)     # (P, C, E)

        # out[b, c] = sum_p  W[p, c, :] · x[b, p, :]
        out = torch.einsum("pce,bpe->bc", W, x)  # (B, C)

        if self._bias is not None:
            out = out + self._bias.to(device)

        return out

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def regularization_loss(self) -> torch.Tensor:
        """Compute the group-structured penalty on the parcel weights.

        Returns:
            Scalar tensor (zero when ``reg_type="none"`` or weights not yet
            initialised).
        """
        if self._W is None or self.reg_type == "none":
            return torch.tensor(0.0, dtype=torch.float32)

        W = self._W  # (P, C, E)

        # Per-parcel Frobenius norms: (P,)
        group_norms = W.norm(dim=(1, 2))

        if self.reg_type == "group_lasso":
            return self.lambda1 * group_norms.sum()

        if self.reg_type == "group_elastic_net":
            return (
                self.lambda1 * group_norms.sum()
                + self.lambda2 * W.pow(2).sum()
            )

        return torch.tensor(0.0, device=W.device, dtype=W.dtype)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def parcel_contributions(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Return per-parcel linear contributions before summation.

        Args:
            x: FloatTensor ``(B, P, E)``.

        Returns:
            FloatTensor ``(B, P, C)`` — each parcel's contribution
            :math:`W_p z_{b,p}` (bias excluded).  Summing over ``dim=1``
            and adding the bias recovers :meth:`forward` exactly.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (B, P, E), got {x.shape}.")
        device = x.device
        self._maybe_init_weights(x.shape[1], device)
        assert self._W is not None
        W = self._W.to(device)  # (P, C, E)
        # contributions[b, p, c] = W[p, c, :] · x[b, p, :]
        return torch.einsum("pce,bpe->bpc", W, x)  # (B, P, C)

    def parcel_scalar_contributions(
        self, x: torch.Tensor, parcel_ids: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """Return per-parcel contributions as a dict keyed by parcel index.

        Args:
            x: FloatTensor ``(B, P, E)``.
            parcel_ids: Optional list of length P giving the integer parcel
                IDs corresponding to each slice of the parcel axis.  If
                ``None``, uses ``[0, 1, ..., P-1]``.

        Returns:
            Dict ``{parcel_id: FloatTensor(B, C)}`` with one entry per parcel.
        """
        contribs = self.parcel_contributions(x)  # (B, P, C)
        P = contribs.shape[1]
        ids = parcel_ids if parcel_ids is not None else list(range(P))
        return {ids[p]: contribs[:, p, :] for p in range(P)}

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        P = self._n_parcels if self._n_parcels is not None else "?"
        return (
            f"AdditiveParcelHead("
            f"embed_dim={self.embed_dim}, "
            f"output_dim={self.output_dim}, "
            f"n_parcels={P}, "
            f"reg={self.reg_type})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_additive_parcel_head(
    embed_dim: int,
    prediction_task: str,
    reg_type: RegType = "none",
    lambda1: float = 1e-3,
    lambda2: float = 1e-4,
    bias: bool = True,
    **_kwargs,
) -> AdditiveParcelHead:
    """Build an :class:`AdditiveParcelHead` for the given task.

    Args:
        embed_dim: Dimensionality of each parcel embedding (``k * F`` from
            the backbone).
        prediction_task: ``"regression"`` or ``"binary_classification"``.
        reg_type: One of ``"none"``, ``"group_lasso"``, ``"group_elastic_net"``.
        lambda1: Group-norm penalty coefficient.
        lambda2: Squared-norm penalty coefficient (elastic net only).
        bias: Whether to include a global bias term.

    Returns:
        Configured :class:`AdditiveParcelHead`.

    Raises:
        ValueError: If *prediction_task* is unrecognised.
    """
    if prediction_task == "regression":
        output_dim = 1
    elif prediction_task == "binary_classification":
        output_dim = 2
    else:
        raise ValueError(
            f"Unknown prediction_task '{prediction_task}'. "
            "Expected 'regression' or 'binary_classification'."
        )

    return AdditiveParcelHead(
        embed_dim=embed_dim,
        output_dim=output_dim,
        reg_type=reg_type,
        lambda1=lambda1,
        lambda2=lambda2,
        bias=bias,
    )

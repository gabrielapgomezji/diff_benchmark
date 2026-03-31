from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


class BaseRegionPositionalEncoding(nn.Module):
    """Base interface for region-aware positional encoders.

    All concrete encoders must map point coordinates and per-point region IDs
    to per-point encoding vectors with shape ``(B, N, D)``.
    """

    def __init__(self, include_size: bool = False) -> None:
        super().__init__()
        self.include_size = include_size
        self.out_dim = 0

    def forward(self, points: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        """Return per-point region encoding.

        Args:
            points: Tensor of shape ``(B, N, 3)``.
            region_ids: Tensor of shape ``(B, N)`` with integer region labels.

        Returns:
            Tensor of shape ``(B, N, D)``.
        """
        raise NotImplementedError


def normalize_points_global(points: torch.Tensor) -> torch.Tensor:
    """Normalize points globally per batch.

    Args:
        points: Tensor of shape ``(B, N, 3)``.

    Returns:
        Globally normalized points with shape ``(B, N, 3)``.
    """
    mean = points.mean(dim=1, keepdim=True)
    std = points.std(dim=1, keepdim=True, unbiased=False) + 1e-6
    return (points - mean) / std


def _validate_inputs(points: torch.Tensor, region_ids: torch.Tensor) -> None:
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
    if region_ids.dim() != 2:
        raise ValueError(f"region_ids must have shape (B, N), got {tuple(region_ids.shape)}")
    if points.shape[:2] != region_ids.shape:
        raise ValueError(
            "points and region_ids dimensions must match on (B, N), "
            f"got {tuple(points.shape)} vs {tuple(region_ids.shape)}"
        )


class AnatomicalRegionEncoding(BaseRegionPositionalEncoding):
    """Per-region anatomical encoding based on centroid and optional region size.

    For each region label within each sample, this computes one encoding vector
    and broadcasts it to all points in that region.
    """

    def __init__(self, include_size: bool = False) -> None:
        super().__init__(include_size=include_size)
        self.out_dim = 3 + int(self.include_size)

    def forward(self, points: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        _validate_inputs(points, region_ids)
        points = normalize_points_global(points)
        bsz, num_points, _ = points.shape
        device = points.device
        dtype = points.dtype
        out = torch.zeros((bsz, num_points, self.out_dim), device=device, dtype=dtype)

        for b in range(bsz):
            points_b = points[b]
            region_ids_b = region_ids[b].to(torch.long)

            _, inverse, counts = torch.unique(
                region_ids_b,
                sorted=False,
                return_inverse=True,
                return_counts=True,
            )

            num_regions = counts.numel()
            sums = torch.zeros((num_regions, 3), device=device, dtype=dtype)
            sums.index_add_(0, inverse, points_b)

            centroids = sums / counts.to(dtype).unsqueeze(1)
            encoded = centroids[inverse]

            if self.include_size:
                size_norm = counts[inverse].to(dtype).unsqueeze(1) / float(num_points)
                encoded = torch.cat([encoded, size_norm], dim=1)

            out[b] = encoded

        return out


class AnatomicalRelativeRegionEncoding(BaseRegionPositionalEncoding):
    """Per-point encoding with centroid, relative position, and optional region size.

    For each point ``x_i`` in region ``R``:
    ``e_i = [centroid(R), x_i - centroid(R)]`` and optionally append
    ``size(R) / total_points``.
    """

    def __init__(self, include_size: bool = False) -> None:
        super().__init__(include_size=include_size)
        self.out_dim = 6 + int(self.include_size)

    def forward(self, points: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        _validate_inputs(points, region_ids)
        points = normalize_points_global(points)
        bsz, num_points, _ = points.shape
        device = points.device
        dtype = points.dtype
        out = torch.zeros((bsz, num_points, self.out_dim), device=device, dtype=dtype)

        for b in range(bsz):
            points_b = points[b]
            region_ids_b = region_ids[b].to(torch.long)

            _, inverse, counts = torch.unique(
                region_ids_b,
                sorted=False,
                return_inverse=True,
                return_counts=True,
            )

            num_regions = counts.numel()
            sums = torch.zeros((num_regions, 3), device=device, dtype=dtype)
            sums.index_add_(0, inverse, points_b)
            centroids = sums / counts.to(dtype).unsqueeze(1)
            centroids_per_point = centroids[inverse]

            rel = points_b - centroids_per_point
            sq_sums = torch.zeros((num_regions, 3), device=device, dtype=dtype)
            sq_sums.index_add_(0, inverse, rel * rel)
            rel_std = torch.sqrt(sq_sums / counts.to(dtype).unsqueeze(1) + 1e-6)
            rel_norm = rel / rel_std[inverse]

            encoded = torch.cat([centroids_per_point, rel_norm], dim=1)

            if self.include_size:
                size_norm = counts[inverse].to(dtype).unsqueeze(1) / float(num_points)
                encoded = torch.cat([encoded, size_norm], dim=1)

            out[b] = encoded

        return out


def build_region_encoder(config: Mapping[str, Any] | None) -> BaseRegionPositionalEncoding | None:
    """Build a region positional encoder from config.

    Supported values:
    - ``None``: disable encoding
    - ``{"type": "none"}``: disable encoding
    - ``{"type": "anatomical"}``
    - ``{"type": "anatomical_relative"}``

    Optional key:
    - ``include_size`` (bool): append normalized region size
    """
    if config is None:
        return None

    cfg = dict(config)
    enc_type = str(cfg.get("type", "none")).lower()
    include_size = bool(cfg.get("include_size", False))

    if enc_type in {"none", "null", "off"}:
        return None

    if enc_type == "anatomical":
        return AnatomicalRegionEncoding(include_size=include_size)

    if enc_type == "anatomical_relative":
        return AnatomicalRelativeRegionEncoding(include_size=include_size)

    raise ValueError(f"Unknown region encoder type: {enc_type}")

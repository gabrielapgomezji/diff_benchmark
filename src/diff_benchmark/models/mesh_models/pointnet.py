from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from diff_benchmark.models.utils_models.region_positional_encoding import (
	BaseRegionPositionalEncoding,
	build_region_encoder,
)


def _chunked_region_knn(
	points_r: torch.Tensor,
	k: int,
	*,
	include_self: bool,
	query_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Compute region-local kNN without materializing an (Nr, Nr) distance matrix.

	For a region with ``Nr`` points, this computes distances in query chunks of
	size ``Q`` and therefore allocates at most ``(Q, Nr)`` temporary distances,
	not ``(Nr, Nr)``.
	"""
	device = points_r.device
	nr = points_r.shape[0]
	k_out = k
	k_eff = min(k_out, nr if include_self else max(nr - 1, 0))

	self_local = torch.arange(nr, device=device)
	neighbor_local_idx = self_local.unsqueeze(1).expand(-1, k_out).clone()
	neighbor_valid_mask = torch.zeros((nr, k_out), dtype=torch.bool, device=device)

	if k_eff <= 0:
		return neighbor_local_idx, neighbor_valid_mask

	for start in range(0, nr, query_chunk_size):
		end = min(start + query_chunk_size, nr)
		chunk_points = points_r[start:end]

		# Memory-efficient KNN: only build distances for (chunk, region), not
		# the full region-by-region matrix.
		chunk_dist = torch.cdist(chunk_points, points_r)

		if not include_self:
			row = torch.arange(end - start, device=device)
			col = torch.arange(start, end, device=device)
			chunk_dist[row, col] = float("inf")

		knn_dist, knn_local = torch.topk(chunk_dist, k=k_eff, dim=-1, largest=False)
		knn_valid = torch.isfinite(knn_dist)

		chunk_self = self_local[start:end].unsqueeze(1)
		knn_local = torch.where(knn_valid, knn_local, chunk_self)

		neighbor_local_idx[start:end, :k_eff] = knn_local
		neighbor_valid_mask[start:end, :k_eff] = knn_valid

	return neighbor_local_idx, neighbor_valid_mask


def _chunked_region_radius_knn(
	points_r: torch.Tensor,
	radius: float,
	max_neighbors: int,
	*,
	include_self: bool,
	query_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Radius-constrained neighbor search in chunks, within one region."""
	device = points_r.device
	nr = points_r.shape[0]
	k_out = max_neighbors
	k_eff = min(k_out, nr if include_self else max(nr - 1, 0))

	self_local = torch.arange(nr, device=device)
	neighbor_local_idx = self_local.unsqueeze(1).expand(-1, k_out).clone()
	neighbor_valid_mask = torch.zeros((nr, k_out), dtype=torch.bool, device=device)

	if k_eff <= 0:
		return neighbor_local_idx, neighbor_valid_mask

	for start in range(0, nr, query_chunk_size):
		end = min(start + query_chunk_size, nr)
		chunk_points = points_r[start:end]
		chunk_dist = torch.cdist(chunk_points, points_r)

		if not include_self:
			row = torch.arange(end - start, device=device)
			col = torch.arange(start, end, device=device)
			chunk_dist[row, col] = float("inf")

		valid_candidates = chunk_dist <= radius
		masked_dist = chunk_dist.masked_fill(~valid_candidates, float("inf"))

		knn_dist, knn_local = torch.topk(masked_dist, k=k_eff, dim=-1, largest=False)
		knn_valid = torch.isfinite(knn_dist)

		chunk_self = self_local[start:end].unsqueeze(1)
		knn_local = torch.where(knn_valid, knn_local, chunk_self)

		neighbor_local_idx[start:end, :k_eff] = knn_local
		neighbor_valid_mask[start:end, :k_eff] = knn_valid

	return neighbor_local_idx, neighbor_valid_mask


def region_constrained_grouping(
	points: torch.Tensor,
	regions: torch.Tensor,
	features: Optional[torch.Tensor] = None,
	*,
	k: Optional[int] = 16,
	radius: Optional[float] = None,
	max_neighbors: Optional[int] = None,
	include_self: bool = True,
	query_chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
	"""Group local neighbors with a hard same-region constraint.

	This implementation never materializes a global ``(N, N)`` distance matrix.
	Instead, it loops over regions and computes distances in query chunks for
	each region only. This enforces the region constraint *before* neighbor
	search and keeps memory bounded by roughly ``O(Q * N_r)`` per region chunk,
	not ``O(N^2)`` for the full mesh.

	Args:
		points: Point coordinates with shape (N, 3) or (B, N, 3).
		regions: Integer region IDs with shape (N,) or (B, N).
		features: Optional per-point features with shape (N, C) or (B, N, C).
		k: Number of neighbors for k-NN mode.
		radius: Radius threshold for radius mode.
		max_neighbors: Maximum neighbors returned in radius mode.
		include_self: If True, a point can always include itself.
		query_chunk_size: Number of query points processed at once per region.

	Returns:
		Tuple of:
			neighbor_idx: (N, K) or (B, N, K)
			neighbor_valid_mask: (N, K) or (B, N, K)
			grouped_points: (N, K, 3) or (B, N, K, 3)
			grouped_features: (N, K, C) or (B, N, K, C) if features are provided, else None

	Raises:
		ValueError: If both k and radius modes are misconfigured.
	"""
	if points.dim() not in (2, 3):
		raise ValueError(f"points must have shape (N, 3) or (B, N, 3), got {tuple(points.shape)}")
	if regions.dim() not in (1, 2):
		raise ValueError(f"regions must have shape (N,) or (B, N), got {tuple(regions.shape)}")

	squeezed = points.dim() == 2
	if squeezed:
		points = points.unsqueeze(0)
		regions = regions.unsqueeze(0)
		if features is not None:
			features = features.unsqueeze(0)

	if points.shape[-1] != 3:
		raise ValueError(f"Last dimension of points must be 3, got {points.shape[-1]}")
	if points.shape[:2] != regions.shape[:2]:
		raise ValueError(
			f"points and regions batch/node dimensions must match, got {tuple(points.shape)} vs {tuple(regions.shape)}"
		)
	if features is not None and features.shape[:2] != points.shape[:2]:
		raise ValueError(
			f"features must match points batch/node dimensions, got {tuple(features.shape)} vs {tuple(points.shape)}"
		)

	if radius is None and k is None:
		raise ValueError("Either k (k-NN) or radius must be provided")

	if query_chunk_size <= 0:
		raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")

	bsz, num_points, _ = points.shape
	device = points.device

	if radius is not None:
		if max_neighbors is None:
			max_neighbors = k if k is not None else 16
		if max_neighbors <= 0:
			raise ValueError(f"max_neighbors must be positive, got {max_neighbors}")
		k_out = max_neighbors
	else:
		if k is None or k <= 0:
			raise ValueError(f"k must be a positive integer for k-NN mode, got {k}")
		k_out = k

	# Pre-fill with self-neighbors so tiny regions (< k points) are naturally
	# padded without extra allocations.
	self_idx = torch.arange(num_points, device=device).view(1, num_points, 1)
	neighbor_idx = self_idx.expand(bsz, num_points, k_out).clone()
	neighbor_valid_mask = torch.zeros((bsz, num_points, k_out), dtype=torch.bool, device=device)

	for b in range(bsz):
		batch_regions = regions[b]
		for region_id in torch.unique(batch_regions):
			region_point_idx = torch.nonzero(batch_regions == region_id, as_tuple=False).squeeze(1)
			if region_point_idx.numel() == 0:
				continue

			points_r = points[b, region_point_idx]

			# Region constraint is enforced here by construction: neighbors are only
			# searched over `points_r`, i.e. points belonging to this region.
			if radius is not None:
				local_idx, local_valid = _chunked_region_radius_knn(
					points_r=points_r,
					radius=radius,
					max_neighbors=k_out,
					include_self=include_self,
					query_chunk_size=query_chunk_size,
				)
			else:
				local_idx, local_valid = _chunked_region_knn(
					points_r=points_r,
					k=k_out,
					include_self=include_self,
					query_chunk_size=query_chunk_size,
				)

			global_idx = region_point_idx[local_idx]
			neighbor_idx[b, region_point_idx] = global_idx
			neighbor_valid_mask[b, region_point_idx] = local_valid

	batch_idx = torch.arange(bsz, device=device).view(bsz, 1, 1).expand_as(neighbor_idx)
	grouped_points = points[batch_idx, neighbor_idx]  # (B, N, K, 3)

	grouped_features = None
	if features is not None:
		grouped_features = features[batch_idx, neighbor_idx]  # (B, N, K, C)

	if squeezed:
		neighbor_idx = neighbor_idx.squeeze(0)
		neighbor_valid_mask = neighbor_valid_mask.squeeze(0)
		grouped_points = grouped_points.squeeze(0)
		if grouped_features is not None:
			grouped_features = grouped_features.squeeze(0)

	return neighbor_idx, neighbor_valid_mask, grouped_points, grouped_features


class RegionConstrainedPointNetPP(nn.Module):
	"""PointNet++-style backbone with strict within-region neighborhood grouping.

	Input mesh samples are expected to contain:

	- ``vertices``: (N, 3)
	- ``parcel_labels`` or ``parcel_labels``: (N,)
	- ``node_features``: (N, F), optional

	During local aggregation, each center point can only aggregate neighbors from
	the same anatomical region label.
	"""

	data_type: str = "mesh"
	collate_fn: Optional[object] = None

	def __init__(
		self,
		in_features: int = 1,
		hidden_dim: int = 128,
		k_neighbors: int = 16,
		radius: Optional[float] = None,
		max_neighbors: int = 16,
		dropout: float = 0.0,
		region_encoder: Optional[BaseRegionPositionalEncoding] = None,
		region_encoder_config: Optional[Dict[str, object]] = None,
		**kwargs,
	) -> None:
		super().__init__()
		self.in_features = in_features
		self.hidden_dim = hidden_dim
		self._inferred_num_regions: Optional[int] = None
		self.k_neighbors = k_neighbors
		self.radius = radius
		self.max_neighbors = max_neighbors
		self.out_dim = hidden_dim
		self.query_chunk_size = int(kwargs.get("query_chunk_size", 2048))
		self.region_encoder = region_encoder
		if isinstance(self.region_encoder, Mapping):
			self.region_encoder = build_region_encoder(self.region_encoder)
		elif self.region_encoder is None:
			cfg = region_encoder_config if region_encoder_config is not None else kwargs.get("region_encoder", None)
			if isinstance(cfg, Mapping):
				self.region_encoder = build_region_encoder(cfg)

		self.region_encoding_dim = 0
		if self.region_encoder is not None:
			self.region_encoding_dim = int(self.region_encoder.out_dim)

		local_in_dim = 3 + max(0, in_features) + self.region_encoding_dim
		self.local_mlp = nn.Sequential(
			nn.Linear(local_in_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)
		self.point_mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
		)

	def _to_batched_tensors(
		self,
		x: Union[
			List[Dict[str, torch.Tensor]],
			Dict[str, torch.Tensor],
			Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
			Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
		],
		device: torch.device,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
		"""Convert supported inputs to batched tensors (B, N, *)."""
		if isinstance(x, list):
			points = torch.stack([sample["vertices"].to(device) for sample in x], dim=0)

			region_key = "parcel_labels" if "parcel_labels" in x[0] else "parcel_labels"
			regions = torch.stack([sample[region_key].to(device) for sample in x], dim=0)

			if "node_features" in x[0] and x[0]["node_features"] is not None:
				features = torch.stack([sample["node_features"].to(device) for sample in x], dim=0)
				if features.shape[-1] == 0:
					features = None
			else:
				features = None
			if "neighbor_idx" in x[0] and "neighbor_mask" in x[0]:
				neighbor_idx = torch.stack([sample["neighbor_idx"].to(device) for sample in x], dim=0).long()
				neighbor_mask = torch.stack([sample["neighbor_mask"].to(device) for sample in x], dim=0).bool()
				return points, features, regions, neighbor_idx, neighbor_mask
			return points, features, regions, None, None

		if isinstance(x, dict):
			points = x["vertices"].to(device).unsqueeze(0)
			region_key = "parcel_labels" if "parcel_labels" in x else "parcel_labels"
			regions = x[region_key].to(device).unsqueeze(0)
			features = x.get("node_features")
			if features is not None:
				features = features.to(device).unsqueeze(0)
				if features.shape[-1] == 0:
					features = None
			if "neighbor_idx" in x and "neighbor_mask" in x:
				neighbor_idx = x["neighbor_idx"].to(device).unsqueeze(0).long()
				neighbor_mask = x["neighbor_mask"].to(device).unsqueeze(0).bool()
				return points, features, regions, neighbor_idx, neighbor_mask
			return points, features, regions, None, None

		if isinstance(x, tuple) and len(x) == 3:
			points, features, regions = x
			points = points.to(device)
			regions = regions.to(device)
			if points.dim() == 2:
				points = points.unsqueeze(0)
			if regions.dim() == 1:
				regions = regions.unsqueeze(0)
			if features is not None:
				features = features.to(device)
				if features.dim() == 2:
					features = features.unsqueeze(0)
				if features.shape[-1] == 0:
					features = None
			return points, features, regions, None, None

		if isinstance(x, tuple) and len(x) == 5:
			points, features, regions, neighbor_idx, neighbor_mask = x
			points = points.to(device)
			regions = regions.to(device)
			if points.dim() == 2:
				points = points.unsqueeze(0)
			if regions.dim() == 1:
				regions = regions.unsqueeze(0)
			if features is not None:
				features = features.to(device)
				if features.dim() == 2:
					features = features.unsqueeze(0)
				if features.shape[-1] == 0:
					features = None
			neighbor_idx = neighbor_idx.to(device)
			neighbor_mask = neighbor_mask.to(device)
			if neighbor_idx.dim() == 2:
				neighbor_idx = neighbor_idx.unsqueeze(0)
			if neighbor_mask.dim() == 2:
				neighbor_mask = neighbor_mask.unsqueeze(0)
			return points, features, regions, neighbor_idx.long(), neighbor_mask.bool()

		raise TypeError(
			"Unsupported input type for RegionConstrainedPointNetPP. "
			"Expected list[dict], dict, tuple(points, features, regions), "
			"or tuple(points, features, regions, neighbor_idx, neighbor_mask)."
		)

	def forward(
		self,
		x: Union[
			torch.Tensor,
			List[Dict[str, torch.Tensor]],
			Dict[str, torch.Tensor],
			Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
			Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
		],
	) -> torch.Tensor:
		"""Compute parcel embeddings with strict same-region local aggregation.

		Global pooling over all points would collapse parcel identity and remove
		parcel-wise interpretability. Instead, the same PointNet++ feature
		extractor is shared across all points, then pooled independently per
		region to return ``(B, P, E)`` embeddings suitable for additive parcel
		heads.
		"""
		# Cached-feature fast path: when backbone is frozen and features are
		# precomputed, x arrives as a tensor from CachedFeatureDataset.
		if isinstance(x, torch.Tensor):
			if x.dim() == 3:
				return x
			if x.dim() == 2:
				d = x.shape[-1]
				if d % self.out_dim != 0:
					raise ValueError(
						f"Cached PointNet feature dim {d} is not divisible by out_dim={self.out_dim}."
					)
				p = d // self.out_dim
				return x.view(x.shape[0], p, self.out_dim)
			raise ValueError(f"Unsupported cached tensor shape for PointNet backbone: {tuple(x.shape)}")

		device = next(self.parameters()).device
		points, features, regions, neighbor_idx, neighbor_mask = self._to_batched_tensors(x, device)
	
		if self.region_encoder is not None:
			region_features = self.region_encoder(points, regions)
			if features is None:
				features = region_features
			else:
				features = torch.cat([features, region_features], dim=-1)

		if neighbor_idx is None or neighbor_mask is None:
			neighbor_idx, neighbor_mask, _, _ = region_constrained_grouping(
				points=points,
				regions=regions,
				features=features,
				k=self.k_neighbors if self.radius is None else None,
				radius=self.radius,
				max_neighbors=self.max_neighbors,
				include_self=True,
				query_chunk_size=self.query_chunk_size,
			)

		# Neighborhood grouping is either provided in input tensors or computed
		# on-the-fly; local tuple assembly is then pure tensor indexing.
		batch_idx = torch.arange(points.shape[0], device=device).view(points.shape[0], 1, 1).expand_as(neighbor_idx)
		grouped_points = points[batch_idx, neighbor_idx]

		grouped_features = None
		if features is not None:
			grouped_features = features[batch_idx, neighbor_idx]

		# Local geometric structure: use relative coordinates to describe each
		# neighbor around its center point.
		rel_xyz = grouped_points - points.unsqueeze(2)
		mask_4d = neighbor_mask.unsqueeze(-1)
		rel_xyz = torch.where(mask_4d, rel_xyz, torch.zeros_like(rel_xyz))

		if grouped_features is not None:
			grouped_features = torch.where(mask_4d, grouped_features, torch.zeros_like(grouped_features))
			local_input = torch.cat([rel_xyz, grouped_features], dim=-1)
		else:
			local_input = rel_xyz

		# PointNet++ style local aggregation: MLP on neighbor tuples then max
		# over neighborhood. Because grouping is hard-masked by region IDs,
		# this aggregation captures only within-region spatial patterns.
		local_encoded = self.local_mlp(local_input)  # (B, N, K, H)
		point_encoded = local_encoded.max(dim=2).values  # (B, N, H)
		point_encoded = self.point_mlp(point_encoded)

		# Build a fixed parcel axis P (excluding background label 0) so
		# downstream heads can learn one coefficient group per parcel.
		# P is inferred from labels on first forward pass and then kept fixed.
		max_region_id = int(regions.max().item())
		inferred = max(0, max_region_id)
		if self._inferred_num_regions is None:
			self._inferred_num_regions = inferred
		num_regions = self._inferred_num_regions
		if inferred > num_regions:
			raise ValueError(
				"Encountered region IDs larger than first observed batch. "
				"Ensure label space is stable across batches."
			)

		region_ids = torch.arange(
			1,
			num_regions + 1,
			device=device,
			dtype=regions.dtype,
		)

		bsz = point_encoded.shape[0]
		region_embeddings = torch.zeros(
			bsz, num_regions, self.hidden_dim, device=device, dtype=point_encoded.dtype
		)

		# Shared weights are preserved because all points were encoded by the same
		# local/point MLPs. Only the final pooling is done region-wise.
		for b in range(bsz):
			batch_regions = regions[b]
			for p_idx, region_id in enumerate(region_ids):
				mask = batch_regions == region_id
				if torch.any(mask):
					region_embeddings[b, p_idx] = point_encoded[b, mask].max(dim=0).values

		return region_embeddings


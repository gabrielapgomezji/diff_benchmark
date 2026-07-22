from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Iterable

import hydra
import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from diff_benchmark.data.prepare_data import DatasetPreparation
from diff_benchmark.models.mesh_models.region_feature_extractor import (
	RegionFeatureExtractor,
)
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.utils.utils_brain_feature_extraction import (
	load_template_surface,
	resample_schaefer_onto_fs_lr,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = (
	PROJECT_ROOT
	/ "exp_outputs"
	/ "summary"
	/ "region_correlation_maps"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCHAEFER_LABELS_JSON = PROJECT_ROOT / "aux_materials" / "schaefer_labels.json"

# -------------------------
# Experiment configuration
# -------------------------
DATASET_SELECTION = "hcp" #camcan
MICROSTRUCTURE_SELECTION = "md"
TISSUE_TYPE = "gray"
SURFACE_SPACE = "fslr_32k" #"fsaverage"
SCALE = 100
PREDICTION_TASK = "binary_classification"
MODEL_NAME = "region_group_lasso"  # ensures mesh pipeline

PCA_N_COMPONENTS = 10
# SUMMARY_MODE = "l2"  # l2 | mean | abs_mean

CORR_METRIC = "cosine"  # pearson | cosine
CLUSTER_DISTANCE_THRESHOLD = None #0.1 
N_CLUSTERS = 17 #None  # set to int to specify number of clusters instead of distance threshold
CLUSTER_IMBALANCE_FRAC = 0.9
DISTANCE_MODE = "adjacency_shortest_path"  # adjacency_shortest_path | centroid

EMBEDDINGS = [
	# "flatten",
	"mean_std",
	"summary_stats",
	"percentiles",
	"pca",
	"region_mean",
]

MAX_SUBJECTS = None  # set to int to subsample (e.g., 200)

logger = logging.getLogger(__name__)


@dataclass
class MeshRecord:
	subject_id: str
	node_features: np.ndarray
	parcel_labels: np.ndarray


def _load_cfg():
	with hydra.initialize(version_base="1.3", config_path="pkg://diff_benchmark.configs"):
		overrides = [
			f"dataset.name={DATASET_SELECTION}",
			f"dataset.metric_to_compute={MICROSTRUCTURE_SELECTION}",
			f"dataset.tissue_type={TISSUE_TYPE}",
			f"dataset.scale={SCALE}",
			f"dataset.surface_space={SURFACE_SPACE}",
			f"model.name={MODEL_NAME}",
			f"pred_head.prediction_task={PREDICTION_TASK}",
			"backend.backend=sklearn",
		]
		return hydra.compose(config_name="main", overrides=overrides)


def _load_mesh_records() -> list[MeshRecord]:
	cfg = _load_cfg()
	dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
	cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]
	dataset_selected = DatasetConfig(
		**dataset_cfg,
		base_dir=Path(cluster_cfg.base_dir),
		results_dir=Path(cluster_cfg.results_dir),
	)
	torch_dataset, _ = DatasetPreparation(cfg=cfg, source_dataset=dataset_selected).pipeline()
	mesh_data = getattr(torch_dataset, "_mesh_data", None)
	if not mesh_data:
		raise RuntimeError("Mesh data not available for selected dataset.")

	subject_ids = list(mesh_data.keys())
	if MAX_SUBJECTS is not None:
		subject_ids = subject_ids[: int(MAX_SUBJECTS)]

	records: list[MeshRecord] = []
	for subject_id in subject_ids:
		paths = mesh_data.get(subject_id)
		if not paths:
			continue
		nodes_path = paths.get("nodes")
		if nodes_path is None or not Path(nodes_path).exists():
			continue
		nodes_df = pd.read_parquet(nodes_path, engine="pyarrow")
		feat_cols = sorted(
			[c for c in nodes_df.columns if c.startswith("feature_")],
			key=lambda c: int(c.split("_")[1]),
		)
		if not feat_cols:
			continue
		node_features = nodes_df[feat_cols].to_numpy(dtype=np.float32)
		parcel_labels = nodes_df["parcel_label"].to_numpy(dtype=np.int32)
		records.append(
			MeshRecord(
				subject_id=str(subject_id),
				node_features=node_features,
				parcel_labels=parcel_labels,
			)
		)
	if not records:
		raise RuntimeError("No mesh records loaded; check dataset paths and filters.")
	return records


def _mesh_records_to_list(records: Iterable[MeshRecord]) -> list[dict]:
	return [
		{
			"node_features": r.node_features,
			"parcel_labels": r.parcel_labels,
		}
		for r in records
	]


def _region_slices(extractor: RegionFeatureExtractor) -> tuple[list[int], list[slice]]:
	region_order = list(extractor.region_order_)
	slices: list[slice] = []
	col = 0
	for rid in region_order:
		width = int(extractor.region_feature_widths_[rid])
		slices.append(slice(col, col + width))
		col += width
	return region_order, slices


def _build_region_vectors(
	features: np.ndarray,
	region_order: list[int],
	region_slices: list[slice],
) -> dict[int, np.ndarray]:
	data: dict[int, np.ndarray] = {}
	for rid, sl in zip(region_order, region_slices):
		block = features[:, sl]
		if block.size == 0:
			data[rid] = np.zeros(features.shape[0], dtype=float)
		else:
			data[rid] = block.reshape(block.shape[0] * block.shape[1])
	return data


def _standardize_region_vectors(region_vectors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
	standardized: dict[int, np.ndarray] = {}
	for rid, vec in region_vectors.items():
		mean = float(np.mean(vec))
		std = float(np.std(vec))
		if std == 0.0:
			standardized[rid] = vec.astype(float)
		else:
			standardized[rid] = (vec - mean) / std
	return standardized


def _build_region_mean_vectors(
	mesh_list: list[dict],
	region_order: list[int],
) -> dict[int, np.ndarray]:
	data: dict[int, np.ndarray] = {}
	for rid in region_order:
		means = []
		for mesh in mesh_list:
			labels = np.asarray(mesh["parcel_labels"]).astype(int)
			values = np.asarray(mesh["node_features"], dtype=float)
			mask = labels == int(rid)
			if not np.any(mask):
				means.append(0.0)
			else:
				means.append(float(values[mask].mean()))
		data[rid] = np.asarray(means, dtype=float)
	return data


def _compute_region_similarity(
	region_vectors: dict[int, np.ndarray],
	metric: str,
) -> pd.DataFrame:
	region_ids = list(region_vectors.keys())
	corr = pd.DataFrame(index=region_ids, columns=region_ids, dtype=float)

	for rid_a in region_ids:
		vec_a = region_vectors[rid_a]
		for rid_b in region_ids:
			vec_b = region_vectors[rid_b]
			if vec_a.shape != vec_b.shape:
				corr.loc[rid_a, rid_b] = np.nan
				continue
			if vec_a.size < 2:
				corr.loc[rid_a, rid_b] = np.nan
				continue
			if metric == "pearson":
				if float(np.std(vec_a)) == 0.0 or float(np.std(vec_b)) == 0.0:
					corr.loc[rid_a, rid_b] = np.nan
					continue
				corr.loc[rid_a, rid_b] = float(np.corrcoef(vec_a, vec_b)[0, 1])
			elif metric == "cosine":
				denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
				if denom == 0.0:
					corr.loc[rid_a, rid_b] = np.nan
					continue
				corr.loc[rid_a, rid_b] = float(np.dot(vec_a, vec_b) / denom)
			else:
				raise ValueError("metric must be 'pearson' or 'cosine'")

	return corr


def _load_subnetwork_map() -> dict:
	if not SCHAEFER_LABELS_JSON.exists():
		return {}
	try:
		return json.loads(SCHAEFER_LABELS_JSON.read_text())
	except Exception:
		return {}


def _parse_schaefer_label(raw_name: str, subnetwork_map: dict) -> str:
	name = raw_name.replace("17Networks_", "")
	parts = name.split("_")

	hemi = "L" if parts[0] == "LH" else "R"
	network_full = parts[1]
	network = network_full[:-1] if network_full[-1] in "ABC" else network_full
	subpart = network_full[-1] if network_full[-1] in "ABC" else ""

	sub_key = "_".join(parts[1:-1])
	sub_key_lookup = (
		sub_key.replace("DefaultA", "Default")
		.replace("DefaultB", "Default")
		.replace("DefaultC", "Default")
	)
	region_name = subnetwork_map.get(sub_key_lookup, sub_key_lookup)

	if subpart:
		header = f"{hemi} {network} ({subpart})"
	else:
		header = f"{hemi} {network}"
	return f"{header}\n{region_name}"


def _format_region_name(name: str, max_words_per_line: int = 2) -> str:
	words = name.replace("_", " ").split()
	lines = []
	for i in range(0, len(words), max_words_per_line):
		lines.append(" ".join(words[i : i + max_words_per_line]))
	return "\n".join(lines)


def _load_schaefer_label_map(scale: int, surface_space: str) -> dict[int, str]:
	schaefer = resample_schaefer_onto_fs_lr(scale=scale, target_space=surface_space)
	tsv_path = Path(schaefer["atlas_meta"]["label_tsv_path"])
	df = pd.read_csv(tsv_path, sep="\t")
	df = df[~df["name"].str.contains("Background", na=False)].copy()

	left_ids = np.unique(schaefer["left.data"])
	right_ids = np.unique(schaefer["right.data"])

	left_ids = np.sort(left_ids[left_ids != 0])
	right_ids = np.sort(right_ids[right_ids != 0])

	left_names = df[df["name"].str.startswith("LH_")]["name"]
	right_names = df[df["name"].str.startswith("RH_")]["name"]

	if len(left_ids) + len(right_ids) != len(df):
		print("WARNING: mismatch between atlas parcels and TSV labels")

	label_map = {int(pid): str(name) for pid, name in zip(left_ids, left_names)}
	offset = int(left_ids.max()) if len(left_ids) > 0 else 0
	label_map.update(
		{
			int(pid + offset): str(name)
			for pid, name in zip(right_ids, right_names)
		}
	)
	return label_map


def _region_id_to_name(region_id: int, label_map: dict[int, str]) -> str:
	return label_map.get(int(region_id), f"region_{region_id}")


def _name_regions(region_ids: list[int]) -> list[str]:
	label_map = _load_schaefer_label_map(SCALE, SURFACE_SPACE)
	subnetwork_map = _load_subnetwork_map()
	names = []
	for rid in region_ids:
		raw = _region_id_to_name(rid, label_map)
		parsed = _parse_schaefer_label(raw, subnetwork_map)
		names.append(_format_region_name(parsed, max_words_per_line=3))
	return names


def _plot_corr_matrix(corr_df: pd.DataFrame, title: str, out_file: Path) -> None:
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize=(10, 9))
	im = ax.imshow(corr_df.values.astype(float), cmap="Reds")
	# im = ax.imshow(corr_df.values.astype(float), vmin=-1, vmax=1, cmap="coolwarm")
	ax.set_xticks(range(len(corr_df.columns)))
	ax.set_yticks(range(len(corr_df.index)))
	ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=6)
	ax.set_yticklabels(corr_df.index, fontsize=6)
	ax.set_title(title)
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Cosine similarity")
	fig.tight_layout()
	fig.savefig(out_file, dpi=150)
	plt.close(fig)


def _cluster_regions(
	corr_df: pd.DataFrame,
	*,
	distance_threshold: float | None,
	n_clusters: int | None = None,
) -> np.ndarray:
	if distance_threshold is None and n_clusters is None:
		raise ValueError("Provide distance_threshold or n_clusters for clustering.")
	if distance_threshold is not None and n_clusters is not None:
		logger.warning(
			"Both distance_threshold and n_clusters set; using n_clusters=%s.",
			n_clusters,
		)
		distance_threshold = None
	corr_filled = corr_df.fillna(0.0).astype(float)
	dist = 1.0 - corr_filled
	cluster = AgglomerativeClustering(
		metric="precomputed",
		linkage="average",
		distance_threshold=(
			float(distance_threshold) if distance_threshold is not None else None
		),
		n_clusters=n_clusters,
	)
	return cluster.fit_predict(dist.values)


def _warn_unbalanced_clusters(labels: np.ndarray) -> None:
	unique, counts = np.unique(labels, return_counts=True)
	if len(unique) <= 1:
		logger.warning("Clustering produced a single cluster.")
		return
	frac = counts.max() / counts.sum()
	if frac >= CLUSTER_IMBALANCE_FRAC:
		logger.warning(
			"Clustering is highly imbalanced: largest cluster fraction=%.2f.",
			frac,
		)


def _build_surface_parcel_labels(scale: int, surface_space: str) -> tuple[np.ndarray, int]:
	schaefer = resample_schaefer_onto_fs_lr(scale=scale, target_space=surface_space)
	left_labels = schaefer["left.data"].astype(int)
	right_labels = schaefer["right.data"].astype(int)
	max_left = int(left_labels.max()) if left_labels.size > 0 else 0
	right_offset = right_labels.copy()
	right_offset[right_offset > 0] += max_left
	parcel_labels = np.concatenate([left_labels, right_offset])
	return parcel_labels, len(left_labels)


def _compute_region_centroids(
	*,
	region_ids: list[int],
	scale: int,
	surface_space: str,
) -> dict[int, np.ndarray]:
	left_mesh = load_template_surface(
		hemi="L", space=surface_space, surf_type="midthickness"
	)
	right_mesh = load_template_surface(
		hemi="R", space=surface_space, surf_type="midthickness"
	)
	vertices = np.concatenate([left_mesh[0], right_mesh[0]], axis=0).astype(float)
	parcel_labels, _ = _build_surface_parcel_labels(scale, surface_space)

	centroids: dict[int, np.ndarray] = {}
	for rid in region_ids:
		mask = parcel_labels == int(rid)
		if not np.any(mask):
			centroids[rid] = np.full(3, np.nan, dtype=float)
			continue
		centroids[rid] = vertices[mask].mean(axis=0)
	return centroids


def _build_template_mesh_faces(surface_space: str) -> tuple[np.ndarray, int]:
	left_vertices, left_faces = load_template_surface(
		hemi="L", space=surface_space, surf_type="midthickness"
	)
	right_vertices, right_faces = load_template_surface(
		hemi="R", space=surface_space, surf_type="midthickness"
	)
	n_left = int(left_vertices.shape[0])
	right_faces = right_faces + n_left
	faces = np.concatenate([left_faces, right_faces], axis=0).astype(int)
	return faces, n_left


def _compute_centroid_distance_matrix(
	*,
	region_ids: list[int],
	scale: int,
	surface_space: str,
) -> pd.DataFrame:
	centroids = _compute_region_centroids(
		region_ids=region_ids,
		scale=scale,
		surface_space=surface_space,
	)
	distances = pd.DataFrame(index=region_ids, columns=region_ids, dtype=float)
	for rid_a in region_ids:
		cent_a = centroids.get(rid_a)
		for rid_b in region_ids:
			cent_b = centroids.get(rid_b)
			if cent_a is None or cent_b is None:
				distances.loc[rid_a, rid_b] = np.nan
				continue
			if np.any(np.isnan(cent_a)) or np.any(np.isnan(cent_b)):
				distances.loc[rid_a, rid_b] = np.nan
				continue
			distances.loc[rid_a, rid_b] = float(np.linalg.norm(cent_a - cent_b))
	return distances


def _compute_region_adjacency(
	*,
	scale: int,
	surface_space: str,
) -> dict[int, set[int]]:
	parcel_labels, _ = _build_surface_parcel_labels(scale, surface_space)
	faces, _ = _build_template_mesh_faces(surface_space)
	adjacency: dict[int, set[int]] = {}

	for tri in faces:
		labels = parcel_labels[tri]
		labels = [int(lbl) for lbl in labels if int(lbl) != 0]
		if len(labels) < 2:
			continue
		unique_labels = sorted(set(labels))
		for i, rid_a in enumerate(unique_labels):
			adjacency.setdefault(rid_a, set())
			for rid_b in unique_labels[i + 1 :]:
				adjacency.setdefault(rid_b, set())
				if rid_a != rid_b:
					adjacency[rid_a].add(rid_b)
					adjacency[rid_b].add(rid_a)

	return adjacency


def _compute_adjacency_shortest_path_matrix(
	*,
	region_ids: list[int],
	scale: int,
	surface_space: str,
) -> pd.DataFrame:
	adjacency = _compute_region_adjacency(scale=scale, surface_space=surface_space)
	all_regions = set(region_ids)

	distances = pd.DataFrame(index=region_ids, columns=region_ids, dtype=float)
	for rid in region_ids:
		if rid not in adjacency:
			distances.loc[rid] = np.nan
			continue
		dist = {rid: 0}
		queue = [rid]
		while queue:
			current = queue.pop(0)
			for neighbor in adjacency.get(current, set()):
				if neighbor in dist:
					continue
				dist[neighbor] = dist[current] + 1
				queue.append(neighbor)
		for other in region_ids:
			if other in dist:
				distances.loc[rid, other] = float(dist[other])
			elif other in all_regions:
				distances.loc[rid, other] = np.nan
	return distances


def _plot_similarity_vs_distance(
	*,
	corr_df: pd.DataFrame,
	region_ids: list[int],
	distance_df: pd.DataFrame,
	metric: str,
	distance_label: str,
	use_violin: bool,
	title: str,
	out_file: Path,
) -> None:
	import matplotlib.pyplot as plt

	distances = []
	similarities = []
	for i, rid_a in enumerate(region_ids):
		for rid_b in region_ids[i + 1 :]:
			dist_val = float(distance_df.loc[rid_a, rid_b])
			if np.isnan(dist_val) or np.isinf(dist_val):
				continue
			val = float(corr_df.loc[rid_a, rid_b])
			if np.isnan(val):
				continue
			distances.append(dist_val)
			similarities.append(val)

	fig, ax = plt.subplots(figsize=(7, 5))
	if use_violin:
		if distances:
			unique_steps = sorted({int(d) for d in distances})
			groups = [
				[s for d, s in zip(distances, similarities) if int(d) == step]
				for step in unique_steps
			]
			ax.violinplot(groups, positions=unique_steps, showmeans=True, showextrema=False)
			ax.set_xticks(unique_steps)
		ax.set_xlabel(distance_label)
	else:
		ax.scatter(distances, similarities, s=10, alpha=0.5, edgecolors="none")
		if len(distances) >= 2:
			slope, intercept = np.polyfit(distances, similarities, 1)
			x_vals = np.asarray([min(distances), max(distances)], dtype=float)
			y_vals = slope * x_vals + intercept
			ax.plot(x_vals, y_vals, color="black", linewidth=1.5)
		ax.set_xlabel(distance_label)
	ax.set_ylabel(f"{metric} similarity")
	ax.set_title(title)
	fig.tight_layout()
	fig.savefig(out_file, dpi=150)
	plt.close(fig)


def _plot_cluster_surface(
	region_ids: list[int],
	cluster_labels: np.ndarray,
	*,
	scale: int,
	surface_space: str,
	title: str,
	out_file: Path,
) -> None:
	from matplotlib import pyplot as plt
	from matplotlib.colors import ListedColormap
	from matplotlib.patches import Patch
	from nilearn import plotting

	left_mesh = load_template_surface(
		hemi="L", space=surface_space, surf_type="midthickness"
	)
	right_mesh = load_template_surface(
		hemi="R", space=surface_space, surf_type="midthickness"
	)
	parcel_labels, n_left = _build_surface_parcel_labels(scale, surface_space)

	label_map = {int(rid): int(lbl) for rid, lbl in zip(region_ids, cluster_labels)}
	texture = np.zeros(parcel_labels.shape[0], dtype=float)
	for rid, cl in label_map.items():
		texture[parcel_labels == int(rid)] = float(cl)

	n_clusters = int(np.max(cluster_labels)) + 1 if cluster_labels.size else 0
	colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
	cmap = ListedColormap(colors)

	fig = plt.figure(figsize=(14, 5))
	ax1 = fig.add_subplot(1, 2, 1, projection="3d")
	ax2 = fig.add_subplot(1, 2, 2, projection="3d")

	plotting.plot_surf_stat_map(
		left_mesh,
		texture[:n_left],
		hemi="left",
		cmap=cmap,
		colorbar=False,
		vmin=0,
		vmax=max(n_clusters - 1, 1),
		axes=ax1,
		title="Left",
	)
	plotting.plot_surf_stat_map(
		right_mesh,
		texture[n_left:],
		hemi="right",
		cmap=cmap,
		colorbar=False,
		vmin=0,
		vmax=max(n_clusters - 1, 1),
		axes=ax2,
		title="Right",
	)

	legend_handles = [
		Patch(color=colors[i], label=f"cluster {i}")
		for i in range(n_clusters)
	]
	fig.legend(
		handles=legend_handles,
		loc="lower center",
		ncol=min(6, max(n_clusters, 1)),
		bbox_to_anchor=(0.5, -0.08),
	)
	fig.suptitle(title)
	fig.savefig(out_file, dpi=150, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	records = _load_mesh_records()
	mesh_list = _mesh_records_to_list(records)

	for embedding in EMBEDDINGS:
		if embedding == "region_mean":
			labels = np.asarray(mesh_list[0]["parcel_labels"]).astype(int)
			region_order = sorted([int(r) for r in np.unique(labels) if int(r) != 0])
			region_vectors = _build_region_mean_vectors(mesh_list, region_order)
			region_vectors = _standardize_region_vectors(region_vectors)
		else:
			extractor = RegionFeatureExtractor(
				region_representation=embedding,
				pca_n_components=PCA_N_COMPONENTS,
			)
			extractor.fit(mesh_list)
			features = extractor.transform(mesh_list)
			features = StandardScaler(copy=False).fit_transform(features)

			region_order, region_slices = _region_slices(extractor)
			region_vectors = _build_region_vectors(
				features,
				region_order,
				region_slices,
			)

		corr_df = _compute_region_similarity(region_vectors, CORR_METRIC)
		corr_df_raw = corr_df.copy()
		if DISTANCE_MODE == "adjacency_shortest_path":
			distance_df = _compute_adjacency_shortest_path_matrix(
				region_ids=region_order,
				scale=SCALE,
				surface_space=SURFACE_SPACE,
			)
			distance_label = "Region distance (adjacency steps)"
			use_violin = True
		elif DISTANCE_MODE == "centroid":
			distance_df = _compute_centroid_distance_matrix(
				region_ids=region_order,
				scale=SCALE,
				surface_space=SURFACE_SPACE,
			)
			distance_label = "Centroid distance (mm)"
			use_violin = False
		else:
			raise ValueError("DISTANCE_MODE must be 'adjacency_shortest_path' or 'centroid'")
		cluster_labels = _cluster_regions(
			corr_df,
			distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
			n_clusters=N_CLUSTERS,
		)
		_warn_unbalanced_clusters(cluster_labels)
		region_names = _name_regions(region_order)
		corr_df.index = region_names
		corr_df.columns = region_names

		out_dir = OUTPUT_DIR / DATASET_SELECTION / MICROSTRUCTURE_SELECTION
		out_dir.mkdir(parents=True, exist_ok=True)

		csv_path = out_dir / f"region_corr_{embedding}.csv"
		fig_path = out_dir / f"region_corr_{embedding}.png"
		cluster_fig_path = out_dir / f"region_clusters_{embedding}.png"
		scatter_fig_path = out_dir / f"region_corr_vs_distance_{embedding}.png"
		corr_df.to_csv(csv_path)
		_plot_corr_matrix(
			corr_df,
			title=f"Region correlation | {embedding} | {DATASET_SELECTION}",
			out_file=fig_path,
		)
		_plot_similarity_vs_distance(
			corr_df=corr_df_raw,
			region_ids=region_order,
			distance_df=distance_df,
			metric=CORR_METRIC,
			distance_label=distance_label,
			use_violin=use_violin,
			title=f"Similarity vs distance | {embedding} | {DATASET_SELECTION}",
			out_file=scatter_fig_path,
		)
		_plot_cluster_surface(
			region_order,
			cluster_labels,
			scale=SCALE,
			surface_space=SURFACE_SPACE,
			title=(
				f"Region clusters | {embedding} | {DATASET_SELECTION} "
				f"(tau={CLUSTER_DISTANCE_THRESHOLD})"
			),
			out_file=cluster_fig_path,
		)


if __name__ == "__main__":
	main()

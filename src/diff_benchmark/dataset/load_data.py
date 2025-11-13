from pathlib import Path

import h5py
import numpy as np


def load_embeddings_and_power_from_h5(h5_path):
    """
    Load embeddings, power, and metadata from an HDF5 file.

    Returns:
        embeddings (dict): Dictionary mapping b-value (int) to embedding array
        power (np.ndarray): Array of shape [num_nodes, 3]
        metadata (dict): Dictionary of metadata attributes
    """
    embeddings = {}
    metadata = {}

    with h5py.File(h5_path, "r") as f:
        # Load embeddings
        emb_grp = f["embeddings"]
        for key in emb_grp:
            bval = int(key[1:])  # from "b1000" to 1000
            embeddings[bval] = emb_grp[key][:]

        # Load power
        power = f["power"][:]

        # Load metadata
        meta_grp = f["metadata"]
        metadata = {k: meta_grp.attrs[k] for k in meta_grp.attrs}

    return embeddings, power, metadata

def _load_global_metadata(f):
    """Extract global metadata and geometry arrays."""
    metadata = {}
    meta_grp = f["meta"]
    for key in meta_grp.attrs:
        metadata[key] = meta_grp.attrs[key]

    metadata["bvals"] = f["bvals"][:]
    metadata["bvecs"] = f["bvecs"][:]
    metadata["sphere_vertices"] = f["sphere_vertices"][:]
    metadata["sphere_faces"] = f["sphere_faces"][:]
    metadata["sphere_edges"] = f["sphere_edges"][:]
    return metadata


def _load_bval_group(f, bval_grp_name):
    """Load attenuation data for one b-value group."""
    bval = int(bval_grp_name.split("_")[1])
    vertex_group = f[bval_grp_name]

    num_vertices = len(vertex_group)
    sample_vertex = next(iter(vertex_group.values()))
    num_directions = sample_vertex["attenuation"].shape[0]

    attenuation_array = np.zeros((num_vertices, num_directions))
    surface_vertices = np.zeros(num_vertices, dtype=np.int32)
    labels = np.zeros(num_vertices, dtype=np.int32)
    neighbors = []

    for i, vgrp in enumerate(vertex_group.values()):
        attenuation_array[i] = vgrp["attenuation"][:]
        surface_vertices[i] = vgrp["surface_vertex"][()]
        labels[i] = vgrp["deen_insula_label"][()]
        neighbors.append(vgrp["mesh_neighbors"][:])

    return bval, attenuation_array, surface_vertices, labels, neighbors


def load_attenuation_from_h5(h5_path: Path):
    """Load attenuation data and associated metadata from a custom HDF5 file."""
    breakpoint()
    attenuation_data = {}
    metadata = {}

    with h5py.File(h5_path, "r") as f:
        metadata = _load_global_metadata(f)

        for bval_grp in f:
            if not bval_grp.startswith("bval_"):
                continue

            bval, att_arr, surf_v, lbls, neigh = _load_bval_group(f, bval_grp)

            attenuation_data[bval] = att_arr
            metadata[f"surface_vertices_b{bval}"] = surf_v
            metadata[f"labels_b{bval}"] = lbls
            metadata[f"neighbors_b{bval}"] = neigh

    return attenuation_data, metadata

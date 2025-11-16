import json
from pathlib import Path
from xml import etree

import h5py
import nibabel as nib
import nilearn as ni
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst import dti
from dipy.reconst.mapmri import MapmriModel
from nilearn import image as nimage
from nilearn import maskers
from scipy import ndimage
from scipy.linalg import LinAlgError
from scipy.spatial import cKDTree
from templateflow import api as tflow
from tqdm import tqdm


def extract_selected_labels(nifti_path):
    """Extract selected labels from a NIfTI file's header extensions."""
    header = nib.load(nifti_path).header
    labels = {
        n.text.lower(): int(n.get("Key"))
        for n in etree.ElementTree.fromstring(header.extensions[0].text).findall(
            ".//Label"
        )
    }
    return {k: v for k, v in labels.items() if k.startswith("ctx") or "ventricle" in k}


def create_masks(parcellation_img, labels):
    """Create context and ventricle masks from parcellation image."""
    ctx_mask = nimage.math_img(
        " + ".join(f"(x == {v})" for k, v in labels.items() if "ctx" in k),
        x=parcellation_img,
    )
    vent_mask_raw = nimage.math_img(
        " + ".join(f"(x == {v})" for k, v in labels.items() if "vent" in k),
        x=parcellation_img,
    )
    vent_mask = nimage.new_img_like(
        parcellation_img, ndimage.binary_erosion(nimage.get_data(vent_mask_raw))
    )
    return ctx_mask, vent_mask


def compute_rtop(
    dwi_nib, mask_img, normalization_mask_img, bvals, bvecs, big_delta, small_delta
):
    """Compute RTOP (Radial Tensor Orientation Profile) from DWI data."""
    masker = maskers.NiftiMasker(mask_img)
    masker.fit()
    dwi_data = masker.transform(dwi_nib)

    gtab = gradient_table(bvals, bvecs, big_delta=big_delta, small_delta=small_delta)
    map_model = MapmriModel(
        gtab,
        radial_order=6,
        laplacian_regularization=True,
        laplacian_weighting=0.2,
        positivity_constraint=False,
    )
    rtop = map_model.fit(dwi_data.T).rtop()
    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit()
        dwi_ventricles = norm_masker.transform(dwi_nib)
        rtop_ventricles = map_model.fit(dwi_ventricles.T).rtop()

        nrtop = rtop / rtop_ventricles.mean()
        nrtop_img = masker.inverse_transform(nrtop.T)
        return nrtop_img

    return masker.inverse_transform(rtop.T)


def compute_md(
    dwi_nib, mask_img, normalization_mask_img, bvals, bvecs, big_delta, small_delta
):
    """Compute Mean Diffusivity (MD) from DWI data."""
    masker = maskers.NiftiMasker(mask_img)
    masker.fit()
    dwi_data = masker.transform(dwi_nib)

    gtab = gradient_table(bvals, bvecs, big_delta=big_delta, small_delta=small_delta)

    dti_model = dti.TensorModel(gtab)

    md = dti_model.fit(dwi_data.T).md

    if normalization_mask_img is not None:
        norm_masker = maskers.NiftiMasker(normalization_mask_img)
        norm_masker.fit()
        dwi_ventricles = norm_masker.transform(dwi_nib)
        md_ventricles = dti_model.fit(dwi_ventricles.T).md

        nmd = md / md_ventricles.mean()
        nmd_img = masker.inverse_transform(nmd.T)
        return nmd_img
    print("Be careful, this is not normalized MD!")
    return masker.inverse_transform(md.T)


def project_to_surface(
    micr_img, ctx_mask, surfaces, output_dir, subject_id, micr_metric
):
    """
    Project image onto surface meshes and save as GIFTI files.
    Image should be in NIfTI format and contain RTOP/MD/microstructure values.
    """
    for h in ("L", "R"):
        insula_surf = ni.surface.vol_to_surf(
            micr_img,
            surfaces[f"{h}.pial"],
            mask_img=ctx_mask,
            inner_mesh=surfaces[f"{h}.white"],
            depth=[0.1, 0.5, 0.9],
        )
        nib.gifti.GiftiImage()
        img = nib.gifti.gifti.GiftiImage()
        img.add_gifti_data_array(
            nib.gifti.gifti.GiftiDataArray(
                insula_surf.astype(np.float32),
                intent="NIFTI_INTENT_DIMLESS",
            )
        )
        nib.save(
            img,
            output_dir / f"sub-{subject_id}_hemi-{h}_param-{micr_metric}.scalar.gii",
        )


def resample_schaefer_onto_fs_lr(scale=1000):
    """Resample Schaefer 2018 parcellation onto fsLR space."""
    fsaverage_left_schaefer_fn = tflow.get(
        "fsaverage",
        hemi="L",
        density="164k",
        atlas="Schaefer2018",
        segmentation="17n",
        scale=str(scale),
        extension="label.gii",
    )
    fsaverage_left_schaefer = nib.load(fsaverage_left_schaefer_fn)

    fsaverage_right_schaefer_fn = tflow.get(
        "fsaverage",
        hemi="R",
        density="164k",
        atlas="Schaefer2018",
        segmentation="17n",
        scale=str(scale),
        extension="label.gii",
    )
    fsaverage_right_schaefer = nib.load(fsaverage_right_schaefer_fn)

    labels = pd.read_csv(
        tflow.get(
            "fsaverage",
            atlas="Schaefer2018",
            segmentation="17n",
            scale=str(scale),
            extension="tsv",
        ),
        sep="\t",
    )

    labels_left = labels[labels["hemi"] == "L"]
    labels_right = labels[labels["hemi"] == "R"]

    fslr_2_fsaverage_left_sphere_fn = tflow.get(
        "fsLR", hemi="L", density="32k", space="fsaverage"
    )
    fsaverage_sphere_left_fn = tflow.get(
        "fsaverage", hemi="L", density="164k", suffix="sphere", desc=None
    )
    fslr_2_fsaverage_left_sphere = nib.load(fslr_2_fsaverage_left_sphere_fn)
    fsaverage_sphere_left = nib.load(fsaverage_sphere_left_fn)
    kdtree_fsaverage_left = cKDTree(fsaverage_sphere_left.darrays[0].data)
    kdtree_fslr_left = cKDTree(fslr_2_fsaverage_left_sphere.darrays[0].data)
    fslr_to_fsaverage_left_sphere = kdtree_fslr_left.query(
        fsaverage_sphere_left.darrays[0].data, k=1
    )
    fsaverage_to_fslr_left_sphere = kdtree_fsaverage_left.query(
        fslr_2_fsaverage_left_sphere.darrays[0].data, k=1
    )

    fslr_2_fsaverage_right_sphere_fn = tflow.get(
        "fsLR", hemi="R", density="32k", space="fsaverage"
    )
    fsaverage_sphere_right_fn = tflow.get(
        "fsaverage", hemi="R", density="164k", suffix="sphere", desc=None
    )
    fslr_2_fsaverage_right_sphere = nib.load(fslr_2_fsaverage_right_sphere_fn)
    fsaverage_sphere_right = nib.load(fsaverage_sphere_right_fn)
    kdtree_fsaverage_right = cKDTree(fsaverage_sphere_right.darrays[0].data)
    kdtree_fslr_right = cKDTree(fslr_2_fsaverage_right_sphere.darrays[0].data)
    fslr_to_fsaverage_right_sphere = kdtree_fslr_right.query(
        fsaverage_sphere_right.darrays[0].data, k=1
    )
    fsaverage_to_fslr_right_sphere = kdtree_fsaverage_right.query(
        fslr_2_fsaverage_right_sphere.darrays[0].data, k=1
    )

    fslr_left_schaefer = np.zeros(len(fsaverage_to_fslr_left_sphere[1]))
    fslr_left_schaefer[fslr_to_fsaverage_left_sphere[1]] = (
        fsaverage_left_schaefer.darrays[0].data
    )

    fslr_right_schaefer = np.zeros(len(fsaverage_to_fslr_right_sphere[1]))
    fslr_right_schaefer[fslr_to_fsaverage_right_sphere[1]] = (
        fsaverage_right_schaefer.darrays[0].data
    )

    fslr_left_sulc = np.zeros(len(fsaverage_to_fslr_left_sphere[1]))
    fslr_left_sulc[fslr_to_fsaverage_left_sphere[1]] = (
        nib.load(
            tflow.get("fsaverage", hemi="L", density="164k", suffix="sulc", desc=None)
        )
        .darrays[0]
        .data
    )

    fslr_right_sulc = np.zeros(len(fsaverage_to_fslr_right_sphere[1]))
    fslr_right_sulc[fslr_to_fsaverage_right_sphere[1]] = (
        nib.load(
            tflow.get("fsaverage", hemi="R", density="164k", suffix="sulc", desc=None)
        )
        .darrays[0]
        .data
    )

    return {
        "left.data": fslr_left_schaefer,
        "left.labels": labels_left,
        "left.sulc": fslr_left_sulc,
        "right.data": fslr_right_schaefer,
        "right.labels": labels_right,
        "right.sulc": fslr_right_sulc,
    }


def resample_brainnetome_onto_fs_lr():
    """Resample Brainnetome parcellation onto fsLR space."""
    fsaverage_left_brainnetome_fn = tflow.get(
        "fsaverage",
        atlas="brainnetome",
        hemi="L",
        density="164k",
        extension="label.gii",
    )

    fsaverage_left_brainnetome = nib.load(fsaverage_left_brainnetome_fn)

    fsaverage_right_brainnetome_fn = tflow.get(
        "fsaverage",
        atlas="brainnetome",
        hemi="R",
        density="164k",
        extension="label.gii",
    )
    fsaverage_right_brainnetome = nib.load(fsaverage_right_brainnetome_fn)

    labels = pd.read_csv(
        tflow.get("fsaverage", atlas="brainnetome", extension="tsv"), sep="\t"
    )

    labels_left = labels[labels["hemi"] == "L"]
    labels_left["name"] = labels_left["name"] + "_LH"
    labels_right = labels[labels["hemi"] == "R"]
    labels_right["name"] = labels_right["name"] + "_RH"

    fslr_2_fsaverage_left_sphere_fn = tflow.get(
        "fsLR", hemi="L", density="32k", space="fsaverage"
    )
    fsaverage_sphere_left_fn = tflow.get(
        "fsaverage", hemi="L", density="164k", suffix="sphere", desc=None
    )
    fslr_2_fsaverage_left_sphere = nib.load(fslr_2_fsaverage_left_sphere_fn)
    fsaverage_sphere_left = nib.load(fsaverage_sphere_left_fn)
    kdtree_fsaverage_left = cKDTree(fsaverage_sphere_left.darrays[0].data)
    kdtree_fslr_left = cKDTree(fslr_2_fsaverage_left_sphere.darrays[0].data)
    fslr_to_fsaverage_left_sphere = kdtree_fslr_left.query(
        fsaverage_sphere_left.darrays[0].data, k=1
    )
    fsaverage_to_fslr_left_sphere = kdtree_fsaverage_left.query(
        fslr_2_fsaverage_left_sphere.darrays[0].data, k=1
    )

    fslr_2_fsaverage_right_sphere_fn = tflow.get(
        "fsLR", hemi="R", density="32k", space="fsaverage"
    )
    fsaverage_sphere_right_fn = tflow.get(
        "fsaverage", hemi="R", density="164k", suffix="sphere", desc=None
    )
    fslr_2_fsaverage_right_sphere = nib.load(fslr_2_fsaverage_right_sphere_fn)
    fsaverage_sphere_right = nib.load(fsaverage_sphere_right_fn)
    kdtree_fsaverage_right = cKDTree(fsaverage_sphere_right.darrays[0].data)
    kdtree_fslr_right = cKDTree(fslr_2_fsaverage_right_sphere.darrays[0].data)
    fslr_to_fsaverage_right_sphere = kdtree_fslr_right.query(
        fsaverage_sphere_right.darrays[0].data, k=1
    )
    fsaverage_to_fslr_right_sphere = kdtree_fsaverage_right.query(
        fslr_2_fsaverage_right_sphere.darrays[0].data, k=1
    )

    fslr_left_brainnetome = np.zeros(len(fsaverage_to_fslr_left_sphere[1]))
    fslr_left_brainnetome[fslr_to_fsaverage_left_sphere[1]] = (
        fsaverage_left_brainnetome.darrays[0].data
    )

    fslr_right_brainnetome = np.zeros(len(fsaverage_to_fslr_right_sphere[1]))
    fslr_right_brainnetome[fslr_to_fsaverage_right_sphere[1]] = (
        fsaverage_right_brainnetome.darrays[0].data
    )

    fslr_left_sulc = np.zeros(len(fsaverage_to_fslr_left_sphere[1]))
    fslr_left_sulc[fslr_to_fsaverage_left_sphere[1]] = (
        nib.load(
            tflow.get("fsaverage", hemi="L", density="164k", suffix="sulc", desc=None)
        )
        .darrays[0]
        .data
    )

    fslr_right_sulc = np.zeros(len(fsaverage_to_fslr_right_sphere[1]))
    fslr_right_sulc[fslr_to_fsaverage_right_sphere[1]] = (
        nib.load(
            tflow.get("fsaverage", hemi="R", density="164k", suffix="sulc", desc=None)
        )
        .darrays[0]
        .data
    )

    return {
        "left.data": fslr_left_brainnetome,
        "left.labels": labels_left,
        "left.sulc": fslr_left_sulc,
        "right.data": fslr_right_brainnetome,
        "right.labels": labels_right,
        "right.sulc": fslr_right_sulc,
    }


def load_rtop_data(config):
    """
    Load RTOP scalar data from left and right .scalar.gii files.
    Assumes filenames follow format *_rtop_cortex.L/R*.scalar.gii
    """
    subject_id = "100206"  # Test subject
    subject_dir = Path(config["data_paths"]["hcp_results"]) / subject_id / "processed"
    rtop_left = (
        nib.load(subject_dir / f"{config['metric_to_compute']}.L.scalar.gii")
        .darrays[0]
        .data
    )
    rtop_right = (
        nib.load(subject_dir / f"{config['metric_to_compute']}.R.scalar.gii")
        .darrays[0]
        .data
    )
    # Clean up (clip and replace NaNs)
    rtop_left = np.nan_to_num(rtop_left).clip(0, 7)
    rtop_right = np.nan_to_num(rtop_right).clip(0, 7)
    # breakpoint()
    return rtop_left, rtop_right


def average_per_parcel(hem_left, hem_right, schaefer_resampled):
    """
    Average RTOP values across parcels in both hemispheres.
    hem_left: RTOP/MD/microstructure values for left hemisphere
    hem_right: RTOP/MD/microstructure values for right hemisphere
    """
    parcellation_left = schaefer_resampled["left.data"]
    parcellation_right = schaefer_resampled["right.data"]
    # breakpoint()
    parcels_left = np.unique(parcellation_left)
    parcels_right = np.unique(parcellation_right)

    # Initialize array to store mean values per parcel
    rtop_avg = np.zeros(len(parcels_left) + len(parcels_right))

    for i, parcel in enumerate(sorted(parcels_left)):
        mask = parcellation_left == parcel
        rtop_avg[i] = hem_left[mask].mean()
    for i, parcel in enumerate(sorted(parcels_right), start=len(parcels_left)):
        mask = parcellation_right == parcel
        rtop_avg[i] = hem_right[mask].mean()
    return rtop_avg


def extract_region_data(
    hem_left, hem_right, schaefer_resampled, target_substring=None, average=False
):
    """
    Average microstructure values across selected parcels in both hemispheres.

    Args:
        hem_left (ndarray): Microstructure values for left hemisphere.
        hem_right (ndarray): Microstructure values for right hemisphere.
        schaefer_resampled (dict): Contains 'left.data' and 'right.data' arrays.
        region_ids (list[int] or None): Optional list of parcel IDs to include.
            If None, all parcels will be included.
        average (bool): If True, return the overall average across selected regions.
    Returns:
        np.ndarray: Mean values per selected region (in order of selection if provided).
    """
    parc_left = schaefer_resampled["left.data"]
    parc_right = schaefer_resampled["right.data"]
    labels_left = schaefer_resampled["left.labels"].copy()
    labels_right = schaefer_resampled["right.labels"].copy()

    unique_left = np.unique(parc_left)
    unique_right = np.unique(parc_right)

    labels_left["array_index"] = unique_left[: len(labels_left)]
    labels_right["array_index"] = unique_right[: len(labels_right)]
    labels_left["hemi"] = "L"
    labels_right["hemi"] = "R"
    all_labels = pd.concat([labels_left, labels_right], ignore_index=True)

    if target_substring:
        matched_labels = all_labels[
            all_labels["name"].str.contains(target_substring, case=False, na=False)
        ]
    else:
        matched_labels = all_labels

    region_data = {}
    region_values = []

    # --- Iterate through matched regions (each has unique hemi + name)
    for _, row in matched_labels.iterrows():
        hemi = row["hemi"]
        region_id = row["array_index"]
        region_name = row["name"]
        if hemi == "L":
            mask = parc_left == region_id
            vals = hem_left[mask]
            region_values.append(vals)
        elif hemi == "R":
            mask = parc_right == region_id
            vals = hem_right[mask]
            region_values.append(vals)
        else:
            continue  # unexpected hemi label

        if vals.size == 0:
            continue  # skip empty

        region_data[region_name] = np.nanmean(vals) if average else vals

    region_values = np.concatenate([np.atleast_1d(v) for v in region_values])
    # return region_data # returns dict and csv is a column per region with the corresponding arrays
    return region_values


def normalize(data):
    """Normalizes the input data to have zero mean and unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


def compute_data(
    data, bvals_to_compute, sphere, model, gtab0, graph_ins, normalize_input
):
    """
    Computes the attenuation data for each b-value and vertex.

    Args:
        data (dict): Input data containing dwi_signal, vertex_indices, labels, etc.
        bvals_to_compute (list): List of b-values to compute.
        sphere (object): Sphere object with vertices and edges.
        model (object): Model used for fitting the signal.
        gtab0 (object): Gradient table for b0.
        graph_ins (networkx.Graph): Graph of the surface mesh.
        normalize_input (bool): Whether to normalize the input signal.

    Returns:
        dict: Computed results for all b-values.
    """
    all_results = {}
    for bval in bvals_to_compute:
        gtab_sphere = gradient_table(
            bvals=np.repeat(bval, len(sphere.vertices)), bvecs=sphere.vertices
        )

        subject_spheres = []

        print(f"{bval}")

        for vertex, attrs in tqdm(graph_ins.nodes(data=True), desc=f"B={bval}"):
            signal = attrs["signal"]
            if normalize_input:
                signal = signal / np.linalg.norm(signal)

            fit_success = True

            try:
                fit = model.fit(signal)
            except LinAlgError:
                fit_success = False
                print(
                    f"Vertex {vertex} - SVD did not converge. Using neighbor average."
                )

                neighbors = list(graph_ins.neighbors(vertex))
                if not neighbors:
                    print(f"Vertex {vertex} has no neighbors. Skipping.")

                neighbor_signals = np.array(
                    [graph_ins.nodes[n]["signal"] for n in neighbors]
                )
                avg_signal = np.mean(neighbor_signals, axis=0)

                if normalize_input:
                    avg_signal = normalize(avg_signal)

                try:
                    fit = model.fit(avg_signal)
                    fit_success = True
                except LinAlgError:
                    print(
                        f"Vertex {vertex} - Averaged neighbor signal also failed. Skipping."
                    )

            if not fit_success:
                attenuation = np.full(len(sphere.vertices), np.nan, dtype=np.float32)
                b0_val = np.nan
                fit_status = "failed"
            else:
                b0_val = fit.predict(gtab0)
                attenuation = fit.predict(gtab_sphere) / b0_val
                fit_status = "success"

            vertex_data = {
                "vertex": vertex,
                "attenuation": attenuation.astype(np.float32),
                "neighbors": list(graph_ins.neighbors(vertex)),
                "label": attrs["label"],
                "fit_status": fit_status,
            }
            subject_spheres.append(vertex_data)

        all_results[str(bval)] = subject_spheres

    return all_results


def load_vertexwise_attenuations(file_path):
    """
    Load attenuations from an HDF5 file into a NumPy array of shape:
        [num_vertices, num_bvals, attenuation_length]
    """
    with h5py.File(file_path, "r") as f:
        # Sort bvals and vertices to ensure consistent order
        # bvals = sorted(f.keys(), key=lambda x: float(x))
        # vertices = sorted(f[bvals[0]].keys(), key=lambda x: int(x))
        bvals = sorted(f.keys(), key=float)
        vertices = sorted(f[bvals[0]].keys(), key=int)

        num_bvals = len(bvals)
        num_vertices = len(vertices)

        # Determine attenuation length from the first dataset
        example_att = f[bvals[0]][vertices[0]]["attenuation"][()]
        att_length = example_att.shape[0] if example_att.ndim > 0 else 1

        # Initialize array [num_vertices, num_bvals, attenuation_length]
        data = np.zeros((num_vertices, num_bvals, att_length), dtype=example_att.dtype)

        # Fill array
        for j, vertex in enumerate(vertices):
            for i, bval in enumerate(bvals):
                vgrp = f[bval][vertex]
                data[j, i, :] = vgrp["attenuation"][()]

        # Optional metadata
        metadata = {
            "subject_id": f.attrs["subject_id"],
            "bvals_to_compute": json.loads(f.attrs["bvals_to_compute"]),
            "sphere_vertices": f.attrs["sphere_vertices"],
            "bvals": bvals,
            "vertices": vertices,
        }

    return data, metadata


def split_data(data, num_splits):
    """Splits data into specified number of splits."""
    split_size = data.shape[0] // num_splits
    return [
        data[i * split_size : (i + 1) * split_size] for i in range(num_splits - 1)
    ] + [data[(num_splits - 1) * split_size :]]

# rtop_pipeline.py
import json
from pathlib import Path

import h5py
import networkx as nx
import nibabel as nib
import nilearn as ni
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.reconst.mapmri import MapmriModel
from nilearn import image as nimage
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

from diff_benchmark.preprocessing.lcot.sliced_lcot import EmbeddingCircleWeights
from diff_benchmark.preprocessing.wrapper_brain_base import DataPreparationBrain
from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
    average_per_parcel,
    extract_region_data,
    compute_data,
    compute_md,
    compute_rtop,
    create_masks,
    extract_selected_labels,
    project_to_surface,
    resample_schaefer_onto_fs_lr,
    load_vertexwise_attenuations,
    split_data,
)


class DefaultHcpPipeline(DataPreparationBrain):
    """
    DefaultHcpPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the Human Connectome Project (HCP) pipeline.
    Attributes:
        hcp_dir (Path): The directory containing HCP data.
        results_root (Path): The root directory for storing results.
        metric (str): The metric to compute (e.g., 'rtop', 'md').
        schaefer_resampled: Resampled Schaefer atlas onto fs_LR.
        big_delta (float): The big delta value for diffusion metrics.
        small_delta (float): The small delta value for diffusion metrics.
    Methods:
        verify_subject_files(subject_id: str, metric: str) -> bool:
            Checks if both hemispheres' .scalar.gii files exist for the given subject and metric.
        compute_microstructure(subject_id: str):
            Computes microstructure metrics for the given subject and saves the results.
        run_analysis():
            Runs the analysis on the scalar files and computes average data per parcel.
        extract_features():
            Placeholder method for extracting features (to be implemented).
    """

    def __init__(self, config):
        super().__init__(config)
        self.hcp_dir = Path(config["base_path"])
        self.results_root = Path(config["results_path"]) / "default"
        self.metric = config["metric_to_compute"]
        # breakpoint()
        self.scale = config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config["big_delta"]
        self.small_delta = config["small_delta"]

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if both hemispheres' .scalar.gii files exist for the given subject and metric.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        left_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-L_param-{metric}.scalar.gii"
        )
        right_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-R_param-{metric}.scalar.gii"
        )

        return left_file.exists() and right_file.exists()

    def compute_microstructure(self, subject_id: str):
        """Compute microstructure metrics for the given subject and save the results."""
        try:
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)

            subject_dir = self.hcp_dir / subject_id

            diffusion_dir = subject_dir / "T1w" / "Diffusion"
            dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
            bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
            bvals = np.loadtxt(bvals)
            bvecs = np.loadtxt(bvecs).T
            nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz"

            aparc_aseg = subject_dir / "T1w" / "aparc+aseg.nii.gz"

            labels = extract_selected_labels(aparc_aseg)
            aparc_resampled = nimage.resample_to_img(
                aparc_aseg,
                nodif_mask,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            ctx_mask, vent_mask = create_masks(aparc_resampled, labels)

            surfaces = {
                f"{h}.{s}": subject_dir
                / "T1w"
                / "fsaverage_LR32k"
                / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
                for s in ("white", "pial")
                for h in ("L", "R")
            }

            if self.metric == "rtop":
                rtop_img = compute_rtop(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    rtop_img,
                    derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
                )
                project_to_surface(
                    rtop_img,
                    ctx_mask,
                    surfaces,
                    derivatives_dir,
                    subject_id,
                    self.metric,
                )
            elif self.metric == "md":
                md_img = compute_md(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
                )
                project_to_surface(
                    md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric
                )

        except Exception as e:
            print(f"[{subject_id}] Error during microstructure: {e}")

    # def run_analysis(self):  # Doing a test
    def run_analysis_good(self):
        scalar_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
            )
        )
        for left_file in tqdm(scalar_files, desc="Running analysis"):
            try:
                subject_id = left_file.stem.split("_")[0].replace("sub-", "")
                right_file = left_file.with_name(
                    left_file.name.replace("hemi-L", "hemi-R")
                )

                left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
                    0, 7
                )
                right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
                    0, 7
                )

                avg_data = average_per_parcel(
                    left_data, right_data, self.schaefer_resampled
                )
                self.results[subject_id] = avg_data
            except Exception as e:
                print(f"[{subject_id}] Error during analysis: {e}")
    
    # def run_analysis_region(self):
    def run_analysis(self):
        scalar_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
            )
        )
        for left_file in tqdm(scalar_files, desc="Running analysis"):
            try:
                subject_id = left_file.stem.split("_")[0].replace("sub-", "")
                right_file = left_file.with_name(
                    left_file.name.replace("hemi-L", "hemi-R")
                )

                left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
                    0, 7
                )
                right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
                    0, 7
                )
                
                # target = "VisCent_Striate"
                # target = self.config["region_name"]
                target = self.config["models"][0]["params"]["region_name"]
                # target = None
                avg_data = extract_region_data(
                    left_data, right_data, self.schaefer_resampled, target_substring=target, average=False
                )
                self.results[subject_id] = avg_data
            except Exception as e:
                print(f"[{subject_id}] Error during analysis: {e}")

    def extract_features(self):
        pass


class ImageHcpPipeline(DataPreparationBrain):
    """
    ImageHcpPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the Human Connectome Project (HCP) pipeline.
    Attributes:
        hcp_dir (Path): The directory containing HCP data.
        results_root (Path): The root directory for storing results.
        metric (str): The metric to compute (e.g., 'rtop', 'md').
        schaefer_resampled: Resampled Schaefer atlas onto fs_LR.
        big_delta (float): The big delta value for diffusion metrics.
        small_delta (float): The small delta value for diffusion metrics.
    Methods:
        verify_subject_files(subject_id: str, metric: str) -> bool:
            Checks if both hemispheres' .scalar.gii files exist for the given subject and metric.
        compute_microstructure(subject_id: str):
            Computes microstructure metrics for the given subject and saves the results.
        run_analysis():
            Runs the analysis on the scalar files and computes average data per parcel.
        extract_features():
            Placeholder method for extracting features (to be implemented).
    """

    def __init__(self, config):
        super().__init__(config)
        self.hcp_dir = Path(config["base_path"])
        self.results_root = Path(config["results_path"]) / "default"
        self.metric = config["metric_to_compute"]
        self.scale = config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config["big_delta"]
        self.small_delta = config["small_delta"]

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if whole brain .nii.gii files exist for the given subject and metric.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        file = derivatives_dir / f"sub-{subject_id}_param-{metric}_dwimap.nii.gz"

        return file.exists()

    def compute_microstructure(self, subject_id: str):
        """Compute microstructure metrics for the given subject and save the results."""
        try:
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)

            subject_dir = self.hcp_dir / subject_id

            diffusion_dir = subject_dir / "T1w" / "Diffusion"
            dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
            bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
            bvals = np.loadtxt(bvals)
            bvecs = np.loadtxt(bvecs).T
            nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz"

            aparc_aseg = subject_dir / "T1w" / "aparc+aseg.nii.gz"

            labels = extract_selected_labels(aparc_aseg)
            aparc_resampled = nimage.resample_to_img(
                aparc_aseg,
                nodif_mask,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            ctx_mask, vent_mask = create_masks(aparc_resampled, labels)

            if self.metric == "rtop":
                rtop_img = compute_rtop(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    rtop_img,
                    derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
                )
            elif self.metric == "md":
                md_img = compute_md(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
                )

        except Exception as e:
            print(f"[{subject_id}] Error during microstructure: {e}")

    def run_analysis(self):
        img_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_param-{self.metric}_dwimap.nii.gz"
            )
        )
        for file in tqdm(img_files, desc="Running analysis"):
            try:
                subject_id = file.stem.split("_")[0].replace("sub-", "")
                self.results[subject_id] = file
            except Exception as e:
                print(f"[{subject_id}] Error during analysis: {e}")
                
    def run_analysis_region(self):
        img_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_param-{self.metric}_dwimap.nii.gz"
            )
        )
        
        results = {}
        
        for file in tqdm(img_files, desc="Running analysis"):
            try:
                subject_id = file.stem.split("_")[0].replace("sub-", "")
                self.results[subject_id] = file
            except Exception as e:
                print(f"[{subject_id}] Error during analysis: {e}")
                
    def extract_features(self):
        pass


class LcotEmbedHcpPipeline(DataPreparationBrain):
    """
    LcotEmbedHcpPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data specifically for LCOT embedding analysis.
    Attributes:
        hcp_dir (Path): The directory containing HCP data.
        results_root (Path): The root directory for storing results.
        metric (str): The metric to compute during analysis.
        schaefer_resampled: Resampled Schaefer atlas onto the specified space.
        big_delta (float): Parameter for big delta in analysis.
        small_delta (float): Parameter for small delta in analysis.
    Methods:
        verify_subject_files(subject_id: str, metric: str) -> bool:
            Verifies the existence of necessary files for a given subject and metric.
        extract_raw_data(subject_id: str):
            Extracts raw diffusion-weighted imaging (DWI) data for a specified subject,
            processes it, and saves it in HDF5 and GIFTI formats.
        load_subject_data(h5_path):
            Loads subject data from an HDF5 file and constructs a graph representation
            of the cortical mesh.
        compute_microstructure(subject_id: str):
            Computes microstructural features from the DWI data of a specified subject
            by merging data from both hemispheres and fitting a MAP-MRI model.
        run_analysis():
            Placeholder method for running the analysis pipeline.
        extract_features():
            Placeholder method for extracting features from the processed data.
    """

    def __init__(self, config):
        super().__init__(config)
        self.hcp_dir = Path(config["base_path"])
        self.results_root = Path(config["results_path"]) / "lcotembed"
        self.metric = config["metric_to_compute"]
        self.scale = config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config["big_delta"]
        self.small_delta = config["small_delta"]
        self.target_substring = "VisCent_ExStr" #config.get("region_name", "VisCent_Striate")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # "cpu" #
        self.dtype = torch.float32

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        target_substring = self.target_substring
        file = derivatives_dir / f"sub-{subject_id}_desc-{target_substring}_spheres.h5"

        return file.exists()
    
    def verify_required_files(self, subject_id: str) -> bool:
        subject_dir = self.hcp_dir / subject_id
        diffusion_dir = subject_dir / "T1w" / "Diffusion"

        required_files = {
            "DWI data": diffusion_dir / "data.nii.gz",
            "bvals": diffusion_dir / "bvals",
            "bvecs": diffusion_dir / "bvecs",
            "nodif mask": diffusion_dir / "nodif_brain_mask.nii.gz",
            "aparc+aseg": subject_dir / "T1w" / "aparc+aseg.nii.gz",
        }

        # Add expected surface files for both hemispheres
        for h in ("L", "R"):
            for s in ("white", "pial"):
                surf_file = (
                    subject_dir
                    / "T1w"
                    / "fsaverage_LR32k"
                    / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
                )
                required_files[f"{h}-{s} surface"] = surf_file

        # Check all files and collect missing ones
        # missing = [name for name, path in required_files.items() if not path.exists()]

        # if missing:
        #     print(
        #         f"[WARNING] Missing files for subject {subject_id}: "
        #         + ", ".join(missing)
        #     )
        #     return False

        # print(f"[INFO] All required files found for subject {subject_id}.")
        # return True
        # Check all files and collect missing or empty ones
        missing_or_empty = []
        for name, path in required_files.items():
            if not path.exists():
                missing_or_empty.append(f"{name} (missing)")
            elif path.is_file() and path.stat().st_size == 0:
                missing_or_empty.append(f"{name} (empty)")

        if missing_or_empty:
            print(
                f"[WARNING] Missing or empty files for subject {subject_id}: "
                + ", ".join(missing_or_empty)
            )
            return False

        print(f"[INFO] All required files found and non-empty for subject {subject_id}.")
        return True

    def extract_raw_data(self, subject_id: str):
        """
        Extract raw data for LCOT embedding.
        This method should be implemented to handle the specific extraction logic.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        derivatives_dir.mkdir(parents=True, exist_ok=True)

        subject_dir = self.hcp_dir / subject_id

        diffusion_dir = subject_dir / "T1w" / "Diffusion"
        dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
        bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
        bvals = np.loadtxt(bvals)
        bvecs = np.loadtxt(bvecs).T
        nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz"

        aparc_aseg = subject_dir / "T1w" / "aparc+aseg.nii.gz"

        labels = extract_selected_labels(aparc_aseg)
        aparc_resampled = nimage.resample_to_img(
            aparc_aseg,
            nodif_mask,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

        ctx_mask, _ = create_masks(aparc_resampled, labels)

        surfaces = {
            f"{h}.{s}": subject_dir
            / "T1w"
            / "fsaverage_LR32k"
            / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
            for s in ("white", "pial")
            for h in ("L", "R")
        }

        for h in ("L", "R"):
            schaefer_hemi_data = (
                self.schaefer_resampled["left.data"]
                if h == "L"
                else self.schaefer_resampled["right.data"]
            )
            # Full 4D projection: output shape (n_vertices, n_directions)
            surf_data = ni.surface.vol_to_surf(
                dwi_nib,
                surfaces[f"{h}.pial"],
                mask_img=ctx_mask,
                inner_mesh=surfaces[f"{h}.white"],
                interpolation="linear",
            )

            pial_gii = nib.load(str(surfaces[f"{h}.pial"]))
            mesh_coords = pial_gii.darrays[0].data  # shape: (n_vertices, 3)
            mesh_faces = pial_gii.darrays[1].data  # shape: (n_faces, 3)

            # Get vertex indices for mask
            mask_surf = ni.surface.vol_to_surf(
                ctx_mask, surfaces[f"{h}.pial"], inner_mesh=surfaces[f"{h}.white"]
            )
            nodes = np.where(mask_surf > 0)[0]
            mask_surf = (mask_surf > 0).astype(np.uint8)

            # Save as HDF5 for embedding pipeline
            h5_path = derivatives_dir / f"sub-{subject_id}_hemi-{h}_raw_dwi.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("dwi_surface", data=surf_data)
                f.create_dataset("bvals", data=bvals)
                f.create_dataset("bvecs", data=bvecs)
                f.create_dataset("surface_coordinates", data=mesh_coords)
                f.create_dataset("surface_faces", data=mesh_faces)
                f.create_dataset("nodes", data=nodes)
                f.create_dataset("labels", data=schaefer_hemi_data)
                f.create_dataset("mask_surf", data=mask_surf).astype(bool)

                f.attrs["subject"] = subject_id
                f.attrs["hemisphere"] = h

            # Optional: save as GIFTI (for visualization)
            gii = nib.gifti.GiftiImage()
            for direction in range(surf_data.shape[1]):
                gii.add_gifti_data_array(
                    nib.gifti.gifti.GiftiDataArray(
                        surf_data[:, direction].astype(np.float32),
                        intent="NIFTI_INTENT_NONE",
                    )
                )

            nib.save(
                gii, derivatives_dir / f"sub-{subject_id}_hemi-{h}_raw_dwi.func.gii"
            )

    def load_subject_data(self, h5_path):
        """Load subject data from HDF5 file and construct graph representation."""
        with h5py.File(h5_path, "r") as f:
            dwi_signal = np.array(f["dwi_surface"])
            bvals = np.array(f["bvals"])
            bvecs = np.array(f["bvecs"])
            vertex_indices = np.array(f["nodes"])  # cortical mask indices
            coords = np.array(f["surface_coordinates"])
            faces = np.array(f["surface_faces"])
            labels = np.array(f["labels"])
            mask_surf = np.array(f["mask_surf"])

        # Build mesh graph for neighbor lookup
        faces = faces.T
        edge_index = np.concatenate([faces[:2], faces[1:], faces[::2]], axis=1)
        edge_index = np.unique(edge_index, axis=1)

        g_graph = nx.Graph()
        g_graph.add_edges_from(edge_index.T)

        graph = g_graph

        # graph = g_graph.subgraph(vertex_indices)

        # # Labels (optional)
        # # labels = np.zeros(
        # #     coords.shape[0], dtype=np.int32
        # # )  # placeholder, can be from aparc

        mapping = {v: i for i, v in enumerate(sorted(vertex_indices))}
        # graph = nx.relabel_nodes(graph, mapping)
        # vertex_indices = np.array([mapping[v] for v in vertex_indices])
        # breakpoint()
        # dwi_signal = dwi_signal[sorted(mapping.keys())]
        # labels = labels[sorted(mapping.keys())]
        # coords = coords[sorted(mapping.keys())]

        # valid_faces_mask = np.all(np.isin(faces, list(mapping.keys())), axis=0)
        # faces_valid = faces[:, valid_faces_mask]
        # faces = np.vectorize(mapping.get)(faces_valid)

        return {
            "dwi_signal": dwi_signal,
            "bvals": bvals,
            "bvecs": bvecs,
            "vertex_indices": vertex_indices,
            "labels": labels,
            "graph": graph,
            "coords": coords,
            "faces": faces,
            "mapping": mapping,
            "mask_surf": mask_surf,
        }

    def compute_microstructure(self, subject_id: str):
        """Compute microstructural features for LCOT embedding."""
        if not self.verify_required_files(subject_id):
            print(f"[{subject_id}] Missing required files — cannot compute microstructure.")
            return
        
        #  subject_id = '101006'
        schaefer = self.schaefer_resampled

        # Create continuous region indices across hemispheres
        hemi_data = {"L": schaefer["left.data"], "R": schaefer["right.data"]}
        hemi_labels = {
            "L": schaefer["left.labels"].copy(),
            "R": schaefer["right.labels"].copy(),
        }

        # Compute unique values per hemisphere
        unique_vals = {h: np.unique(hemi_data[h]) for h in ("L", "R")}

        # Build continuous mapping across both hemispheres
        offset = 0
        maps = {}
        for h in ("L", "R"):
            vals = unique_vals[h]
            new_vals = np.arange(offset + 1, offset + len(vals) + 1)
            maps[h] = dict(zip(vals, new_vals))
            hemi_labels[h]["region_value"] = new_vals
            hemi_data[h] = np.vectorize(maps[h].get)(hemi_data[h])
            offset += len(vals)

        # Store back into schaefer structure
        schaefer["left.data"], schaefer["right.data"] = hemi_data["L"], hemi_data["R"]
        schaefer["left.labels"], schaefer["right.labels"] = (
            hemi_labels["L"],
            hemi_labels["R"],
        )

        # Combine for later use
        all_labels = pd.concat(hemi_labels.values(), ignore_index=True)
        full_parc = np.concatenate(list(hemi_data.values()))

        # Example: build mask for a target region
        target_substring = self.target_substring #"VisCent_Striate"  # "17Networks_LH_VisCent_Striate_1" #
        regions_of_interest = all_labels[
            all_labels["name"].str.contains(target_substring, case=False, na=False)
        ]
        region_value = regions_of_interest["region_value"].values
        vertex_mask = np.isin(full_parc, region_value)
        split_idx = len(hemi_data["L"])
        schaefer_mask = {
            "L": vertex_mask[:split_idx],
            "R": vertex_mask[split_idx:],
        }

        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            derivatives_dir / f"sub-{subject_id}_desc-{target_substring}_spheres.h5"
        )

        # if output_file.exists():
        #     print(
        #         f"[INFO] Found precomputed microstructure features for {subject_id}. Loading..."
        #     )
        #     with h5py.File(output_file, "r") as f:
        #         all_results = {
        #             bval: [
        #                 {
        #                     "vertex": int(v),
        #                     "attenuation": np.array(d["attenuation"]),
        #                     "neighbors": json.loads(d.attrs["neighbors"]),
        #                     "label": int(d.attrs["label"]),
        #                     "fit_status": fit_status
        #                 }
        #                 for v, d in f[bval].items()
        #             ]
        #             for bval in f.keys()
        #         }
        #     return all_results

        # Otherwise, compute the data
        print(f"[INFO] Computing microstructure features for {subject_id}...")
        self.extract_raw_data(subject_id)

        sphere = create_unit_sphere(7)
        self.sphere = sphere
        # Gradient table for b0 reference (needed for attenuation normalization)
        gtab0 = gradient_table(
            bvals=np.zeros(len(sphere.vertices)), bvecs=sphere.vertices
        )

        # --- Storage for merged hemispheres ---
        test = []
        merged_data = {
            "dwi_signal": [],
            "bvals": None,
            "bvecs": None,
            "vertex_indices": [],
            "labels": [],
            "faces": [],
        }
        merged_graph = nx.Graph()

        offset = 0
        for h, mask in schaefer_mask.items():
            rawdwi_dir = (
                self.results_root
                / "derivatives"
                / f"sub-{subject_id}"
                / "dwi"
                / f"sub-{subject_id}_hemi-{h}_raw_dwi.h5"
            )
            data_dict = self.load_subject_data(rawdwi_dir)

            n_vertices = len(data_dict["vertex_indices"])

            # Apply mask (keep only region vertices)
            vertex_indices = np.arange(
                len(data_dict["dwi_signal"])
            )  # original vertex indices
            selected_vertices = vertex_indices[mask]  # actual indices within the hemi
            masked_signal = data_dict["dwi_signal"][mask]  # corresponding DWI signals
            masked_labels = data_dict["labels"][mask]

            g_graph = nx.relabel_nodes(
                data_dict["graph"].subgraph(selected_vertices),
                {v: v + offset for v in selected_vertices},
            )

            for i, node in enumerate(g_graph.nodes()):
                g_graph.nodes[node].update(
                    {
                        "signal": masked_signal[i],
                        "label": int(masked_labels[i]),
                        "subject_id": subject_id,
                        "hemisphere": h,
                    }
                )

            merged_graph.update(g_graph)
            merged_data["dwi_signal"].append(masked_signal)
            merged_data["vertex_indices"].extend(g_graph.nodes())
            merged_data["labels"].extend(masked_labels)

            if merged_data["bvals"] is None:
                merged_data.update(
                    {"bvals": data_dict["bvals"], "bvecs": data_dict["bvecs"]}
                )

            # offset += n_vertices
            offset += len(data_dict["vertex_indices"])

        # Stack signal arrays: final shape = (n_total_vertices, n_directions)
        merged_data["dwi_signal"] = np.vstack(merged_data["dwi_signal"])
        merged_data["vertex_indices"] = np.array(
            merged_data["vertex_indices"], dtype=int
        )
        merged_data["labels"] = np.array(merged_data["labels"], dtype=int)

        # Build MAP-MRI model once for all vertices
        gtab_all = gradient_table(merged_data["bvals"], merged_data["bvecs"])
        model = MapmriModel(gtab_all, radial_order=6, laplacian_regularization=True)

        # Choose b-values to simulate on the sphere
        bvals_to_compute = [1000, 2000, 3000]
        normalize_input = False

        # Run fitting on merged hemispheres
        all_results = compute_data(
            data=merged_data,
            bvals_to_compute=bvals_to_compute,
            sphere=sphere,
            model=model,
            gtab0=gtab0,
            graph_ins=merged_graph,
            normalize_input=normalize_input,
        )
        
        all_results = self.handle_nan_values(all_results)
        
        print(f"[INFO] Saving computed features to {output_file}")
        with h5py.File(output_file, "w") as f:
            for bval, vertex_list in all_results.items():
                grp = f.create_group(bval)
                for vdata in vertex_list:
                    v = str(vdata["vertex"])
                    vgrp = grp.create_group(v)
                    vgrp.create_dataset("attenuation", data=vdata["attenuation"])
                    vgrp.attrs["neighbors"] = json.dumps([int(n) for n in vdata["neighbors"]]) #json.dumps(vdata["neighbors"])
                    vgrp.attrs["label"] = vdata["label"]
                    vgrp.attrs["fit_status"] = vdata["fit_status"]

            # Store metadata
            f.attrs["subject_id"] = subject_id
            f.attrs["bvals_to_compute"] = json.dumps(bvals_to_compute)
            f.attrs["sphere_vertices"] = len(sphere.vertices)

        print(f"[INFO] Microstructure features saved to {output_file}")
        
        return all_results

    def handle_nan_values(self, data):
        """Handle NaN values in the data after spheres have been computed."""
        # n_nan_vertices = sum([np.isnan(np.sum(v["attenuation"])) for v in data["1000"]])
        # n_nan_vertices = sum(n_nan_vertices)
        for bvalue in data.keys():
            vertex_map = {v["vertex"]: v for v in data[bvalue]}
            for element in data[bvalue]:
                att = element["attenuation"]
                if np.isnan(att).any():
                    neighbor_atts = []
                    for neighbor in element["neighbors"]:
                        neighbor_data = vertex_map.get(neighbor)
                        neighbor_att = neighbor_data["attenuation"]
                        if not np.isnan(neighbor_att).any():
                            neighbor_atts.append(neighbor_att)
                    if neighbor_atts:
                        element["attenuation"] = np.nanmean(neighbor_atts, axis=0)
                        element["fit_status"] = "repaired"
        return data
    
    def compute_embedding(self, subject_id: str):
        """Compute LCOT embedding for the given subject."""
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        spheres_file = (
            derivatives_dir / f"sub-{subject_id}_desc-{self.target_substring}_spheres.h5"
        )
        assert spheres_file.exists(), f"Spheres file not found for subject {subject_id}"
        
        output_file = (derivatives_dir / f"sub-{subject_id}_desc-{self.target_substring}_lcotembedding.h5")
        
        data, _ = load_vertexwise_attenuations(spheres_file)
        power = (data**2).mean(axis=-1)
        self.sphere = create_unit_sphere(7)
        coordinates = self.sphere.vertices
        coordinates = torch.tensor(coordinates, device=self.device, dtype=self.dtype)

        embedding = EmbeddingCircleWeights(
            d=3,
            n_projections=100,
            x_coords=coordinates,
            num_ts=100,
            device=self.device,
            dtype=self.dtype,
            random_state=42,
        )
        
        data_results_embeddings = [[], [], []]
        data_torch = torch.tensor(data, device=self.device, dtype=self.dtype)
        print(f"[INFO] Computing LCOT embedding for subject {subject_id}...")
        for i, data_split in tqdm(enumerate(split_data(data_torch, 5))):
            data_split_torch = torch.tensor(data_split, device=self.device, dtype=self.dtype)
            for s in range(3):
                    result = embedding.get_features(data_split_torch[:, s]).to("cpu")
                    data_results_embeddings[s].append(result)
        print(f"[INFO] Saving LCOT embedding to {output_file}...")
        bvals = [1000, 2000, 3000]       
        with h5py.File(output_file, "w") as f:
            # Save embeddings
            grp = f.create_group("embeddings")
            for emb, bval in zip(data_results_embeddings, bvals):
                emb = torch.cat(emb).numpy()
                grp.create_dataset(f"{bval}", data=emb)
            # Save power
            f.create_dataset("power", data=power)

            # Metadata
            meta = f.create_group("metadata")
            meta.attrs["subject_id"] = subject_id
            meta.attrs["bvals"] = bvals
        
        
    # def run_analysis(self):
    #     target_substring = self.target_substring
    #     h5_files = sorted(
    #         self.results_root.glob(
    #             f"derivatives/sub-*/dwi/*_desc-{target_substring}_spheres.h5"
    #         )
    #     )
    #     for file in tqdm(h5_files, desc="Running analysis"):
    #         try:
    #             subject_id = file.stem.split("_")[0].replace("sub-", "")
    #             self.compute_embedding(subject_id)
    #             if file.stat().st_size == 0:
    #                 print(f"[{subject_id}] Warning: File is empty.")
    #                 continue
    #             self.results[subject_id] = file
    #         except Exception as e:
    #             print(f"[{subject_id}] Error during analysis: {e}")
    
    def run_analysis(self):
        target_substring = self.target_substring
        h5_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_desc-{target_substring}_spheres.h5"
            )
        )

        def process_file(file):
            try:
                subject_id = file.stem.split("_")[0].replace("sub-", "")
                derivatives_dir = (
                    self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
                )
                embeddings_file = (derivatives_dir / f"sub-{subject_id}_desc-{self.target_substring}_lcotembedding.h5")

                if not embeddings_file.exists() or embeddings_file.stat().st_size == 0:
                    print(f"[{subject_id}] Embedding file missing or empty → computing embedding.")
                    self.compute_embedding(subject_id)
                else:
                    # embeddings_data = h5py.File(embeddings_file, "r")
                    # embeddings_dataset = embeddings_data["embeddings"]
                    # if len(embeddings_dataset.keys()) != 3:
                    #     self.compute_embedding(subject_id)
                    # print(f"[{subject_id}] Embedding file already exists and is valid.")
                    recompute = False
                    
                    with h5py.File(embeddings_file, "r") as embeddings_data:
                        if "embeddings" not in embeddings_data:
                            recompute = True
                        else:
                            embeddings_group = embeddings_data["embeddings"]
                            n_members = len(embeddings_group.keys())
                            if n_members != 3:
                                recompute = True
                            elif embeddings_group["1000"].shape == (1, 3363, 10000): 
                                recompute = True

                    if recompute:
                        print(f"[{subject_id}] Invalid embedding file (wrong #members) → recomputing.")
                        self.compute_embedding(subject_id)
                    else:
                        print(f"[{subject_id}] Embedding file already exists and is valid.")


                if file.stat().st_size == 0:
                    print(f"[{subject_id}] Warning: File is empty.")
                    return (subject_id, None)
                
                return (subject_id, embeddings_file)
            
            except Exception as e:
                print(f"[{subject_id}] Error during analysis: {e}")
                return (subject_id, None)

        # Run in parallel
        results = Parallel(n_jobs=1)(
            delayed(process_file)(file)
            for file in tqdm(h5_files, desc="Running analysis")
        )

        # Collect valid results
        for subject_id, file in results:
            if file is not None:
                self.results[subject_id] = file

    def extract_features(self):
        pass

class DefaultWandPipeline(DataPreparationBrain):
    """
    DefaultWandPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the Wand pipeline.
    Attributes:
        wand_dir (Path): The directory containing Wand data.
        results_root (Path): The root directory for storing results.
        metric (str): The metric to compute (e.g., 'rtop', 'md').
        schaefer_resampled: Resampled Schaefer atlas onto fs_LR.
        big_delta (float): The big delta value for diffusion metrics.
        small_delta (float): The small delta value for diffusion metrics.
    Methods:
        verify_subject_files(subject_id: str, metric: str) -> bool:
            Checks if both hemispheres' .scalar.gii files exist for the given subject and metric.
        compute_microstructure(subject_id: str):
            Computes microstructure metrics for the given subject and saves the results.
        run_analysis():
            Runs the analysis on the scalar files and computes average data per parcel.
        extract_features():
            Placeholder method for extracting features (to be implemented).
    """

    def __init__(self, config):
        super().__init__(config)
        self.wand_dir = Path(config["wand_base_path"])
        self.derivatives_in = self.wand_dir / "derivatives"
        self.results_root = Path(config["wand_results_path"]) / "default"
        self.metric = config["metric_to_compute"]
        # breakpoint()
        self.scale = config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config.get("big_delta_wand", 24e-3)
        self.small_delta = config.get("small_delta_wand", 7e-3)
        self.big_delta_per_bvalue = config.get(
            "big_delta_per_bvalue",
            {2200: 24, 4000: 30, 4400: 24, 8000: 30, 5800: 42, 7750: 55, 11600: 42, 15500: 55},
        )

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if both hemispheres' .scalar.gii files exist for the given subject and metric.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        left_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-L_param-{metric}.scalar.gii"
        )
        right_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-R_param-{metric}.scalar.gii"
        )

        return left_file.exists() and right_file.exists()

    def compute_microstructure(self, subject_id: str):
        """Compute microstructure metrics for the given subject and save the results."""
        try:
            # breakpoint()
            # layout = bids.BIDSLayout(self.wand_dir, derivatives=self.derivatives_in, validate=False)
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)
            ####
            # derivatives_dir = self.derivatives_in / f"sub-{subject_id}" / "dwi"

            # subject_dir = self.wand_dir / subject_id / "ses-02"

            # diffusion_dir = subject_dir / "dwi"
            
            aparcaseg_path = self.derivatives_in / f"smriprep/sub-{subject_id}/ses-02/anat/sub-{subject_id}_ses-02_desc-aparcaseg_dseg.nii.gz"
            dwi_path = self.derivatives_in / f"preprocess/sub-{subject_id}/sub-{subject_id}_ses-02_acq-AxCaliberConcat_space-individualT1_desc-eddycorrected_bbreg_dwi.nii.gz"
            bvals_path = self.wand_dir / f"sub-{subject_id}/ses-02/dwi/sub-{subject_id}_ses-02_acq-AxCaliberConcat_dwi.bval"
            bvecs_path = self.derivatives_in / f"preprocess/sub-{subject_id}/ses-02/dwi/sub-{subject_id}_ses-02_acq-AxCaliberConcat_desc-rotated_dwi.bvec"
            bvecs = np.loadtxt(bvecs_path).T
            bvals = np.loadtxt(bvals_path)
            dwi_nib  = nib.load(dwi_path)
            
            parcellation_dwi = nimage.resample_img(
                aparcaseg_path,
                target_affine=dwi_nib.affine,
                target_shape=dwi_nib.shape[:3],
                interpolation='nearest',
                force_resample=True, copy_header=True
            )
            
            b0 = nimage.index_img(dwi_nib, 0)
            
            # dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
            # bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
            # bvals = np.loadtxt(bvals)
            # bvecs = np.loadtxt(bvecs).T

            labels = extract_selected_labels(aparc_aseg)
            aparc_resampled = nimage.resample_to_img(
                aparc_aseg,
                nodif_mask,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            ctx_mask, vent_mask = create_masks(aparc_resampled, labels)

            surfaces = {
                f"{h}.{s}": subject_dir
                / "T1w"
                / "fsaverage_LR32k"
                / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
                for s in ("white", "pial")
                for h in ("L", "R")
            }

            if self.metric == "rtop":
                rtop_img = compute_rtop(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    rtop_img,
                    derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
                )
                project_to_surface(
                    rtop_img,
                    ctx_mask,
                    surfaces,
                    derivatives_dir,
                    subject_id,
                    self.metric,
                )
            elif self.metric == "md":
                md_img = compute_md(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
                )
                project_to_surface(
                    md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric
                )

        except Exception as e:
            print(f"[{subject_id}] Error during microstructure: {e}")

    # def run_analysis(self):  # Doing a test
    def run_analysis_good(self):
        scalar_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
            )
        )
        for left_file in tqdm(scalar_files, desc="Running analysis"):
            try:
                subject_id = left_file.stem.split("_")[0].replace("sub-", "")
                right_file = left_file.with_name(
                    left_file.name.replace("hemi-L", "hemi-R")
                )

                left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
                    0, 7
                )
                right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
                    0, 7
                )

                avg_data = average_per_parcel(
                    left_data, right_data, self.schaefer_resampled
                )
                self.results[subject_id] = avg_data
            except Exception as e:
                print(f"[{subject_id}] Error during analysis: {e}")
    
    # def run_analysis_region(self):
    def run_analysis(self):
        scalar_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
            )
        )
        for left_file in tqdm(scalar_files, desc="Running analysis"):
            try:
                subject_id = left_file.stem.split("_")[0].replace("sub-", "")
                right_file = left_file.with_name(
                    left_file.name.replace("hemi-L", "hemi-R")
                )

                left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
                    0, 7
                )
                right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
                    0, 7
                )
                
                # target = "VisCent_Striate"
                # target = self.config["region_name"]
                target = self.config["models"][0]["params"]["region_name"]
                # target = None
                avg_data = extract_region_data(
                    left_data, right_data, self.schaefer_resampled, target_substring=target, average=False
                )
                self.results[subject_id] = avg_data
            except Exception as e:
                print(f"[{subject_id}] Error during analysis: {e}")

    def extract_features(self):
        pass  
    
# class DefaultCamcanPipeline(DataPreparationBrain):
#     """
#     DefaultCamcanPipeline is a class that extends the DataPreparationBrain class to handle
#     the preprocessing of brain data for the CamCAN pipeline.
#     Attributes:
#         camcan_dir (Path): The directory containing CamCAN data.
#         results_root (Path): The root directory for storing results.
#         metric (str): The metric to compute (e.g., 'rtop', 'md').
#         schaefer_resampled: Resampled Schaefer atlas onto fs_LR.
#         big_delta (float): The big delta value for diffusion metrics.
#         small_delta (float): The small delta value for diffusion metrics.
#     Methods:
#         verify_subject_files(subject_id: str, metric: str) -> bool:
#             Checks if both hemispheres' .scalar.gii files exist for the given subject and metric.
#         compute_microstructure(subject_id: str):
#             Computes microstructure metrics for the given subject and saves the results.
#         run_analysis():
#             Runs the analysis on the scalar files and computes average data per parcel.
#         extract_features():
#             Placeholder method for extracting features (to be implemented).
#     """

#     def __init__(self, config):
#         super().__init__(config)
#         self.camcan_dir = Path(config["camcan_base_path"])
#         self.results_root = Path(config["camcan_results_path"]) / "default"
#         self.metric = config["metric_to_compute"]
#         breakpoint()
#         self.scale = config.get("scale", 1000)
#         self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
#         self.big_delta = config["big_delta"]
#         self.small_delta = config["small_delta"]

#     def verify_subject_files(self, subject_id: str, metric: str) -> bool:
#         """
#         Check if both hemispheres' .scalar.gii files exist for the given subject and metric.
#         """
#         derivatives_dir = (
#             self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
#         )
#         left_file = (
#             derivatives_dir / f"sub-{subject_id}_hemi-L_param-{metric}.scalar.gii"
#         )
#         right_file = (
#             derivatives_dir / f"sub-{subject_id}_hemi-R_param-{metric}.scalar.gii"
#         )

#         return left_file.exists() and right_file.exists()

#     def compute_microstructure(self, subject_id: str):
#         """Compute microstructure metrics for the given subject and save the results."""
#         try:
#             derivatives_dir = (
#                 self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
#             )
#             derivatives_dir.mkdir(parents=True, exist_ok=True)

#             subject_dir = self.hcp_dir / subject_id

#             diffusion_dir = subject_dir / "T1w" / "Diffusion"
#             dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
#             bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
#             bvals = np.loadtxt(bvals)
#             bvecs = np.loadtxt(bvecs).T
#             nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz"

#             aparc_aseg = subject_dir / "T1w" / "aparc+aseg.nii.gz"

#             labels = extract_selected_labels(aparc_aseg)
#             aparc_resampled = nimage.resample_to_img(
#                 aparc_aseg,
#                 nodif_mask,
#                 interpolation="nearest",
#                 force_resample=True,
#                 copy_header=True,
#             )

#             ctx_mask, vent_mask = create_masks(aparc_resampled, labels)

#             surfaces = {
#                 f"{h}.{s}": subject_dir
#                 / "T1w"
#                 / "fsaverage_LR32k"
#                 / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
#                 for s in ("white", "pial")
#                 for h in ("L", "R")
#             }

#             if self.metric == "rtop":
#                 rtop_img = compute_rtop(
#                     dwi_nib,
#                     ctx_mask,
#                     vent_mask,
#                     bvals,
#                     bvecs,
#                     self.big_delta,
#                     self.small_delta,
#                 )
#                 nib.save(
#                     rtop_img,
#                     derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
#                 )
#                 project_to_surface(
#                     rtop_img,
#                     ctx_mask,
#                     surfaces,
#                     derivatives_dir,
#                     subject_id,
#                     self.metric,
#                 )
#             elif self.metric == "md":
#                 md_img = compute_md(
#                     dwi_nib,
#                     ctx_mask,
#                     vent_mask,
#                     bvals,
#                     bvecs,
#                     self.big_delta,
#                     self.small_delta,
#                 )
#                 nib.save(
#                     md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
#                 )
#                 project_to_surface(
#                     md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric
#                 )

#         except Exception as e:
#             print(f"[{subject_id}] Error during microstructure: {e}")

#     # def run_analysis(self):  # Doing a test
#     def run_analysis_good(self):
#         scalar_files = sorted(
#             self.results_root.glob(
#                 f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
#             )
#         )
#         for left_file in tqdm(scalar_files, desc="Running analysis"):
#             try:
#                 subject_id = left_file.stem.split("_")[0].replace("sub-", "")
#                 right_file = left_file.with_name(
#                     left_file.name.replace("hemi-L", "hemi-R")
#                 )

#                 left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
#                     0, 7
#                 )
#                 right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
#                     0, 7
#                 )

#                 avg_data = average_per_parcel(
#                     left_data, right_data, self.schaefer_resampled
#                 )
#                 self.results[subject_id] = avg_data
#             except Exception as e:
#                 print(f"[{subject_id}] Error during analysis: {e}")
    
#     # def run_analysis_region(self):
#     def run_analysis(self):
#         scalar_files = sorted(
#             self.results_root.glob(
#                 f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
#             )
#         )
#         for left_file in tqdm(scalar_files, desc="Running analysis"):
#             try:
#                 subject_id = left_file.stem.split("_")[0].replace("sub-", "")
#                 right_file = left_file.with_name(
#                     left_file.name.replace("hemi-L", "hemi-R")
#                 )

#                 left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
#                     0, 7
#                 )
#                 right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
#                     0, 7
#                 )
                
#                 # target = "VisCent_Striate"
#                 # target = self.config["region_name"]
#                 target = self.config["models"][0]["params"]["region_name"]
#                 # target = None
#                 avg_data = extract_region_data(
#                     left_data, right_data, self.schaefer_resampled, target_substring=target, average=False
#                 )
#                 self.results[subject_id] = avg_data
#             except Exception as e:
#                 print(f"[{subject_id}] Error during analysis: {e}")

#     def extract_features(self):
#         pass

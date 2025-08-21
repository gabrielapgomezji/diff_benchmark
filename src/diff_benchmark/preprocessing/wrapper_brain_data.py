from pathlib import Path

import h5py
import networkx as nx
import nibabel as nib
import nilearn as ni

# rtop_pipeline.py
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.reconst.mapmri import MapmriModel
from nilearn import image as nimage
from tqdm import tqdm

from diff_benchmark.preprocessing.wrapper_brain_base import DataPreparationBrain
from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
    average_per_parcel,
    compute_data,
    compute_md,
    compute_rtop,
    create_masks,
    extract_selected_labels,
    project_to_surface,
    resample_schaefer_onto_fs_lr,
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

                avg_data = average_per_parcel(
                    left_data, right_data, self.schaefer_resampled
                )
                self.results[subject_id] = avg_data
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
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config["big_delta"]
        self.small_delta = config["small_delta"]

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        pass

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

            # Save as HDF5 for embedding pipeline
            h5_path = derivatives_dir / f"sub-{subject_id}_hemi-{h}_raw_dwi.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("dwi_surface", data=surf_data)
                f.create_dataset("bvals", data=bvals)
                f.create_dataset("bvecs", data=bvecs)
                f.create_dataset("surface_coordinates", data=mesh_coords)
                f.create_dataset("surface_faces", data=mesh_faces)
                f.create_dataset("nodes", data=nodes)

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
            # breakpoint()
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

        # Build mesh graph for neighbor lookup
        faces = faces.T
        edge_index = np.concatenate([faces[:2], faces[1:], faces[::2]], axis=1)
        edge_index = np.unique(edge_index, axis=1)

        g_graph = nx.Graph()
        g_graph.add_edges_from(edge_index.T)

        graph = g_graph.subgraph(vertex_indices)

        # Labels (optional)
        labels = np.zeros(
            coords.shape[0], dtype=np.int32
        )  # placeholder, can be from aparc

        return {
            "dwi_signal": dwi_signal,
            "bvals": bvals,
            "bvecs": bvecs,
            "vertex_indices": vertex_indices,
            "labels": labels,
            "graph": graph,
            "coords": coords,
            "faces": faces,
        }

    def compute_microstructure(self, subject_id: str):
        """Compute microstructural features for LCOT embedding."""
        sphere = create_unit_sphere(7)

        # Gradient table for b0 reference (needed for attenuation normalization)
        gtab0 = gradient_table(
            bvals=np.zeros(len(sphere.vertices)), bvecs=sphere.vertices
        )

        # --- Storage for merged hemispheres ---
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
        for h in ("L", "R"):
            # rawdwi_dir = self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi" / f"sub-{subject_id}_hemi-{h}_raw_dwi.func.gii"
            rawdwi_dir = (
                self.results_root
                / "derivatives"
                / f"sub-{subject_id}"
                / "dwi"
                / f"sub-{subject_id}_hemi-{h}_raw_dwi.h5"
            )
            data_dict = self.load_subject_data(rawdwi_dir)

            n_vertices = len(data_dict["vertex_indices"])

            # Offset vertex indices and faces so R hemisphere doesn't overwrite L indices
            data_dict["vertex_indices"] = data_dict["vertex_indices"] + offset
            data_dict["faces"] = data_dict["faces"] + offset

            # Append hemisphere's signals
            merged_data["dwi_signal"].append(data_dict["dwi_signal"])
            merged_data["vertex_indices"].extend(data_dict["vertex_indices"])
            merged_data["labels"].extend(data_dict["labels"])
            merged_data["faces"].extend(data_dict["faces"])

            # Save bvals/bvecs (identical for both hemispheres)
            if merged_data["bvals"] is None:
                merged_data["bvals"] = data_dict["bvals"]
                merged_data["bvecs"] = data_dict["bvecs"]

            # Merge graphs
            g_graph = data_dict["graph"].copy()
            mapping = {old: old + offset for old in g_graph.nodes()}
            g_graph = nx.relabel_nodes(g_graph, mapping)
            merged_graph.add_nodes_from(g_graph.nodes())
            merged_graph.add_edges_from(g_graph.edges())

            offset += n_vertices

        # breakpoint()
        # Stack signal arrays: final shape = (n_total_vertices, n_directions)
        merged_data["dwi_signal"] = np.vstack(merged_data["dwi_signal"])
        merged_data["vertex_indices"] = np.array(
            merged_data["vertex_indices"], dtype=int
        )
        merged_data["labels"] = np.array(merged_data["labels"], dtype=int)
        merged_data["faces"] = np.array(merged_data["faces"], dtype=int)

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
        # breakpoint()
        return all_results

    def run_analysis(self):
        pass

    def extract_features(self):
        pass

Data and Preprocessing

extract_raw_data.py
This module handles the extraction and preprocessing of raw DWI (Diffusion-Weighted Imaging) data by projecting it onto the cortical surface. The processed data is saved in a structured HDF5 format for downstream analysis and modeling.

NOTES: subjects are skipped automatically ifthe output file alreaddy exists or any required input file is missing. Also, errors in the processing are logged in the console.

Steps:

For each subject, and in parallel, the script:

- Loads DWI data and associated metadata (bvals, bvecs).

- Projects the volumetric DWI signal onto the left hemisphere cortical surface using nilearn.

- Loads cortical surface labels and mesh geometry (coordinates and faces).

- Saves all processed data into a structured .h5 file.


Configuration requirements in comfiguration.yml file:

base_path: "~/data/HCP"         # Root directory containing subject folders
data_path: "~/data/masks"       # Directory with subject-specific mask files
deen_path: "~/data/labels/deen.L.label.gii"  # Path to left hemisphere surface labels
results_path: "~/processed_data"            # Output directory for processed files


Python requirements:

pip install numpy nibabel nilearn h5py pyyaml joblib


How to run the code (considering you are in the directory of the project):

poetry run python scripts/data/extract_raw_data.py


Output:

For each subject, a new folder is created under results_path/<subject>/ containing raw_surface_data.h5 that includes:

- left_dwi_surface: Projected DWI values on the cortical surface.

- surface_labels: Cortical region labels (from DEEN parcellation).

- nodes_left: Node indices with non-zero labels.

- surface_coordinates: Vertex coordinates of the cortical mesh.

- surface_faces: Mesh faces defining surface triangles.

- bvals, bvecs: Diffusion gradient info.

Metadata stored as HDF5 attributes:
- subject: Subject ID

- hemisphere: "left"

- source: "projected DWI on surface"

- description: "Raw DWI signal projected on cortical surface using nilearn.surface.vol_to_surf"
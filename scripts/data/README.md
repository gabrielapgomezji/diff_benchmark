## Data and Preprocessing

### extract_raw_data.py
This module handles the extraction and preprocessing of raw DWI (Diffusion-Weighted Imaging) data by projecting it onto the cortical surface. The processed data is saved in a structured HDF5 format for downstream analysis and modeling.

NOTES: subjects are skipped automatically ifthe output file alreaddy exists or any required input file is missing. Also, errors in the processing are logged in the console.

**Steps:**

For each subject, and in parallel, the script:

- Loads DWI data and associated metadata (bvals, bvecs).

- Projects the volumetric DWI signal onto the left hemisphere cortical surface using nilearn.

- Loads cortical surface labels and mesh geometry (coordinates and faces).

- Saves all processed data into a structured .h5 file.


**Configuration requirements in comfiguration.yml file:**

`base_path: "~/data/HCP"`        # Root directory containing subject folders

`data_path: "~/data/masks"`       # Directory with subject-specific mask files

`deen_path: "~/data/labels/deen.L.label.gii"`  # Path to left hemisphere surface labels

`results_path: "~/processed_data"`            # Output directory for processed files


**Python requirements:**

`pip install numpy nibabel nilearn h5py pyyaml joblib`


How to run the code (considering you are in the directory of the project):

`poetry run python scripts/data/extract_raw_data.py`


**Output:**

For each subject, a new folder is created under results_path/<subject>/ containing raw_surface_data.h5 that includes:

- left_dwi_surface: Projected DWI values on the cortical surface.

- surface_labels: Cortical region labels (from DEEN parcellation).

- nodes_left: Node indices with non-zero labels.

- surface_coordinates: Vertex coordinates of the cortical mesh.

- surface_faces: Mesh faces defining surface triangles.

- bvals, bvecs: Diffusion gradient info.

**Metadata stored as HDF5 attributes:**
- subject: Subject ID

- hemisphere: "left"

- source: "projected DWI on surface"

- description: "Raw DWI signal projected on cortical surface using nilearn.surface.vol_to_surf"



### brain_data_preprocessing.py

This script performs preprocessing of the raw surface-projected DWI data for a single subject. It loads the raw .h5 file generated in the previous step, applies custom transformations, and saves the result in a processed format ready for model input.

This script takes one command-line argument: the subject ID.

**How to run:**
python preprocessing_brain_data.py <subject_id>

** Replace <subject_id> with the actual subject folder name (e.g., 100206). (e.g. python preprocessing_brain_data.py 100206)

**What It Does**
- Loads the raw surface data from:
results_path/<subject_id>/raw_surface_data.h5

- Saves preprocessed output to:
results_path/<subject_id>/processed/

- Calls the function:
preprocess_subject()
(defined in diff_benchmark.preprocessing.brain_data)

- Measures and prints the total runtime of the preprocessing task.


**Inputs:**
raw_surface_data.h5: The raw cortical surface projection data for the subject.

**Outputs:**
A new folder: results_path/<subject_id>/processed/

Contains subject-specific preprocessed data (format depends on what preprocess_subject generates).

**Error Handling:**
- If no subject ID is provided, the script will raise an error:
    ValueError: Please provide a subject ID as the first argument.
- Paths are checked via the YAML configuration. Make sure the raw .h5 file exists before running this step.


**Inputs:**

- CSV file path (csv_path): Path to the CSV file.

- Target columns (target_columns): A NumPy array of column names to retain.

**Output:**
Returns a cleaned pandas DataFrame with:

- Only the relevant columns (Subject + target_columns)

- All selected rows free of missing values

- Gender encoded as numeric (if present): NEED TO WORK ON HOW TO HANDLE THIS

**Processing Steps**
- Reads the CSV into a pandas DataFrame (expects "Subject" as a column).

- Selects the Subject column plus the specified target_columns.

- Encodes "Gender" column as binary if it exists:

    "M" → 1

    "F" → 0

- Drops any row that has a missing (NaN) value in one or more of the target columns.
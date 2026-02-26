# How to run

## Using Docker

- Pull the docker image

  ```bash
  docker pull ghcr.io/man-shu/diffusion-preprocessing:main
  ```

- **Optionally**, you can also build the docker image

  - Clone this repository and navigate to the directory

    ```bash
    git clone git@github.com:man-shu/diffusion-preprocessing.git
    cd diffusion-preprocessing
    ```

  - If you're using a machine with x86_64 architecture (check with `uname -m`):

    ```bash
    docker image build --tag ghcr.io/man-shu/diffusion-preprocessing:main .
    ```

  - If you're using a machine with ARM architecture (for example, Apple M1):

    ```bash
    docker image build --platform linux/x86_64 --tag ghcr.io/man-shu/diffusion-preprocessing:main .
    ```

- Run the container

  - To run the full pipeline, with surface reconstruction (smriprep workflow) and diffusion preprocessing:

    ```bash
    docker container run --rm --interactive \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source=/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data,target=/home/input \
    ghcr.io/man-shu/diffusion-preprocessing:main /home/input/WAND-downsampled \
    /home/input/WAND-downsampled/derivatives \
    --work-dir /home/input/cache \
    --output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
    --cifti-output 91k \
    --nprocs 1 \
    --omp-nthreads 8 \
    --participant-label sub-01187 \
    --no-msm \
    --no-submm-recon \
    --recon \
    --preproc 
    ```

  - If you already have surface reconstruction results and just want to run diffusion preprocessing, just specify the `--preproc` flag:

    ```bash
    docker container run --rm --interactive \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source=/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data,target=/home/input \
    ghcr.io/man-shu/diffusion-preprocessing:main /home/input/WAND-downsampled \
    /home/input/WAND-downsampled/derivatives \
    --work-dir /home/input/cache \
    --output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
    --nprocs 1 \
    --omp-nthreads 8 \
    --participant-label sub-01187 \
    --preproc
    ```

  - If you're using a machine with ARM architecture (for example, Apple M1), you may need to specify the platform explicitly
    with the `--platform` flag:

    ```bash
    docker container run --rm --interactive \
    --platform linux/x86_64 \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source=/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data,target=/home/input \
    ghcr.io/man-shu/diffusion-preprocessing:main /home/input/WAND-downsampled \
    /home/input/WAND-downsampled/derivatives \
    --work-dir /home/input/cache \
    --output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
    --cifti-output 91k \
    --nprocs 1 \
    --omp-nthreads 8 \
    --participant-label sub-01187 \
    --no-msm \
    --no-submm-recon \
    --recon \
    --preproc 
    ```

- If you want to filter certain files from the BIDS dataset, you can use the `--bids-filter-file` flag to specify a JSON file with the filtering criteria.
  For example, to include only diffusion-weighted images (DWIs) with the acquisition label `AxCaliber1`, create a file named `bids_filter.json` with the following content:

  ```json
  {"dwi" : {"acquisition": "AxCaliber1"},
   "bval": {"acquisition": "AxCaliber1"}, 
   "bvec": {"acquisition": "AxCaliber1"}}
  ```

- Then, run the container with the `--bids-filter-file` flag:

  ```bash
    docker container run --rm --interactive \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source=/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data,target=/home/input \
    ghcr.io/man-shu/diffusion-preprocessing:main /home/input/WAND-downsampled \
    /home/input/WAND-downsampled/derivatives \
    --work-dir /home/input/cache \
    --output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
    --nprocs 1 \
    --omp-nthreads 8 \
    --participant-label sub-01187 \
    --bids-filter-file /home/input/bids_filter.json \
    --preproc
  ```

- See here for details about `--bids-filter-file`: <https://fmriprep.org/en/25.1.4/faq.html#how-do-i-select-only-certain-files-to-be-input-to-fmriprep>

## Using Singularity

- Pull the singularity image

  ```bash
  singularity pull oras://ghcr.io/man-shu/diffusion-preprocessing:main_singularity
  ```

- Run the singularity image

  ```bash
  singularity exec \
  --env-file /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/singularity_env.txt \
  --bind /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data:/home/input \
  /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/diffusion-preprocessing_main_singularity.sif \
  /opt/miniconda3/bin/diffusion_pipelines \
  /home/input/WAND-downsampled \
  /home/input/WAND-downsampled/derivatives \
  --work-dir /home/input/cache \
  --output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
  --cifti-output 91k \
  --nprocs 1 \
  --omp-nthreads 8 \
  --participant-label sub-01187 \
  --no-msm \
  --no-submm-recon \
  --recon \
  --preproc 
  ```

- Alternatively, you can run the singularity image in an interactive shell

  ```bash
  singularity shell --env-file singularity_env.txt \
  --bind ./data:/home/input diffusion-preprocessing_main_singularity.sif
  ```

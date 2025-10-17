# LAUNCH BENCHMARK AND CONFIGURATION FILE
## Launch Benchmark
To launch the main script if:
- You're using poetry: 
`poetry install`
`poetry run python scripts/main.py`
- You're using Jean Zay:
`module load pytorch-gpu/py3/2.8.0`
`python scripts/main.py`

This benchmark allows to evaluate how well different models predict cognitive and demographic information from different microstructural parameters. For every microstructura

Depending on the models, make sure that in the main the input features correspond to the type of model you're evaluating. (For the moment we can only evaluate at the same time models that take as input the same type of data. However, it is possible to re-run the benchmark with models that have other configurations and then compare the obtained results).
Make sure that just one of this `brain_preparator` is defined
- For array-like input models (as classical ML pipelines or dummy models)
`brain_preparator = DefaultHcpPipeline(config)` for HCP dataset
- For image-like input models (as 2d or 3d convolutional models)
`brain_preparator = ImageHcpPipeline(config)` for HCP dataset
- For sphere-like input models 
`brain_preparator = LcotEmbedHcpPipeline(config)` for HCP dataset

## Configurationo file
Some of the specifications for the `configuration.yaml` file and the elements that must contain:
### configuration.yaml elements
n_jobs: 40
base_path: # Path to the unprocessed dataset
data_path: # Path to the mask data (processed for each subject)
deen_path: # Path to the ROI to analyse
results_path_2: # Path to temporary storage of data
csv_file: # Path to csv demographic data
target_columns:
  - 'Gender'
train_size: 0.8
val_size: 0.1
validation_size: 0.1
n_splits: 5
random_state: 42
batch_size: 32
num_epochs: 100
n_components: 2
model_name: dummy_classifier
debugging_analysis: True # True or False
models:
- name: "pca_forest"
  params:
    name: pca_forest with rtop
    comment: "PCA + Random Forest"
    batch_size: 32
- name: 2dcnn_lite
  params:
    name: 2dcnn_lite
    comment: Prediction with best checkpoint. With folds. Test 
    input_slices: 145
    num_classes: 2
    device: cuda
    pretrained: true
    freeze_backbone: true
    trainable_blocks: 0
    epochs: 50
    batch_size: 32
    dropout: 0.3
    scheduler_type: exponential # plateau # step # 
    learning_rate: 0.0001
    weight_decay: 0.1


Please, ensure that in the model configurations the variable `name` correspond to an existing name of valuable and included models in the benchmark. To confirm that yoour model name is included go to `src/diff_benchmark/models/model_configuration.py`



# ADD A MODEL TO THE BENCHMARK
Steps:
- Go to `src/diff_benchamrk/moodels/base.py`
There you will see 3 abstract classes that you will be able to create your model from: `NumpyAbstractModel` for classical and `sklearn` pipelines, `TorchAbstractModel` for Deep Learning moodels where you have to specifically define the training in the fit function and `LightningModel` for Deep Learning models where the training, testing and validation is handled by lightning (RECOOMMENDED for deep learning).
- Create a script with your model with the mandatory functions
These are mainly `fit`,  `predict` and `_dataloader_to_numpy`(which is a function to ensure the correct functioning of the main script for all models classical and deep learning).
### configuration.yaml elements
n_jobs: 40
base_path: # Path to the unprocessed dataset
data_path: # Path to the mask data (processed for each subject)
deen_path: # Path to the ROI to analyse
results_path_2: # Path to temporary storage of data
csv_file: # Path to csv demographic data
target_columns:
  # - 'Age_in_Yrs'
  # - 'WM_Task_Acc'
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
## MLP CLASSIFIER
input_dim: # input dimension of the data
hidden_layers: [128, 64]
output_dim: 1
learning_rate: 1e-3
dropout_rate: 0.2
epochs: 100
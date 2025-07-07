import h5py

def save_dataset(X, y, gender, output_file='dataset.h5'):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('y', data=y)
        f.create_dataset('gender', data=gender)

def load_dataset(dataset_filename):
    with h5py.File('custom_dataset.h5', 'r') as f:
        X = f['X'][:]
        y = f['y'][:]
        gender = f['gender'][:]
    return X, y, gender
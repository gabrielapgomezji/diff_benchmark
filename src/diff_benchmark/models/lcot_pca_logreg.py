import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diff_benchmark.models.base import NumpyAbstractModel


class EmbeddingsPCALogisticRegression(NumpyAbstractModel):
    """
    PCA + Logistic Regression for LCOT embeddings.
    
    MEMORY-OPTIMIZED for high-dimensional embeddings (millions of features).
    
    This is a simpler baseline to test whether the embeddings contain useful information
    before diving into more complex kernel methods. It:
    1. Loads embeddings from the dataloader (n_subjects, n_spheres, n_bvals, d)
    2. Flattens them to (n_subjects, n_spheres * n_bvals * d)
    3. Applies StandardScaler + PCA + LogisticRegression in a sklearn pipeline
    4. Uses GridSearchCV to find optimal n_components and regularization strength
    
    Memory optimizations:
    - Uses IncrementalPCA or TruncatedSVD for memory-efficient dimensionality reduction
    - Reduced hyperparameter grid
    - Configurable n_jobs to control parallel memory usage
    - Option to skip scaling for further memory savings
    
    This provides a quick sanity check on whether the embeddings are informative.
    """

    def __init__(self, dtype=torch.float32, n_jobs=1, 
                 reducer='truncated_svd',  # 'incremental_pca', 'truncated_svd', or 'pca'
                 pca_batch_size=100, 
                 max_pca_components=300,  # Reduced from 500
                 skip_scaling=True,  # Default to True to save memory
                 n_components_grid=None,  # Custom grid for n_components
                 C_grid=None,  # Custom grid for C values
                 cv=3,  # Reduced from 5 to save memory
                 **kwargs):
        """
        Initialize the PCA + Logistic Regression model with grid search.
        
        Args:
            dtype: Data type for loading embeddings (default: torch.float32)
            n_jobs: Number of parallel jobs for GridSearchCV (default: 1 to save memory)
            reducer: Dimensionality reduction method:
                - 'truncated_svd': Very memory-efficient, no centering (fastest, least memory) [DEFAULT]
                - 'incremental_pca': Memory-efficient PCA (good for n_features >> n_samples)
                - 'pca': Standard PCA (use only for smaller feature spaces)
            pca_batch_size: Batch size for IncrementalPCA (default: 100)
            max_pca_components: Maximum number of PCA components to try (default: 300)
            skip_scaling: Skip StandardScaler to save memory (default: True)
            n_components_grid: Custom grid for n_components (default: [10, 50, 100, 200])
            C_grid: Custom grid for regularization C (default: [0.1, 1, 10])
            cv: Number of cross-validation folds (default: 3 to save memory)
        """
        self.dtype = dtype
        self.reducer_type = reducer
        self.pca_batch_size = pca_batch_size
        self.skip_scaling = skip_scaling
        
        # Choose dimensionality reduction method
        if reducer == 'incremental_pca':
            from sklearn.decomposition import IncrementalPCA
            reducer_estimator = IncrementalPCA(batch_size=pca_batch_size)
            print(f"Using IncrementalPCA with batch_size={pca_batch_size}")
        elif reducer == 'truncated_svd':
            from sklearn.decomposition import TruncatedSVD
            reducer_estimator = TruncatedSVD()
            print("Using TruncatedSVD (most memory efficient, no mean centering)")
        else:  # 'pca'
            reducer_estimator = PCA()
            print("Using standard PCA")
        
        # Create sklearn pipeline
        if skip_scaling:
            print("Skipping StandardScaler to save memory")
            pipeline = Pipeline(
                [
                    ("reducer", reducer_estimator),
                    ("logreg", LogisticRegression(max_iter=1000)),
                ]
            )
            reducer_param_prefix = "reducer"
        else:
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reducer", reducer_estimator),
                    ("logreg", LogisticRegression(max_iter=1000)),
                ]
            )
            reducer_param_prefix = "reducer"

        # Grid of hyperparameters - VERY CONSERVATIVE to avoid memory issues
        # For 735 samples, we can't use more than ~730 components anyway
        max_components = min(max_pca_components, 700)  # Conservative upper bound
        
        # Use custom grids if provided, otherwise use conservative defaults
        if n_components_grid is None:
            n_components_grid = [10, 50, 100, min(200, max_components)]  # Only 4 values
        
        if C_grid is None:
            C_grid = [0.1, 1, 10]  # Only 3 values - reduced from 4
        
        param_grid = {
            f"{reducer_param_prefix}__n_components": n_components_grid,
            "logreg__C": C_grid,
            "logreg__penalty": ["l2"],
            "logreg__solver": ["lbfgs"],
        }

        total_combinations = len(n_components_grid) * len(C_grid)
        print(f"Grid search with {n_jobs} parallel jobs, {cv}-fold CV")
        print(f"Hyperparameter grid: {total_combinations} combinations")
        print(f"Total fits: {total_combinations * cv}")
        
        # Grid search object with AGGRESSIVE memory-conscious settings
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,  # Reduced from 5 to save memory
            n_jobs=n_jobs,  # Should be 1 for safety
            verbose=2,
            pre_dispatch='1',  # CRITICAL: Only dispatch 1 job at a time to minimize memory
        )

    def _dataloader_to_numpy(self, dataloader):
        """
        Convert dataloader with embeddings to flattened numpy arrays.
        
        Args:
            dataloader: DataLoader yielding (data, targets, _) where
                       data is dict with 'embeddings' key
        
        Returns:
            features: numpy array of shape (n_subjects, n_spheres * n_bvals * d)
            targets: numpy array of shape (n_subjects,)
        """
        print("\nLoading embeddings from dataloader...")
        
        embeddings_list = []
        targets_list = []
        
        for i, (data, targets_batch, _) in enumerate(dataloader):
            print(f"  Processing batch {i+1}/{len(dataloader)}")
            
            # Extract embeddings
            embeddings = data["embeddings"].to(self.dtype)
            
            # Remove batch dimension if present
            if embeddings.dim() == 5:
                embeddings = embeddings.squeeze(1)
            
            # embeddings shape: (batch_size, n_spheres, n_bvals, d)
            # Flatten to (batch_size, n_spheres * n_bvals * d)
            batch_size = embeddings.shape[0]
            embeddings_flat = embeddings.reshape(batch_size, -1)
            
            # CRITICAL: Convert to float32 to halve memory usage (sklearn uses float64 by default)
            embeddings_list.append(embeddings_flat.cpu().numpy().astype(np.float32))
            targets_list.append(targets_batch.cpu().numpy())
            
            del embeddings, embeddings_flat
        
        # Concatenate all batches
        features = np.concatenate(embeddings_list, axis=0)  # Now in float32!
        targets = np.concatenate(targets_list, axis=0)
        
        print(f"Loaded embeddings shape: {features.shape}")
        print(f"Loaded embeddings dtype: {features.dtype} (using float32 to save memory)")
        print(f"Memory usage: ~{features.nbytes / 1e9:.2f} GB")
        print(f"Targets shape: {targets.shape}")
        
        return features, targets

    def fit(self, dataloader):
        """
        Fit PCA and logistic regression on embeddings with grid search.
        
        Args:
            dataloader: DataLoader yielding (data, targets, _)
        """
        print("=" * 60)
        print("EmbeddingsPCALogisticRegression - Training")
        print("=" * 60)
        
        features, targets = self._dataloader_to_numpy(dataloader)
        
        print("\nRunning GridSearchCV (5-fold CV)...")
        print(f"Feature dimensionality: {features.shape[1]}")
        print(f"Number of samples: {features.shape[0]}")
        
        self.model.fit(features, targets.flatten())
        
        print("\n" + "=" * 60)
        print("Grid Search Results:")
        print("=" * 60)
        print(f"Best parameters: {self.model.best_params_}")
        print(f"Best CV score: {self.model.best_score_:.4f}")
        
        # Print top 5 configurations
        results = self.model.cv_results_
        indices = np.argsort(results['mean_test_score'])[::-1][:5]
        
        print("\nTop 5 configurations:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. Score: {results['mean_test_score'][idx]:.4f} ± {results['std_test_score'][idx]:.4f}")
            print(f"     Params: {results['params'][idx]}")
        
        print("=" * 60)
        print("Training complete!")
        print("=" * 60)

    def predict(self, dataloader):
        """
        Predict on new embeddings.
        
        Args:
            dataloader: DataLoader yielding (data, _, _)
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        features, _ = self._dataloader_to_numpy(dataloader)
        predictions = self.model.predict(features)
        return predictions.reshape(-1, 1)

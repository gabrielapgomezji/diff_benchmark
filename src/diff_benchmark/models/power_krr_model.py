"""
Kernel Ridge Regression using only power values (no embeddings).

This model is designed to assess the predictive power of the power feature alone
in diffusion MRI sphere data, without using the LCOT embeddings.
"""

import numpy as np
import torch
from torch import nn


class PowerOnlyKernelRidgeRegression(nn.Module):
    """
    Kernel Ridge Regression using only power values from diffusion MRI spheres.
    
    This baseline model helps assess how much predictive information is contained
    in the power values alone (before normalization for optimal transport).
    
    Args:
        lmbd: Regularization parameter
        bandwidth: Kernel bandwidth (if None, estimated from data)
        aggregate_method: How to aggregate power across spheres ('mean', 'std', 'both', 'raw')
        chunk_size: Batch size for chunked computation
        device: Compute device
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        lmbd: float = 0.1,
        bandwidth: float = None,
        aggregate_method: str = 'both',
        chunk_size: int = 64,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        n_samples: int = 500,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.lmbd = lmbd
        self.bandwidth_init = bandwidth
        self.aggregate_method = aggregate_method
        self.chunk_size = chunk_size
        self.n_samples = n_samples
        self.seed = seed
        
        # Always set a valid device
        if device is None:
            self.compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.compute_device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Always set a valid dtype
        self.dtype = dtype if dtype is not None else torch.float32
        
        # Training data storage (kept on CPU)
        self.train_features = None
        self.beta = None
        self.kernel_bandwidth = None
        
    def _aggregate_power(self, power):
        """
        Aggregate power values across spheres into a feature vector per subject.
        
        Args:
            power: shape (n_subjects, n_spheres, n_bvals) - power values
        
        Returns:
            features: shape (n_subjects, feature_dim) - aggregated features
        """
        n_subjects, n_spheres, n_bvals = power.shape
        
        if self.aggregate_method == 'mean':
            # Mean power across spheres for each b-value
            features = power.mean(dim=1)  # (n_subjects, n_bvals)
        elif self.aggregate_method == 'std':
            # Standard deviation of power across spheres
            features = power.std(dim=1)  # (n_subjects, n_bvals)
        elif self.aggregate_method == 'both':
            # Both mean and std
            mean_power = power.mean(dim=1)  # (n_subjects, n_bvals)
            std_power = power.std(dim=1)  # (n_subjects, n_bvals)
            features = torch.cat([mean_power, std_power], dim=1)  # (n_subjects, 2*n_bvals)
        elif self.aggregate_method == 'raw':
            # Flatten all power values (memory intensive)
            features = power.reshape(n_subjects, -1)  # (n_subjects, n_spheres * n_bvals)
        else:
            raise ValueError(f"Unknown aggregate_method: {self.aggregate_method}")
        
        return features
    
    def _compute_rbf_kernel_chunked(self, X1, X2, bandwidth):
        """
        Compute RBF kernel matrix between X1 and X2 in chunks.
        
        Args:
            X1: shape (n, d) - on CPU
            X2: shape (m, d) - on CPU
            bandwidth: kernel bandwidth
        
        Returns:
            K: shape (n, m) - kernel matrix (on CPU)
        """
        n, d = X1.shape
        m = X2.shape[0]
        
        # Initialize kernel matrix on CPU
        K = torch.zeros(n, m, dtype=self.dtype)
        
        # Compute in chunks to avoid OOM
        for i in range(0, n, self.chunk_size):
            end_i = min(i + self.chunk_size, n)
            
            # Move chunk to device
            chunk_X1 = X1[i:end_i].to(self.compute_device)  # (chunk_size, d)
            X2_device = X2.to(self.compute_device)  # (m, d)
            
            # Compute squared Euclidean distance for this chunk
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
            X1_norm = (chunk_X1 ** 2).sum(dim=1, keepdim=True)  # (chunk_size, 1)
            X2_norm = (X2_device ** 2).sum(dim=1, keepdim=True).t()  # (1, m)
            dist_sq = X1_norm + X2_norm - 2 * torch.mm(chunk_X1, X2_device.t())
            
            # RBF kernel: exp(-||x - y||^2 / bandwidth)
            K_chunk = torch.exp(-dist_sq / bandwidth)
            
            # Move result back to CPU
            K[i:end_i] = K_chunk.cpu()
            
            # Clean up
            del chunk_X1, X2_device, X1_norm, X2_norm, dist_sq, K_chunk
            if self.compute_device.type == "cuda":
                torch.cuda.empty_cache()
        
        return K
    
    def _estimate_bandwidth(self, features, n_samples=100, n_bootstrap=10):
        """
        Estimate kernel bandwidth using median heuristic with bootstrap sampling.
        
        Args:
            features: shape (n_subjects, feature_dim)
            n_samples: number of samples to use for each bootstrap iteration
            n_bootstrap: number of bootstrap iterations
        
        Returns:
            bandwidth: mean bandwidth across bootstrap samples
        """
        n = features.shape[0]
        n_samples = min(n_samples, n)
        
        bandwidths = []
        
        for bootstrap_iter in range(n_bootstrap):
            # Set seed for reproducibility but vary across bootstrap iterations
            torch.manual_seed(self.seed + bootstrap_iter)
            
            # Sample random subjects
            sample_idx = torch.randperm(n)[:n_samples]
            sample_features = features[sample_idx]
            
            # Compute pairwise squared distances
            # Use CPU for small sample computation
            sample_features_cpu = sample_features.cpu()
            n_s = sample_features_cpu.shape[0]
            
            # Compute all pairwise distances
            diffs = sample_features_cpu.unsqueeze(0) - sample_features_cpu.unsqueeze(1)  # (n_s, n_s, d)
            dist_sq = (diffs ** 2).sum(dim=2)  # (n_s, n_s)

            # Get upper triangular part (excluding diagonal)
            triu_indices = torch.triu_indices(n_s, n_s, offset=1)
            dist_sq_upper = dist_sq[triu_indices[0], triu_indices[1]]
            
            # Median heuristic: bandwidth = median(pairwise_distances^2)
            bandwidth = dist_sq_upper.median().item()
            bandwidths.append(bandwidth)
        
        # Compute statistics
        bandwidths_array = np.array(bandwidths)
        mean_bandwidth = float(np.mean(bandwidths_array))
        std_bandwidth = float(np.std(bandwidths_array))
        
        print(f"  Bootstrap bandwidth estimates: mean={mean_bandwidth:.6f}, std={std_bandwidth:.6f}")
        print(f"  Individual estimates: {[f'{b:.6f}' for b in bandwidths]}")
        
        return mean_bandwidth
    
    def _grid_search_cv(self, features, targets, n_folds=5):
        """
        Perform grid search cross-validation to select optimal lambda.
        
        OPTIMIZED VERSION: Computes the full kernel matrix ONCE, then reuses it for all folds
        and all lambda values by indexing. This is much faster than recomputing kernels.
        
        Steps:
        1. Estimate bandwidth on full dataset (bootstrap)
        2. Compute full kernel matrix K (n × n) once
        3. For each fold:
           - Index K to get K_train (train × train) and K_val (val × train)
           - For each lambda: solve and evaluate (fast, no kernel computation!)
        
        Args:
            features: shape (n_subjects, feature_dim)
            targets: shape (n_subjects,) - binary {-1, 1}
            n_folds: number of cross-validation folds
        
        Returns:
            best_lambda: optimal regularization parameter
            bandwidth: estimated bandwidth used for CV
        """
        print("\nPerforming grid search cross-validation for lambda...")
        print("OPTIMIZED: Computing kernel matrix once and reusing for all folds/lambdas")
        
        # Define lambda grid (log scale from 1e-2 to 1e2)
        lambda_grid = np.logspace(-2, 2, num=20)
        print(f"Lambda grid: {lambda_grid}")
        
        n_subjects = features.shape[0]
        fold_size = n_subjects // n_folds
        
        # Create fold indices
        torch.manual_seed(self.seed)
        indices = torch.randperm(n_subjects)
        
        # Step 1: Estimate bandwidth on full dataset using bootstrap
        print(f"\nEstimating bandwidth on full dataset for CV...")
        cv_bandwidth = self._estimate_bandwidth(
            features, 
            n_samples=min(self.n_samples, n_subjects), 
            n_bootstrap=10
        )
        print(f"CV bandwidth: {cv_bandwidth:.6f}")
        
        # Step 2: Compute full kernel matrix ONCE - this is the expensive operation
        print(f"\nComputing full kernel matrix ({n_subjects} × {n_subjects}) - this is done ONCE...")
        K_full = self._compute_rbf_kernel_chunked(features, features, cv_bandwidth)
        print(f"Kernel matrix computed! Shape: {K_full.shape}")
        
        # Step 3: For each fold, just INDEX into the kernel matrix
        cv_scores = {lmbd: [] for lmbd in lambda_grid}
        
        print(f"\nRunning {n_folds}-fold cross-validation with pre-computed kernel...")
        
        for fold in range(n_folds):
            print(f"\n  Fold {fold + 1}/{n_folds}")
            
            # Create train/val split indices
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_subjects
            
            val_idx = indices[val_start:val_end]
            train_idx = torch.cat([indices[:val_start], indices[val_end:]])
            
            train_targets_fold = targets[train_idx]
            val_targets_fold = targets[val_idx]
            
            print(f"    Train size: {len(train_idx)}, Val size: {len(val_idx)}")
            
            # INDEX into pre-computed kernel matrix (very fast!)
            # K_train: kernel between training samples
            K_train = K_full[train_idx.unsqueeze(1), train_idx.unsqueeze(0)]
            # K_val: kernel between validation and training samples
            K_val = K_full[val_idx.unsqueeze(1), train_idx.unsqueeze(0)]
            
            # Test each lambda (now very fast since kernel is pre-computed)
            print(f"    Testing {len(lambda_grid)} lambda values (fast - no kernel computation)...")
            for lmbd in lambda_grid:
                # Solve ridge regression: (K_train + λI)β = y_train
                n_train = len(train_idx)
                I = torch.eye(n_train, dtype=self.dtype)
                K_reg = K_train + lmbd * I
                
                # Solve using Cholesky decomposition
                try:
                    L = torch.linalg.cholesky(K_reg)
                    beta = torch.cholesky_solve(train_targets_fold.unsqueeze(-1), L).squeeze(-1)
                except RuntimeError:
                    beta = torch.linalg.lstsq(K_reg, train_targets_fold).solution
                
                # Predict on validation set: f(x_val) = K_val @ β
                val_scores = K_val @ beta
                val_pred = (val_scores > 0).float()
                
                # Compute accuracy
                val_accuracy = (val_pred == ((val_targets_fold + 1) / 2)).float().mean().item()
                cv_scores[lmbd].append(val_accuracy)
                
                # Clean up
                del I, K_reg, beta, val_scores, val_pred
            
            # Clean up fold-specific data
            del K_train, K_val
            
            print(f"    Fold {fold + 1} complete!")
        
        # Clean up full kernel matrix
        del K_full
        if self.compute_device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Compute mean CV score for each lambda
        mean_scores = {lmbd: np.mean(scores) for lmbd, scores in cv_scores.items()}
        std_scores = {lmbd: np.std(scores) for lmbd, scores in cv_scores.items()}
        
        # Find best lambda
        best_lambda = max(mean_scores, key=mean_scores.get)
        best_score = mean_scores[best_lambda]
        
        print("\n" + "=" * 60)
        print("Cross-validation results:")
        print("=" * 60)
        for lmbd in lambda_grid:
            marker = " <-- BEST" if lmbd == best_lambda else ""
            print(f"  lambda={lmbd:.6f}: {mean_scores[lmbd]:.4f} ± {std_scores[lmbd]:.4f}{marker}")
        
        print(f"\nBest lambda: {best_lambda:.6f} (CV accuracy: {best_score:.4f})")
        print("=" * 60)
        
        return float(best_lambda), cv_bandwidth
    
    def fit(self, dataloader):
        """
        Fit the kernel ridge regression model using only power values.
        Uses grid search CV to select optimal lambda and bootstrap for bandwidth.
        
        Args:
            dataloader: DataLoader yielding (data, targets, _) where
                        data is dict with 'power' key
        """
        print("=" * 60)
        print("PowerOnlyKernelRidgeRegression - Training")
        print("=" * 60)
        print(f"Aggregation method: {self.aggregate_method}")
        print(f"Initial regularization lambda: {self.lmbd}")
        
        print("\nLoading training data in chunks...")
        
        # Collect power features and targets
        feature_chunks = []
        target_chunks = []
        
        for i, (data, targets_batch, _) in enumerate(dataloader):
            print(f"  Processing batch {i+1}/{len(dataloader)}")
            
            # Extract power values
            power = data["power"].to(self.dtype)
            
            # Remove batch dimension if present
            if power.dim() == 4:
                power = power.squeeze(1)
            
            # Aggregate power into features (CPU)
            features = self._aggregate_power(power)
            
            # Store aggregated results
            feature_chunks.append(features)
            target_chunks.append(targets_batch.to(self.dtype))
            
            # Clear the large raw power from memory
            del power, data
        
        print("Concatenating features...")
        # Concatenate all features
        features = torch.cat(feature_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)
        
        # Clear chunks
        del feature_chunks, target_chunks
        
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Convert targets to {-1, 1}
        targets = targets * 2 - 1
        
        # Grid search cross-validation for lambda
        # OPTIMIZED: Computes kernel once, estimates bandwidth once, reuses for all folds/lambdas
        if self.bandwidth_init is None:
            optimal_lambda, cv_bandwidth = self._grid_search_cv(features, targets, n_folds=5)
            print(f"\nUsing optimal lambda from CV: {optimal_lambda:.6f}")
            print(f"Using bandwidth from CV: {cv_bandwidth:.6f}")
            # Use the same bandwidth from CV for final model (already estimated on full dataset)
            self.kernel_bandwidth = cv_bandwidth
        else:
            # If bandwidth is provided, still do CV for lambda but use provided bandwidth
            print(f"\nUsing provided bandwidth: {self.bandwidth_init:.6f}")
            self.kernel_bandwidth = self.bandwidth_init
            # Simplified CV with provided bandwidth
            print(f"\nEstimating bandwidth on full dataset for CV...")
            optimal_lambda, _ = self._grid_search_cv(features, targets, n_folds=5)
            print(f"\nUsing optimal lambda from CV: {optimal_lambda:.6f}")

        
        # Store for prediction (on CPU)
        self.train_features = features
        
        # Train final model with optimal lambda on full training set
        print("\nTraining final model on full training set...")
        n_subjects = features.shape[0]
        
        with torch.no_grad():
            K = self._compute_rbf_kernel_chunked(features, features, self.kernel_bandwidth)
            print(f"Kernel matrix shape: {K.shape}")
            
            # Solve ridge regression with optimal lambda
            print(f"Solving ridge regression with lambda={optimal_lambda:.6f}...")
            
            # Move to GPU for solving
            K_gpu = K.to(self.compute_device)
            targets_gpu = targets.to(self.compute_device)
            I = torch.eye(n_subjects, device=self.compute_device, dtype=self.dtype)
            K_reg = K_gpu + optimal_lambda * I
            
            # Use Cholesky decomposition for numerical stability
            try:
                L = torch.linalg.cholesky(K_reg)
                beta_gpu = torch.cholesky_solve(targets_gpu.unsqueeze(1), L).squeeze(1)
            except RuntimeError:
                print("Warning: Cholesky decomposition failed, using standard solve")
                beta_gpu = torch.linalg.solve(K_reg, targets_gpu)
            
            # Move beta back to CPU
            self.beta = beta_gpu.cpu()
            
            print(f"Beta coefficients shape: {self.beta.shape}")
            
            # Compute training accuracy
            train_pred = (K_gpu @ beta_gpu) > 0
            train_acc = (train_pred.float() == ((targets_gpu + 1) / 2)).float().mean()
            print(f"\nFinal training accuracy: {train_acc.item():.4f}")
            print(f"Final lambda used: {optimal_lambda:.6f}")
            print("=" * 60)
            
            # Clean up GPU memory
            del K_gpu, targets_gpu, I, K_reg, beta_gpu, train_pred
            if self.compute_device.type == "cuda":
                torch.cuda.empty_cache()
    
    def predict(self, dataloader):
        """
        Predict on new data using only power values.
        
        Args:
            dataloader: DataLoader yielding (data, _, _) where data has 'power'
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if self.train_features is None or self.beta is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        print("\nPredicting on test data...")
        all_predictions = []
        
        with torch.no_grad():
            for i, (data, _, _) in enumerate(dataloader):
                print(f"  Batch {i+1}/{len(dataloader)}")
                
                # Extract and aggregate power features
                power = data["power"].to(self.dtype)
                
                # Remove batch dimension if present
                if power.dim() == 4:
                    power = power.squeeze(1)
                
                # Aggregate power (on CPU)
                features = self._aggregate_power(power)
                
                # Compute kernel with training data
                K_test = self._compute_rbf_kernel_chunked(
                    features, self.train_features, self.kernel_bandwidth
                )
                
                # Move to GPU for prediction
                K_test_gpu = K_test.to(self.compute_device)
                beta_gpu = self.beta.to(self.compute_device)
                
                # Predict
                scores = K_test_gpu @ beta_gpu
                predictions = (scores > 0).float().cpu()
                
                all_predictions.append(predictions)
                
                # Clean up GPU memory
                del K_test_gpu, beta_gpu, scores
                if self.compute_device.type == "cuda":
                    torch.cuda.empty_cache()
        
        print("Prediction complete!")
        return torch.cat(all_predictions, dim=0).cpu()

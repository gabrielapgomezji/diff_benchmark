import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import math


class FullEmbeddingsKernelRidgeRegression(nn.Module):
    """
    Memory-efficient Kernel Ridge Regression for sphere embeddings.
    
    Key innovations:
    2. **Additive distance accumulation**: K = exp(-sum(distances) / bandwidth)
       - Accumulates distances in float32, applies exp once at end
       - Faster and more stable than multiplicative accumulation
       - Fewer exp() calls (1 vs n_spheres * n_bvals per tile)
       - Uses SUM of distances (not mean) for proper multi-view kernel interpretation
    3. **Shared kernel computation**: Single _compute_kernel_tile() helper
       - Used by both matrix formation and matvec operations
       - Avoids code duplication, easier to maintain
    4. **Tiled computation** over samples AND spheres
       - Sample blocks: (n1_blk, n2_blk) kernel tiles
       - Sphere blocks: process spheres in batches on GPU
       - Only sphere batches moved to GPU, not full sample blocks
    5. **Maximum GPU tensor**: (n1_blk, n2_blk, sphere_batch_size, d) during distance computation
       - No full 4D broadcasted tensors across all spheres!
    
    Kernel formulation:
    - For each (sphere, bval) pair: compute mean distance over embedding dim d
      * Circular distance (default): min(|x-y|, 1-|x-y|) wraps around at 1
      * L2 norm (if use_l2_norm=True): standard Euclidean squared distance (x-y)²
    - Sum these distances across all n_spheres * n_bvals pairs → total_distance
    - Kernel: K(x, y) = exp(-total_distance / bandwidth)
    - Bandwidth is automatically scaled by n_spheres * n_bvals during estimation
    
    Memory requirements:
    - CPU: O(n_train * n_spheres * n_bvals * d) for training embeddings
    - CPU: O(n_train) for beta (matrix-free) or O(n_train²) for K (direct)
    - GPU: O(n1_blk * n2_blk * sphere_batch_size * d) peak during computation
    
    Performance tuning:
    - sphere_batch_size: Number of spheres processed together (default: 64)
      * Larger = faster (more vectorization) but more GPU memory
      * Smaller = slower but less GPU memory
      * Recommended: 32-128 depending on GPU
      * GPU memory per tile: ~(n1_blk * n2_blk * sphere_batch_size * d * 4) bytes
    - sample_batch_size: Size of kernel tiles (n1_blk, n2_blk) (default: 50)
      * Larger = faster (fewer CPU↔GPU transfers) but more GPU memory
      * Smaller = slower but less GPU memory
      * Recommended: 25-100 depending on GPU and n_spheres
    
    Lambda (regularization) tuning:
    - Kernel values are typically in [0, 1] with diagonal ≈ 1
    - Recommended lambda range: [kernel_std * 1e-3, kernel_std * 10]
    - The fit() method prints kernel diagnostics to guide lambda selection
    
    Args:
        lmbd: Regularization parameter (default: 0.1)
        use_power_weighting: Whether to weight sphere distances by power values
        device: Deprecated, auto-detected
        dtype: Data type for tensors (default: torch.float32)
        sphere_batch_size: Number of spheres to process together (default: 64)
        sample_batch_size: Size of sample tiles for kernel computation (default: 50)
        use_l2_norm: Whether to use L2 norm instead of circular distance (default: False)
    """

    data_type = "lcot_embed"

    def __init__(
        self,
        lmbd: float = 1.,
        use_power_weighting: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        sphere_batch_size: int = 64,  # Number of spheres to process together
        sample_batch_size: int = 50,  # Number of samples to tile over (n1_blk, n2_blk)
        n_bootstrap: int = 10,  # Number of bootstrap iterations for bandwidth
        seed: int = 42,  # Random seed for reproducibility
        use_l2_norm: bool = False,  # Use L2 norm instead of circular distance
        **kwargs,
    ):
        super().__init__()
        self.lmbd = lmbd
        self.use_power_weighting = use_power_weighting
        self.sphere_batch_size = sphere_batch_size
        self.sample_batch_size = sample_batch_size
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.use_l2_norm = use_l2_norm
        
        # Always set a valid device
        self.compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Always set a valid dtype
        self.dtype = torch.float32

        # Training data storage (kept on CPU)
        self.train_embeddings = None
        self.train_power = None
        self.beta = None
        self.kernel_bandwidth = None 
    
    def _compute_kernel_tile(self, emb1_block, emb2_block, bandwidth=None):
        """
        Helper method to compute kernel values for a tile of samples.
        
        OPTIMIZED: Uses one bandwidth per bvalue (3 bandwidths total). Each bvalue's
        distances are normalized by its own bandwidth, allowing different spheres
        to have different characteristic scales.
        
        This method is shared by both _compute_kernel_matrix_tiled and _kernel_matvec
        to avoid code duplication. It accumulates distances additively (faster and more
        stable than multiplicative accumulation), then applies exp once at the end.
        
        Distance metric: Uses either circular distance (default) or L2 norm based on
        self.use_l2_norm flag. Circular distance wraps around at 1, while L2 is standard
        Euclidean distance.
        
        Expects embeddings to be on CPU - will move sphere batches to GPU as needed.
        
        Args:
            emb1_block: shape (n1_blk, n_spheres, n_bvals, d) - first embeddings block (on CPU)
            emb2_block: shape (n2_blk, n_spheres, n_bvals, d) - second embeddings block (on CPU)
            bandwidth: kernel bandwidths (array of length n_bvals or scalar for backward compatibility)
        
        Returns:
            K_block: shape (n1_blk, n2_blk) - kernel values (on GPU, float64)
        """
        bandwidth = bandwidth if bandwidth is not None else self.kernel_bandwidth
        n1_blk, n_spheres, n_bvals, d = emb1_block.shape
        n2_blk = emb2_block.shape[0]
        
        # Convert bandwidth to array if scalar (backward compatibility)
        if np.isscalar(bandwidth):
            bandwidth_array = np.array([bandwidth] * n_bvals)
        else:
            bandwidth_array = np.array(bandwidth)
        
        # Additive accumulation with float32 for speed and memory
        # Now we'll accumulate normalized distances: sum(dist_bval / bw_bval)
        sum_normalized_distances = torch.zeros(n1_blk, n2_blk, device=self.compute_device, dtype=torch.float32)

        # Tile over spheres and bvals - move only sphere batches to GPU
        for sphere_start in range(0, n_spheres, self.sphere_batch_size):
            sphere_end = min(sphere_start + self.sphere_batch_size, n_spheres)
            
            for bval_idx in range(n_bvals):
                # Extract embeddings for this sphere block and bval - NOW move to GPU
                # Shape: (n1_blk, n_spheres_batch, d)
                emb1_sb = emb1_block[:, sphere_start:sphere_end, bval_idx, :].to(self.compute_device)
                emb2_sb = emb2_block[:, sphere_start:sphere_end, bval_idx, :].to(self.compute_device)

                # Vectorized computation across all spheres in batch
                # (n1_blk, 1, n_spheres_batch, d) - (1, n2_blk, n_spheres_batch, d)
                # = (n1_blk, n2_blk, n_spheres_batch, d)
                diff = emb1_sb.unsqueeze(1) - emb2_sb.unsqueeze(0)

                if self.use_l2_norm:
                    # L2 norm: mean squared Euclidean distance
                    # (n1_blk, n2_blk, n_spheres_batch)
                    msd = torch.mean(diff ** 2, dim=-1)
                else:
                    # Circular distance: wrap around at 1
                    # Mean absolute circular distance over dimension d: (n1_blk, n2_blk, n_spheres_batch)
                    diff_abs = torch.abs(diff)
                    msd = torch.mean(torch.minimum(diff_abs, 1 - diff_abs), dim=-1)

                # Sum over spheres and NORMALIZE by this bvalue's bandwidth
                # (n1_blk, n2_blk, n_spheres_batch) -> (n1_blk, n2_blk)
                dist_sum = torch.sum(msd, dim=-1)
                sum_normalized_distances += dist_sum / bandwidth_array[bval_idx]
                
                del msd, emb1_sb, emb2_sb, diff, dist_sum

        # Apply exp once at the end
        # K = exp(-sum(dist_bval / bw_bval))
        # This allows each bvalue to contribute with its own characteristic scale
        K_block = torch.exp(-sum_normalized_distances)
        del sum_normalized_distances
        
        return K_block
        
    def _compute_kernel_matrix(self, emb1, emb2, bandwidth=None):
        """
        Compute RBF kernel matrix using additive accumulation with tiling.
        
        This avoids the memory-killer 4D broadcasted diff tensor by:
        1. Tiling over samples (n1_blk × n2_blk blocks)
        2. Using shared _compute_kernel_tile helper with additive accumulation
        
        Memory-efficient because:
        - Max GPU tensor: (n1_blk, n2_blk, n_spheres_batch, d) during distance computation
        - No large 4D tensors ever materialized
        - Uses float32 for distance accumulation (faster, less memory)
        
        Args:
            emb1: shape (n1, n_spheres, n_bvals, d) - on CPU
            emb2: shape (n2, n_spheres, n_bvals, d) - on CPU
            bandwidth: kernel bandwidth (if None, use self.kernel_bandwidth)
        
        Returns:
            K: shape (n1, n2) - kernel matrix (on CPU)
        """
        bandwidth = bandwidth if bandwidth is not None else self.kernel_bandwidth
        n1, n_spheres, n_bvals, d = emb1.shape
        n2 = emb2.shape[0]
        
        # Result kernel matrix on CPU
        K_full = torch.zeros(n1, n2, dtype=self.dtype)
        
        # Tile over samples (n1 and n2 dimensions)
        for i1 in range(0, n1, self.sample_batch_size):
            print(f"    Progress: sample block {i1+1}-{min(i1 + self.sample_batch_size, n1)} / {n1}")
            end_i1 = min(i1 + self.sample_batch_size, n1)
            
            for i2 in range(0, n2, self.sample_batch_size):
                end_i2 = min(i2 + self.sample_batch_size, n2)
                
                # Extract sample blocks (keep on CPU - _compute_kernel_tile will move to GPU)
                emb1_block = emb1[i1:end_i1]
                emb2_block = emb2[i2:end_i2]
                
                # Compute kernel tile using shared helper
                K_block = self._compute_kernel_tile(
                    emb1_block, emb2_block, bandwidth
                )
                
                # Store result for this sample block
                K_full[i1:end_i1, i2:end_i2] = K_block.cpu().to(self.dtype)
                del K_block, emb1_block, emb2_block
                
                # Clean up GPU cache
                if self.compute_device.type == "cuda":
                    torch.cuda.empty_cache()
        
        return K_full

    def _estimate_bandwidth_bootstrap(self, embeddings, n_bootstrap=None):
        """
        Bootstrap median heuristic with ONE bandwidth PER bvalue (3 total).
        
        OPTIMIZED: Samples self.sphere_batch_size spheres per bvalue, estimates bandwidth
        for each bvalue independently. This eliminates the inner sphere loop and allows
        different spheres to have different bandwidths based on their b-value characteristics.

        Vectorized over a single sample block of size m = min(self.sample_batch_size, n).
        Computes the full (m x m) pairwise distance matrix per bootstrap iteration,
        sampling spheres per bvalue to control memory.
        
        Distance metric: Uses either circular distance (default) or L2 norm based on
        self.use_l2_norm flag.

        Args:
            embeddings: (n, n_spheres, n_bvals, d) on CPU
            n_bootstrap: int or None, number of bootstrap iterations (default: self.n_bootstrap)

        Returns:
            np.ndarray: Array of 3 bandwidths [bw_bval0, bw_bval1, bw_bval2]
        """
        n = embeddings.shape[0]
        m = int(min(self.sample_batch_size, n))
        if m < 2:
            return np.array([1e-3, 1e-3, 1e-3])

        n_bootstrap = int(n_bootstrap or self.n_bootstrap)
        n_spheres = embeddings.shape[1]
        n_bvals   = embeddings.shape[2]
        
        # Number of spheres to sample per bvalue (limited by sphere_batch_size)
        n_spheres_sample = min(self.sphere_batch_size, n_spheres)

        # Store bandwidths per bvalue: bandwidths_per_bval[bval_idx] = list of bootstrap estimates
        bandwidths_per_bval = [[] for _ in range(n_bvals)]
        
        print(f"  Estimating {n_bvals} bandwidths (one per bvalue) with {n_bootstrap} bootstrap iterations...")
        print(f"    Sampling {n_spheres_sample} spheres per bvalue per iteration")
        
        for b in range(n_bootstrap):
            # different seed per bootstrap for reproducibility
            torch.manual_seed(self.seed + b)
            
            # Sample subjects
            idx = torch.randperm(n)[:m]
            samp = embeddings[idx]  # (m, n_spheres, n_bvals, d) on CPU
            
            # Process each bvalue independently
            for bval_idx in range(n_bvals):
                # Sample spheres for this bvalue
                sphere_idx = torch.randperm(n_spheres, generator=torch.Generator().manual_seed(self.seed + b + bval_idx))[:n_spheres_sample]
                
                # Extract sampled spheres for this bvalue: (m, n_spheres_sample, d)
                samp_bval = samp[:, sphere_idx, bval_idx, :].to(self.compute_device)
                
                # Compute pairwise distances across all sampled spheres
                # shapes: (m,1,n_spheres_sample,d) - (1,m,n_spheres_sample,d) -> (m,m,n_spheres_sample,d)
                diff = samp_bval.unsqueeze(1) - samp_bval.unsqueeze(0)
                
                if self.use_l2_norm:
                    # L2 norm: mean squared Euclidean distance
                    msd = (diff ** 2).mean(dim=-1)
                else:
                    # Circular distance: wrap around at 1
                    diff_abs = torch.abs(diff)
                    msd = torch.minimum(diff_abs, 1 - diff_abs).mean(dim=-1)
                
                # Sum over spheres to get total distance per pair: (m, m)
                pair_sum = msd.sum(dim=-1)
                
                # take strict upper triangle (i < j) once, then median
                iu = torch.triu_indices(m, m, offset=1, device=pair_sum.device)
                upp = pair_sum[iu[0], iu[1]].to("cpu")
                
                median_dist = torch.median(upp).item() if upp.numel() > 0 else 0.0
                bw = 2.0 * median_dist if median_dist > 0 else 1e-3
                bandwidths_per_bval[bval_idx].append(bw)

                del samp_bval, diff, msd, pair_sum, upp

                if b < 3 or b == n_bootstrap - 1:
                    print(f"    Bootstrap {b+1}/{n_bootstrap}, bval {bval_idx}: bandwidth = {bw}")

            if self.compute_device.type == "cuda":
                torch.cuda.empty_cache()

        # Compute mean bandwidth for each bvalue
        mean_bandwidths = np.array([np.mean(bw_list) for bw_list in bandwidths_per_bval])
        std_bandwidths = np.array([np.std(bw_list) for bw_list in bandwidths_per_bval])

        # NEW: scale by spheres ratio so distances are on the same scale
        scale = embeddings.shape[1] / n_spheres_sample  # total_spheres / sampled_spheres
        mean_bandwidths = mean_bandwidths * scale

        print(f"  Final bandwidths per bvalue (mean±std):")
        for i, (m, s) in enumerate(zip(mean_bandwidths, std_bandwidths)):
            print(f"    bval {i}: {m} ± {s}")
        return mean_bandwidths

    def _report_kernel_stats(self, K):
        with torch.no_grad():
            n = K.shape[0]
            diag = torch.diag(K)
            off = K[~torch.eye(n, dtype=torch.bool)]
            print(f"diag: mean={diag.mean():.6f}, min={diag.min():.6f}, max={diag.max():.6f}")
            print(f"offdiag: mean={off.mean():.6f}, std={off.std():.6f}, "
                f"min={off.min():.6f}, max={off.max():.6f}")
            print(f"overall mean={K.mean():.6f}, "
                f"expected 1/n ≈ {1.0/n:.6f} (if identity-like)")
    
    def _grid_search_cv(self, embeddings, targets, n_folds=5):
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
            embeddings: shape (n_subjects, n_spheres, n_bvals, d)
            targets: shape (n_subjects,) - binary {-1, 1}
            n_folds: number of cross-validation folds
        
        Returns:
            best_lambda: optimal regularization parameter
            bandwidth: estimated bandwidth used for CV
            K_full: full kernel matrix (n × n) for reuse in training
        """
        print("\nPerforming grid search cross-validation for lambda...")
        print("OPTIMIZED: Computing kernel matrix once and reusing for all folds/lambdas")

        # Define lambda grid (log scale from 1e-2 to 1e2)
        lambda_grid = np.logspace(-4, 0, num=20)
        print(f"Lambda grid: {lambda_grid}")
        
        n_subjects = embeddings.shape[0]
        fold_size = n_subjects // n_folds
        
        # Create fold indices
        torch.manual_seed(self.seed)
        indices = torch.randperm(n_subjects)
        
        # Step 1: Estimate bandwidth on full dataset using bootstrap
        print(f"\nEstimating bandwidth on full dataset for CV...")
        cv_bandwidth = self._estimate_bandwidth_bootstrap(
            embeddings, 
            n_bootstrap=self.n_bootstrap
        )
        print(f"CV bandwidths (per bvalue): {cv_bandwidth}")
        
        # Step 2: Compute full kernel matrix ONCE - this is the expensive operation
        print(f"\nComputing full kernel matrix ({n_subjects} × {n_subjects}) - this is done ONCE...")
        K_full = self._compute_kernel_matrix(embeddings, embeddings, cv_bandwidth)
        self._report_kernel_stats(K_full)
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
            print(f"  lambda={lmbd}: {mean_scores[lmbd]:.4f} ± {std_scores[lmbd]:.4f}{marker}")
        
        print(f"\nBest lambda: {best_lambda} (CV accuracy: {best_score:.4f})")
        print("=" * 60)
        
        # Return K_full for reuse in training (IMPORTANT: don't delete it!)
        return float(best_lambda), cv_bandwidth, K_full

    def fit(self, dataloader):
        """
        Fit the kernel ridge regression model.
        Uses grid search CV to select optimal lambda and bootstrap for bandwidth.
        
        Args:
            dataloader: DataLoader yielding (data, targets, _) where
                        data is dict with 'embeddings' and 'power'
        """
        print("=" * 60)
        print("FullEmbeddingsKernelRidgeRegression - Training")
        print("=" * 60)
        print(f"Initial regularization lambda: {self.lmbd}")
        
        print("\nLoading training data in batches...")
        
        # Collect embeddings, power, and targets in chunks
        embeddings_chunks = []
        power_chunks = []
        target_chunks = []
        
        for i, (data, targets_batch, _) in enumerate(dataloader):
            print(f"  Processing batch {i+1}/{len(dataloader)}")
            
            # Process this batch
            embeddings = data["embeddings"].to(self.dtype)
            power = data["power"].to(self.dtype)
            
            # Remove batch dimension if present
            if embeddings.dim() == 5:
                embeddings = embeddings.squeeze(1)
            if power.dim() == 4:
                power = power.squeeze(1)
            
            # Store on CPU
            embeddings_chunks.append(embeddings.cpu())
            power_chunks.append(power.cpu())
            target_chunks.append(targets_batch.to(self.dtype))
            
            # Clear from memory
            del embeddings, power, data
        
        print("Concatenating data...")
        # Concatenate all batches
        self.train_embeddings = torch.cat(embeddings_chunks, dim=0)  # (n_subjects, n_spheres, n_bvals, d)
        self.train_power = torch.cat(power_chunks, dim=0)  # (n_subjects, n_spheres, n_bvals)
        targets = torch.cat(target_chunks, dim=0)

        # Clear chunks
        del embeddings_chunks, power_chunks, target_chunks
        
        print(f"Training embeddings shape: {self.train_embeddings.shape}")
        print(f"Training power shape: {self.train_power.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Convert targets to {-1, 1}
        targets = targets * 2 - 1
        
        # Grid search cross-validation for lambda
        # OPTIMIZED: Computes kernel once, estimates bandwidth once, reuses for CV, training, and accuracy
        print("\nPerforming CV and computing kernel matrix (done once)...")
        optimal_lambda, cv_bandwidth, K_train_full = self._grid_search_cv(self.train_embeddings, targets, n_folds=5)
        print(f"\nUsing optimal lambda from CV: {optimal_lambda}")
        print(f"Using bandwidths from CV (per bvalue): {cv_bandwidth}")
        
        # Use the same bandwidth from CV for final model (already estimated on full dataset)
        self.kernel_bandwidth = cv_bandwidth
        
        # Solve ridge regression with optimal lambda using PRE-COMPUTED kernel
        print(f"\nSolving ridge regression with optimal lambda (reusing kernel matrix)...")
        n_subjects = self.train_embeddings.shape[0]
        
        with torch.no_grad():
            # REUSE the kernel matrix from CV - no recomputation!
            print(f"Reusing kernel matrix from CV: shape {K_train_full.shape}")

            # Add regularization: K + λI
            print(f"Kernel matrix (mean abs value): {torch.abs(K_train_full).mean()}")
            K_reg = K_train_full + optimal_lambda * torch.eye(n_subjects, dtype=self.dtype)

            # Solve via Cholesky decomposition (stable for positive definite matrices)
            print("Solving with Cholesky decomposition...")
            try:
                L = torch.linalg.cholesky(K_reg)
                # Solve L L^T β = y in two steps
                # targets is 1D, need to make it 2D for solve_triangular
                y_2d = targets.unsqueeze(-1)  # (n, 1)
                z = torch.linalg.solve_triangular(L, y_2d, upper=False)
                beta_2d = torch.linalg.solve_triangular(L.T, z, upper=True)
                self.beta = beta_2d.squeeze(-1)  # Back to 1D
                del L, z, y_2d, beta_2d
            except Exception as e:
                print(f"Cholesky failed: {e}, falling back to least squares")
                self.beta = torch.linalg.lstsq(K_reg, targets).solution
            
            del K_reg
            
            print(f"Beta coefficients shape: {self.beta.shape}")
            
            # Compute training accuracy using SAME kernel matrix (third reuse!)
            print("\nComputing final training accuracy (reusing kernel matrix)...")
            train_scores = K_train_full @ self.beta
            train_pred = (train_scores > 0).float()
            train_acc = (train_pred == ((targets + 1) / 2)).float().mean()
            del train_scores, train_pred
            
            # NOW we can delete the kernel matrix
            del K_train_full
            
            print(f"\nFinal training accuracy: {train_acc.item():.4f}")
            print(f"Final lambda used: {optimal_lambda}")
            print("=" * 60)
            print("Training complete!")
            
            # Clean up GPU cache
            if self.compute_device.type == "cuda":
                torch.cuda.empty_cache()
    
    def predict(self, dataloader):
        """
        Predict on new data.
        
        Args:
            dataloader: DataLoader yielding (data, _, _)
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if self.train_embeddings is None or self.beta is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        all_predictions = []
        
        with torch.no_grad():
            for i, (data, _, _) in enumerate(dataloader):
                print(f"  Batch {i+1}/{len(dataloader)}")
                
                # Load test embeddings and power - keep on CPU
                embeddings = data["embeddings"].to(self.dtype)
                power = data["power"].to(self.dtype)
                
                # Remove batch dimension if present
                if embeddings.dim() == 5:
                    embeddings = embeddings.squeeze(1)
                if power.dim() == 4:
                    power = power.squeeze(1)
        
                # Direct approach: form K_test matrix and multiply
                # Memory: O(n_test × n_train)
                K_test = self._compute_kernel_matrix(
                    embeddings,
                    self.train_embeddings,
                )
                scores = K_test @ self.beta
                del K_test
                
                predictions = (scores > 0).float()
                all_predictions.append(predictions)
                
                # Clean up
                del embeddings, power, scores
        
        print("Prediction complete!")
        return torch.cat(all_predictions, dim=0)

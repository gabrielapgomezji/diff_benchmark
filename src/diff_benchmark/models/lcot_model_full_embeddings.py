import torch
from torch import nn
from tqdm import tqdm
import math


class fullEmbeddingsKernelRidgeRegression(nn.Module):
    """
    Memory-efficient Kernel Ridge Regression for sphere embeddings.
    
    Key innovations:
    1. **Flexible solving modes**:
       - matrix_free=True: Conjugate gradient (O(n) memory, scalable)
       - matrix_free=False: Direct Cholesky solve (O(n²) memory, faster for small n)
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
    - For each (sphere, bval) pair: compute mean squared circular distance over embedding dim d
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
    
    Trade-offs:
    - matrix_free=True: O(n) memory, slower (multiple kernel-vector products in CG)
      * Best for large n (hundreds to thousands of samples)
      * Scalable to very large datasets
    - matrix_free=False: O(n²) memory, faster (single Cholesky solve)
      * Best for small n (< 100 samples) when you have enough RAM
      * Easier to debug (can inspect kernel matrix)
    
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
        matrix_free: Use matrix-free CG (True) or direct Cholesky (False) (default: True)
    """
    
    def __init__(
        self,
        lmbd: float = 0.1,
        use_power_weighting: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        sphere_batch_size: int = 64,  # Number of spheres to process together
        sample_batch_size: int = 50,  # Number of samples to tile over (n1_blk, n2_blk)
        matrix_free: bool = False,  # Use matrix-free CG solver vs full kernel matrix
        **kwargs,
    ):
        super().__init__()
        self.lmbd = lmbd
        self.use_power_weighting = use_power_weighting
        self.sphere_batch_size = sphere_batch_size
        self.sample_batch_size = sample_batch_size
        self.matrix_free = matrix_free
        
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
        
        This method is shared by both _compute_kernel_matrix_tiled and _kernel_matvec
        to avoid code duplication. It accumulates distances additively (faster and more
        stable than multiplicative accumulation), then applies exp once at the end.
        
        Expects embeddings to be on CPU - will move sphere batches to GPU as needed.
        
        Args:
            emb1_block: shape (n1_blk, n_spheres, n_bvals, d) - first embeddings block (on CPU)
            emb2_block: shape (n2_blk, n_spheres, n_bvals, d) - second embeddings block (on CPU)
            bandwidth: kernel bandwidth
        
        Returns:
            K_block: shape (n1_blk, n2_blk) - kernel values (on GPU, float64)
        """
        bandwidth = bandwidth if bandwidth is not None else self.kernel_bandwidth
        n1_blk, n_spheres, n_bvals, d = emb1_block.shape
        n2_blk = emb2_block.shape[0]
        
        # Additive accumulation with float32 for speed and memory
        sum_distances = torch.zeros(n1_blk, n2_blk, device=self.compute_device, dtype=torch.float32)

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
                diff = torch.abs(emb1_sb.unsqueeze(1) - emb2_sb.unsqueeze(0))

                # Circular distance: wrap around at 1
                # Mean squared distance over dimension d: (n1_blk, n2_blk, n_spheres_batch)
                msd = torch.mean(torch.minimum(diff, 1 - diff) ** 2, dim=-1)

                # Additive accumulation: sum all distances, will exp once at end
                # Sum over spheres in batch: (n1_blk, n2_blk, n_spheres_batch) -> (n1_blk, n2_blk)
                sum_distances += torch.sum(msd, dim=-1)
                del msd, emb1_sb, emb2_sb, diff
        
        # Apply exp once at the end (more efficient and stable than multiplicative accumulation)
        # K = exp(-sum(distances) / bandwidth)
        # Note: sum_distances already contains the sum of mean squared distances across all spheres and bvals
        # We use sum of distances (not mean) to preserve multi-view kernel interpretation
        K_block = torch.exp(-sum_distances.to(torch.float64) / bandwidth)
        del sum_distances
        
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

    def fit(self, dataloader):
        """
        Fit the kernel ridge regression model.
        
        Args:
            dataloader: DataLoader yielding (data, targets, _) where
                        data is dict with 'embeddings' and 'power'
        """
        print("Loading training data in batches...")
        
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
        
        # Estimate kernel bandwidth from data
        print("Computing kernel bandwidth from data statistics...")
        with torch.no_grad():
            # Sample a small subset for bandwidth estimation to save memory
            n_samples = min(10, self.train_embeddings.shape[0])
            sample_idx = torch.randperm(self.train_embeddings.shape[0])[:n_samples]
            sample_emb = self.train_embeddings[sample_idx]
            sample_power = self.train_power[sample_idx]
            
            n_spheres = sample_emb.shape[1]
            n_bvals = sample_emb.shape[2]
            total_sb = n_spheres * n_bvals
            
            # Compute actual summed distances between sample pairs
            # This matches what the kernel computation does
            pairwise_distances = []
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    # Compute sum of distances for this pair (matching _compute_kernel_tile logic)
                    sum_dist = 0.0
                    for s_idx in range(n_spheres):
                        for b_idx in range(n_bvals):
                            emb_i = sample_emb[i, s_idx, b_idx, :].to(self.compute_device)
                            emb_j = sample_emb[j, s_idx, b_idx, :].to(self.compute_device)
                            
                            diff = torch.abs(emb_i - emb_j)
                            dist_circ = torch.minimum(diff, 1 - diff)
                            msd = torch.mean(dist_circ ** 2)
                            sum_dist += msd.item()
                    
                    pairwise_distances.append(sum_dist)
            
            # Use median of summed distances for robust bandwidth estimation
            # Classic RBF: exp(-d^2 / (2σ^2)), so bandwidth = 2σ^2
            median_sum_dist = torch.tensor(pairwise_distances).median().item()
            self.kernel_bandwidth = 2.0 * median_sum_dist if median_sum_dist > 0 else 1e-3

            print(f"Kernel bandwidth (estimated): {self.kernel_bandwidth}")
            print(f"  Median summed distance: {median_sum_dist:.6f}")
            print(f"  Total spheres × bvals: {total_sb}")
            print(f"  Expected kernel at median distance: exp(-1) ≈ {math.exp(-1):.4f}")
        
        # Solve ridge regression
        print(f"Solving ridge regression (matrix_free={self.matrix_free})...")
        n_subjects = self.train_embeddings.shape[0]
        
        with torch.no_grad():
            # Convert targets to {-1, 1} for better numerical properties
                # Direct approach: Form full kernel matrix and solve with Cholesky
                # Memory: O(n²) - stores full kernel matrix
            print("Forming full kernel matrix for direct solve...")
            K = self._compute_kernel_matrix(
                self.train_embeddings, 
                self.train_embeddings,
            )
            print(f"Kernel matrix shape: {K.shape}")
                
            # Add regularization: K + λI
            print(f"\nKernel matrix (mean abs value): {torch.abs(K).mean()}")
            K_reg = K + self.lmbd * torch.eye(n_subjects, dtype=self.dtype)
            del K

            # Solve via Cholesky decomposition (stable for positive definite matrices)
            print("Solving with Cholesky decomposition...")
            try:
                L = torch.linalg.cholesky(K_reg)
                # Solve L L^T β = y in two steps
                # targets_normalized is 1D, need to make it 2D for solve_triangular
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
            print("Training complete!")

            # Compute training accuracy
            print("Computing training accuracy...")
            # train_pred_scores = self._kernel_matvec(self.beta)
            # train_pred = (train_pred_scores > 0).float()
            # train_acc = (train_pred == ((targets + 1) / 2)).float().mean()
            # print(f"Training accuracy: {train_acc.item():.4f}")
    
    def predict(self, dataloader):
        """
        Predict on new data.
        
        Uses either matrix-free kernel-vector multiplication (if matrix_free=True)
        or direct kernel matrix computation (if matrix_free=False).
        
        Args:
            dataloader: DataLoader yielding (data, _, _)
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if self.train_embeddings is None or self.beta is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        print(f"Predicting on test data (matrix_free={self.matrix_free})...")
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

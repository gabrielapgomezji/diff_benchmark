import torch
from torch import nn
from tqdm import tqdm


def dist_emb_circle_pairwise_chunked(emb_u, emb_v, chunk_size=100, compute_device=None):
    """
    Compute pairwise distance between embeddings with chunking to reduce memory.
    Keeps embeddings on CPU and moves chunks to GPU for computation.
    
    Args:
        emb_u: shape (n, d) - first set of embeddings (on CPU)
        emb_v: shape (m, d) - second set of embeddings (on CPU)
        chunk_size: number of rows to process at once
        compute_device: device to use for computation (GPU if available)
    
    Returns:
        dist: shape (n, m) - pairwise squared distances (on CPU)
    """
    n, d = emb_u.shape
    m = emb_v.shape[0]
    dtype = emb_u.dtype
    
    # Keep result on CPU
    dist_matrix = torch.zeros(n, m, dtype=dtype)
    
    # Determine compute device - ensure it's always valid
    if compute_device is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif compute_device is not None and not isinstance(compute_device, torch.device):
        compute_device = torch.device(compute_device)

    # Process in chunks to avoid memory issues
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        
        # Move chunk to device for computation
        chunk_u = emb_u[i:end_i].to(compute_device)  # (chunk_size, d)
        emb_v_device = emb_v.to(compute_device)  # (m, d)
        
        # Compute distance for this chunk
        diff = torch.abs(chunk_u[:, None, :] - emb_v_device[None, :, :])  # (chunk_size, m, d)
        dist_uv = torch.minimum(diff, 1 - diff)
        chunk_dist = torch.mean(dist_uv ** 2, dim=-1)  # (chunk_size, m)
        
        # Move result back to CPU
        dist_matrix[i:end_i] = chunk_dist.cpu()
        
        # Clean up memory
        del chunk_u, emb_v_device, diff, dist_uv, chunk_dist
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()
    
    return dist_matrix


class EfficientKernelRidgeRegression(nn.Module):
    """
    Memory-efficient Kernel Ridge Regression for sphere embeddings.
    
    Key optimizations:
    1. Average sphere embeddings weighted by power (instead of per-sphere kernels)
    2. Chunked distance computation to avoid OOM
    3. No learnable alpha parameters (uniform weighting)
    4. Direct kernel computation without storing large intermediate matrices
    
    This reduces memory from O(n_spheres * n_subjects^2) to O(n_subjects^2).
    
    Args:
        lmbd: Regularization parameter
        chunk_size: Batch size for chunked distance computation
        use_power_weighting: Whether to weight spheres by their power values
        device: Compute device
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        lmbd: float = 0.1,
        chunk_size: int = 64,
        use_power_weighting: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.lmbd = lmbd
        self.chunk_size = chunk_size
        self.use_power_weighting = use_power_weighting
        
        # Always set a valid device
        self.compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Always set a valid dtype
        self.dtype = torch.float32

        # Training data storage (kept on CPU)
        self.train_embeddings = None
        self.beta = None
        self.kernel_bandwidth = None 
        
    def _aggregate_embeddings(self, embeddings, power=None):
        """
        Aggregate sphere embeddings into a single representation per subject.
        
        Args:
            embeddings: shape (n_subjects, n_spheres, n_bvals, d)
            power: shape (n_subjects, n_spheres, n_bvals) - optional power weights
        
        Returns:
            aggregated: shape (n_subjects, n_bvals, d) - one embedding per subject per bval
        """
        n_subjects, n_spheres, n_bvals, d = embeddings.shape
        
        if power is None or not self.use_power_weighting:
            # Simple average across spheres
            return embeddings.mean(dim=1)  # (n_subjects, n_bvals, d)
        else:
            # Weighted average by power
            # Normalize power to sum to 1 across spheres
            power_weights = power / (power.sum(dim=1, keepdim=True) + 1e-8)  # (n_subjects, n_spheres, n_bvals)
            power_weights = power_weights.unsqueeze(-1)  # (n_subjects, n_spheres, n_bvals, 1)
            
            # Weighted sum
            weighted_emb = (embeddings * power_weights).sum(dim=1)  # (n_subjects, n_bvals, d)
            return weighted_emb
    
    def _compute_kernel_matrix(self, emb1, emb2, bandwidth=None):
        """
        Compute RBF kernel matrix between two sets of aggregated embeddings.
        Keeps embeddings on CPU and computes in batches on GPU.
        
        Args:
            emb1: shape (n, n_bvals, d) - on CPU
            emb2: shape (m, n_bvals, d) - on CPU
            bandwidth: kernel bandwidth (if None, use self.kernel_bandwidth)
        
        Returns:
            K: shape (n, m) - kernel matrix (on CPU)
        """
        if bandwidth is None:
            bandwidth = self.kernel_bandwidth
            
        n = emb1.shape[0]
        m = emb2.shape[0]
        n_bvals = emb1.shape[1]
        
        # Flatten bvals dimension for distance computation
        emb1_flat = emb1.reshape(n, -1)  # (n, n_bvals * d) - on CPU
        emb2_flat = emb2.reshape(m, -1)  # (m, n_bvals * d) - on CPU
        
        # Compute distances in chunks (keeps data on CPU, computes on GPU)
        dist_matrix = dist_emb_circle_pairwise_chunked(
            emb1_flat, emb2_flat, chunk_size=self.chunk_size, compute_device=self.compute_device
        )
        
        # Move to GPU for kernel computation
        dist_matrix_gpu = dist_matrix.to(self.compute_device)
        K_gpu = torch.exp(-dist_matrix_gpu / bandwidth)
        K = K_gpu.cpu()
        
        # Clean up
        del dist_matrix_gpu, K_gpu
        if self.compute_device.type == "cuda":
            torch.cuda.empty_cache()
        
        return K
    
    def fit(self, dataloader):
        """
        Fit the kernel ridge regression model.
        
        Args:
            dataloader: DataLoader yielding (data, targets, _) where
                        data is dict with 'embeddings' and 'power'
        """
        print("Loading training data in chunks...")
        
        # First pass: collect aggregated embeddings and targets in chunks
        aggregated_chunks = []
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
            
            # Aggregate embeddings for this batch (CPU)
            aggregated_emb = self._aggregate_embeddings(embeddings, power)
            
            # Store aggregated results (much smaller than raw embeddings)
            aggregated_chunks.append(aggregated_emb)
            target_chunks.append(targets_batch.to(self.dtype))
            
            # Clear the large raw embeddings from memory
            del embeddings, power, data
        
        print("Concatenating aggregated embeddings...")
        # Now concatenate only the aggregated embeddings (much smaller)
        aggregated_emb = torch.cat(aggregated_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)
        
        # Clear chunks
        del aggregated_chunks, target_chunks
        
        print(f"Aggregated embeddings shape: {aggregated_emb.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Convert targets to {-1, 1}
        targets = targets * 2 - 1
        
        # Store for prediction (on CPU)
        self.train_embeddings = aggregated_emb
        
        # Estimate kernel bandwidth from data
        print("Computing kernel bandwidth from data statistics...")
        with torch.no_grad():
            # Sample a subset for bandwidth estimation to save memory
            n_samples = min(100, aggregated_emb.shape[0])
            sample_idx = torch.randperm(aggregated_emb.shape[0])[:n_samples]
            sample_emb = aggregated_emb[sample_idx].reshape(n_samples, -1)
            
            # Compute pairwise distances on sample
            sample_dist = dist_emb_circle_pairwise_chunked(
                sample_emb, sample_emb, chunk_size=self.chunk_size
            )
            self.kernel_bandwidth = sample_dist.median().item()
            print(f"Kernel bandwidth: {self.kernel_bandwidth:.6f}")
        
        # Compute kernel matrix
        print("Computing kernel matrix...")
        n_subjects = aggregated_emb.shape[0]
        
        with torch.no_grad():
            K = self._compute_kernel_matrix(aggregated_emb, aggregated_emb)  # K is on CPU
            print(f"Kernel matrix shape: {K.shape}")
            
            # Solve ridge regression: beta = (K + lambda*I)^{-1} * y
            print("Solving ridge regression...")
            
            # Move to GPU for solving
            K_gpu = K.to(self.compute_device)
            targets_gpu = targets.to(self.compute_device)
            I = torch.eye(n_subjects, device=self.compute_device, dtype=self.dtype)
            K_reg = K_gpu + self.lmbd * I
            
            # Use Cholesky decomposition for numerical stability
            try:
                L = torch.linalg.cholesky(K_reg)
                beta_gpu = torch.cholesky_solve(targets_gpu.unsqueeze(1), L).squeeze(1)
            except RuntimeError:
                print("Warning: Cholesky decomposition failed, using standard inverse")
                beta_gpu = torch.linalg.solve(K_reg, targets_gpu)
            
            # Move beta back to CPU
            self.beta = beta_gpu.cpu()
            
            print(f"Beta coefficients shape: {self.beta.shape}")
            print("Training complete!")
            
            # Compute training accuracy
            train_pred = (K_gpu @ beta_gpu) > 0
            train_acc = (train_pred.float() == ((targets_gpu + 1) / 2)).float().mean()
            print(f"Training accuracy: {train_acc.item():.4f}")
            
            # Clean up GPU memory
            del K_gpu, targets_gpu, I, K_reg, beta_gpu, train_pred
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
        
        print("Predicting on test data...")
        all_predictions = []
        
        with torch.no_grad():
            for i, (data, _, _) in enumerate(dataloader):
                print(f"  Batch {i+1}/{len(dataloader)}")
                
                # Load and aggregate test embeddings - keep on CPU
                embeddings = data["embeddings"].to(self.dtype)
                power = data["power"].to(self.dtype)
                
                # Remove batch dimension if present
                if embeddings.dim() == 5:
                    embeddings = embeddings.squeeze(1)
                if power.dim() == 4:
                    power = power.squeeze(1)
                
                # Aggregate embeddings (on CPU)
                aggregated_emb = self._aggregate_embeddings(embeddings, power)
                
                # Compute kernel with training data (computation happens on GPU in chunks)
                K_test = self._compute_kernel_matrix(aggregated_emb, self.train_embeddings)
                
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

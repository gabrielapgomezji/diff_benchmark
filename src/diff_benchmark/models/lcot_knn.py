import numpy as np
import torch
from torch import nn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


class kNNLCOT(nn.Module):
    """
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
        seed: int = 42,  # Random seed for reproducibility
        use_l2_norm: bool = False,  # Use L2 norm instead of circular distance
        **kwargs,
    ):
        super().__init__()
        self.lmbd = lmbd
        self.use_power_weighting = use_power_weighting
        self.sphere_batch_size = sphere_batch_size
        self.sample_batch_size = sample_batch_size
        self.seed = seed
        self.use_l2_norm = use_l2_norm
        
        # Always set a valid device
        self.compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Always set a valid dtype
        self.dtype = torch.float32

        # Training data storage (kept on CPU)
        self.train_embeddings = None
        self.train_power = None


    def _compute_dist_tile(self, emb1_block, emb2_block):
        """
        Helper method to compute kernel values for a tile of samples.
        
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
        
        Returns:
            K_block: shape (n1_blk, n2_blk) - kernel values (on GPU, float64)
        """
        n1_blk, n_spheres, n_bvals, d = emb1_block.shape
        n2_blk = emb2_block.shape[0]
                
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

                # (n1_blk, n2_blk, n_spheres_batch) -> (n1_blk, n2_blk)
                dist_sum = torch.sum(msd, dim=-1)
                sum_normalized_distances += dist_sum
                
                del msd, emb1_sb, emb2_sb, diff, dist_sum

        return sum_normalized_distances


    def _compute_dist_matrix(self, emb1, emb2):
        """
        Compute distance matrix using additive accumulation with tiling.
        
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
        
        Returns:
            K: shape (n1, n2) - kernel matrix (on CPU)
        """
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
                K_block = self._compute_dist_tile(
                    emb1_block, emb2_block
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
        """
        print("=" * 60)
        print("kNN - Training")
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
        self.targets = torch.cat(target_chunks, dim=0)

        # Clear chunks
        del embeddings_chunks, power_chunks, target_chunks
        
        print(f"Training embeddings shape: {self.train_embeddings.shape}")
        print(f"Training power shape: {self.train_power.shape}")
        print(f"Targets shape: {self.targets.shape}")


        d_train = self._compute_dist_matrix(self.train_embeddings, self.train_embeddings)

        knn = KNeighborsClassifier(n_neightbors=5, metric="precomputed")

        # param_grid = {"n_neighbors": np.arange(1, 25)}
        # knn_gscv = GridSearchCV(knn, param_grid, cv=5)
        # knn_gscv.fit(d_train, self.targets)
        # self.model = knn_gscv
        # print(f"Best parameters:", knn_gscv.best_params_)

        knn.fit(d_train, self.targets)
        self.model = knn
        

    def predict(self, dataloader):
        """
        Predict on new data.
        
        Args:
            dataloader: DataLoader yielding (data, _, _)
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if self.train_embeddings is None:
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

                dist_test = self._compute_dist_matrix(self.train_embeddings, embeddings)

                predictions = self.model.predict(dist_test.numpy().T)
                all_predictions.append(torch.from_numpy(predictions))
                
                # Clean up
                del embeddings, power
        
        print("Prediction complete!")
        return torch.cat(all_predictions, dim=0)

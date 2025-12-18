import torch
from torch import nn
from tqdm import tqdm

from diff_benchmark.models.utils_models.lcot_utils import dist_emb_circle_pairwise


class KernelRidgeRegression(nn.Module):
    """
    Implementation of a kernel ridge regression model.
    It uses a learned combination of kernel functions to perform regression on spherical embeddings.
    Attributes:
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        alphas (torch.nn.Parameter): Learnable parameters representing the weights for each sphere.
        lmbd (float): Regularization parameter lambda.
        embeddings (torch.Tensor or None): Training embeddings used for fitting the model.
        beta (torch.Tensor or None): Coefficients learned during training.
        device (torch.device or None): Device on which tensors are allocated.
        dtype (torch.dtype or None): Data type of the tensors.
    Methods:
        fit(embeddings, targets):
            Trains the model using the provided embeddings and target values.
                embeddings (torch.Tensor): Input embeddings of shape (N, S, D), where N is the number of samples,
                                            S is the number of spheres, and D is the dimensionality of each embedding.
                targets (torch.Tensor): Target values of shape (N,).
            Returns:
                None
        predict(embedding):
            Predicts target values for the given embeddings using the trained model.
                embedding (torch.Tensor): Input embeddings of shape (N, S, D), where N is the number of samples,
                                           S is the number of spheres, and D is the dimensionality of each embedding.
            Returns:
                torch.Tensor: Predicted target values of shape (N,)."""

    data_type = "lcot_embed"

    def __init__(
        self,
        num_spheres: int,
        epochs: int = 100,
        lmbd: float = 0.1,
        lr: float = 0.01,
        device: torch.device = None,
        dtype: torch.dtype = None,
        **kwargs,
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        self.alphas = nn.Parameter(torch.zeros(num_spheres, device=device, dtype=dtype))
        # self.alphas = nn.Parameter(torch.zeros(num_spheres, device=device, dtype=dtype))
        self.lmbd = lmbd
        self.embeddings = None
        self.beta = None
        self.device = device
        self.dtype = dtype
        self.std_distances = None
        self.bval_idx = 2

    def fit(self, dataloader):
        """Fit the Kernel Ridge Regression model using the provided dataloader."""
        all_embeddings = []
        all_targets = []
        all_powers = []

        for i, (data, targets, _) in enumerate(dataloader):
            print(f"Loading batch {i+1}/{len(dataloader)}")
            all_embeddings.append(data["embeddings"])
            all_targets.append(targets)
            all_powers.append(data["power"])
        # --- Concatenate into single tensors ---
        embeddings = torch.cat(all_embeddings, dim=0).to(self.device).to(self.dtype)
        targets = torch.cat(all_targets, dim=0).to(self.device)
        targets = targets * 2 - 1  # Convert to -1, 1
        power = torch.cat(all_powers, dim=0).to(self.device).to(self.dtype)
        # Remove batch dimension if needed
        embeddings = embeddings.squeeze(dim=1)
        power = power.squeeze(dim=1)
        # targets = targets.to(self.device)
        # embeddings = data['embeddings'].to(self.device).to(self.dtype).squeeze(dim=1)
        # power = data['power'].to(self.device).to(self.dtype).squeeze(dim=1)
        embeddings = embeddings[:, :, self.bval_idx, :][
            :, :, None, :
        ]  # Use only the first b-value for distance computation DEBUGGING
        self.embeddings = embeddings
        optimizer = torch.optim.Adam([self.alphas], lr=self.lr)
        with torch.no_grad():
            n_subjects, _, n_spheres, _ = embeddings.shape
            # n_total_spheres = n_spheres * n_bval
            n_total_spheres = 1

            dist_matrix = torch.zeros(
                n_total_spheres,
                n_subjects,
                n_subjects,
                device=self.device,
                dtype=self.dtype,
            )
            for s in tqdm(range(n_spheres)):
                dist_matrix[s] = dist_emb_circle_pairwise(
                    embeddings[:, s, 0, :], embeddings[:, s, 0, :]
                )
            print("Distance matrix ready.")
            self.std_distances = dist_matrix.std()
            dist_matrix_norm = dist_matrix / self.std_distances
            id_mat = torch.eye(n_subjects, device=self.device, dtype=self.dtype)

            weights = nn.functional.softmax(self.alphas, dim=0)
            K = torch.exp(-(weights[:, None, None] * dist_matrix_norm).sum(dim=0))
            print("K computed for train")

            self.beta = (K + self.lmbd * id_mat).inverse() @ targets
            print("Beta computed")

        # for epoch in range(self.epochs):
        #     weights = nn.functional.softmax(self.alphas, dim=0)
        #     # weights_expanded = weights.repeat_interleave(n_bval) # If all bvals used
        
        #     # weights = weights_expanded
        #     breakpoint()
        #     K = torch.exp(-(weights[:, None, None] * dist_matrix_norm).sum(dim=0))
        #     beta = (K + self.lmbd * id_mat).inverse() @ targets
        #     loss = ((K @ beta - targets) ** 2).mean()
        #     # breakpoint()
        #     loss.backward()
        #     print(loss.item())
        #     optimizer.step()
        #     optimizer.zero_grad()
        # self.beta = beta
        return

    def predict(self, dataloader):
        """Predict using the trained Kernel Ridge Regression model."""
        # embedding:torch.Tensor
        all_predictions = []

        for data, _, _ in dataloader:

            # --- Concatenate into single tensors ---
            embeddings = (
                data["embeddings"]
                .squeeze(dim=1)[:, :, self.bval_idx, :][:, :, None, :]
                .to(self.device)
                .to(self.dtype)
            )

            # Remove batch dimension if needed
            # data , _, _ = next(iter(dataloader))
            # embeddings = data['embeddings'].to(self.device).to(self.dtype).squeeze(dim=1)
            self_n_subjects, self_n_spheres, _, _ = self.embeddings.shape
            n_subjects, _, _, _ = embeddings.shape
            with torch.no_grad():
                # assert self_n_spheres * self_n_bval == n_spheres * n_bval, f"Number of spheres in the training set {self_n_spheres} does not match the number of spheres in the test set {n_spheres}"
                # dist_matrix = torch.zeros(self_n_spheres * self_n_bval, self_n_subjects, n_subjects, device=self.device, dtype=self.dtype)
                dist_matrix = torch.zeros(
                    self_n_spheres,
                    n_subjects,
                    self_n_subjects,
                    device=self.device,
                    dtype=self.dtype,
                )
                for s in range(self_n_spheres):
                    dist_matrix[s] = dist_emb_circle_pairwise(
                        embeddings[:, s, 0, :], self.embeddings[:, s, 0, :]
                    )
                dist_matrix /= self.std_distances
                weights = nn.functional.softmax(self.alphas, dim=0)
                K = torch.exp(-(weights[:, None, None] * dist_matrix).sum(dim=0))
                print("K computed for prediction")
                outputs = ((K @ self.beta) > 0).float()
                all_predictions.append(outputs)
        return torch.cat(all_predictions, dim=0).cpu()

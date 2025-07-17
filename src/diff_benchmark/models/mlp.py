import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[128, 64],
        dropout=0.2,
        lr=1e-3,
        epochs=10,
        device=None,
    ):
        """
        Initialize the Multi-Layer Perceptron (MLP) model.
        Parameters:
            input_dim (int): The number of input features.
            output_dim (int): The number of output classes.
            hidden_dims (list, optional): A list of integers representing the number of units in each hidden layer. Default is [128, 64].
            dropout (float, optional): The dropout rate to be applied after each hidden layer. Default is 0.2.
            lr (float, optional): The learning rate for the optimizer. Default is 1e-3.
            epochs (int, optional): The number of training epochs. Default is 10.
            device (str, optional): The device to run the model on ('cuda' or 'cpu'). If None, it will use 'cuda' if available, otherwise 'cpu'.
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.lr = lr

        # Build the MLP layers
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        Args:
            x (torch.Tensor): Input tensor to the network.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """

        return self.network(x)

    def _dataloader_to_numpy(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        """
        Converts a PyTorch DataLoader to NumPy arrays.
        This method iterates through the provided DataLoader, extracting batches of
        input data and corresponding labels, and converts them from PyTorch tensors
        to NumPy arrays. The resulting arrays are concatenated along the first axis
        to form a single input array and a single label array.
        Args:
            dataloader (torch.utils.data.DataLoader): The DataLoader containing
            batches of data and labels.
        Returns:
            tuple: A tuple containing two NumPy arrays:
                - X (np.ndarray): Concatenated input data from all batches.
                - Y (np.ndarray): Concatenated labels from all batches.
        """

        X_list, Y_list = [], []
        for x_batch, y_batch, _ in dataloader:
            X_list.append(x_batch.numpy())
            Y_list.append(y_batch.numpy())
        return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)

    def fit(self, dataloader: DataLoader):
        """
        Fit the model to the provided dataloader.
        This method trains the model for a specified number of epochs using the data
        from the dataloader. It processes each batch of input data, computes the
        loss, and updates the model parameters accordingly.
        Args:
            dataloader (DataLoader): A PyTorch DataLoader object that provides
            batches of input data and corresponding labels.
        Returns:
            None
        """

        self.train()
        for epoch in range(self.epochs):
            for x_batch, y_batch, _ in dataloader:
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).long()

                logits = self(x_batch)
                loss = self.loss_fn(logits, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, dataloader: DataLoader):
        """
        Predicts the output for the given dataloader.
        Args:
            dataloader (DataLoader): A PyTorch DataLoader object that provides batches of input data.
        Returns:
            numpy.ndarray: A concatenated array of predicted class indices for all input batches.
        """

        self.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _, _ in dataloader:
                x_batch = x_batch.to(self.device).float()
                logits = self(x_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds)

    def predict_proba(self, dataloader: DataLoader):
        """
        Predict the class probabilities for the given dataloader.
        Args:
            dataloader (DataLoader): A PyTorch DataLoader object that provides batches of input data.
        Returns:
            numpy.ndarray: A concatenated array of predicted probabilities for each class,
                           with shape (n_samples, n_classes).
        """

        self.eval()
        all_probs = []
        with torch.no_grad():
            for x_batch, _, _ in dataloader:
                x_batch = x_batch.to(self.device).float()
                probs = torch.softmax(self(x_batch), dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64], dropout=0.2, lr=1e-3, epochs=10, device=None):
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

    def forward(self, x):
        return self.network(x)

    def _dataloader_to_numpy(self, dataloader):
        X_list, Y_list = [], []
        for x_batch, y_batch, _ in dataloader:
            X_list.append(x_batch.numpy())
            Y_list.append(y_batch.numpy())
        return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)

    def fit(self, dataloader: DataLoader):
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
        self.eval()
        all_probs = []
        with torch.no_grad():
            for x_batch, _, _ in dataloader:
                x_batch = x_batch.to(self.device).float()
                probs = torch.softmax(self(x_batch), dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs)

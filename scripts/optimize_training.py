import os
import time

import torch
import torch.nn as nn
from main import GoogleViTBackbone, GoogleViTClassifier
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Constants
NUM_SAMPLES = 20  # Reduced for fast testing
BATCH_SIZE = 1  # Often 1 for the heavy 3D backbone due to VRAM
HEAD_BATCH_SIZE = 32  # Can be large for the lightweight head training
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = "cached_features.pt"


class CachedFeatureDataset(Dataset):
    """
    On init:
    1. Checks if 'cache_path' exists.
    2. IF YES: Loads features from disk (Fast I/O).
    3. IF NO: Runs backbone on 'source_dataloader', saves features to disk.
    """

    def __init__(self, cache_path, backbone=None, source_dataloader=None, device=None):
        self.cache_path = cache_path

        if os.path.exists(cache_path):
            print(f"\n--- Loading Features from Cache: {cache_path} ---")
            print("Skipping backbone computation...")
            start = time.time()
            data = torch.load(cache_path)
            self.features = data["features"]
            self.labels = data["labels"]
            print(
                f"Loaded {len(self.features)} samples in {time.time() - start:.4f} seconds."
            )
            print(f"File size: {os.path.getsize(cache_path) / 1024**2:.2f} MB")

        else:
            if backbone is None or source_dataloader is None:
                raise ValueError(
                    "Cache not found. backbone and source_dataloader are required to generate it."
                )

            print(f"\n--- Cache miss. Computing Features (The Slow Part) ---")
            self.features, self.labels = self._precompute(
                backbone, source_dataloader, device
            )

            print(f"Saving features to {cache_path}...")
            torch.save({"features": self.features, "labels": self.labels}, cache_path)
            print(f"Saved. Future runs will be instant.")

    def _precompute(self, backbone, dataloader, device):
        backbone.eval()
        all_features = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            with torch.autocast(device_type=device.type):
                for i, (x, y) in enumerate(dataloader):
                    if i % 10 == 0:
                        print(f"Processing sample {i}/{len(dataloader)}...", end="\r")

                    x = x.to(device)
                    # Squeeze channel dim if present (N, 1, D, H, W) -> (N, D, H, W)
                    if x.ndim == 5 and x.shape[1] == 1:
                        x = x.squeeze(1)

                    features = backbone(x)

                    all_features.append(features.cpu())
                    all_labels.append(y)

        print(
            f"\nFeature extraction finished in {time.time() - start_time:.2f} seconds."
        )
        return torch.cat(all_features), torch.cat(all_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_dummy_data():
    """Simulating your dataset of 800 volumes."""
    print(f"Generating {NUM_SAMPLES} dummy samples...")
    # Using random data (0-1) to simulate images
    data = torch.rand(NUM_SAMPLES, 1, 192, 256, 256)  # (N, C, D, H, W)
    labels = torch.randint(0, 10, (NUM_SAMPLES,))  # Dummy classification labels
    return TensorDataset(data, labels)


def train_head_only(head_model, cached_dataset, device):
    print("\n--- Phase 2: Training Head on Cached Features (The Fast Part) ---")

    # We can use a much larger batch size now because input is just a vector, not a 3D volume
    train_loader = DataLoader(cached_dataset, batch_size=HEAD_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(head_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    head_model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = head_model.head(
                features
            )  # Access the 'head' part of the classifier directly
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.4f}"
            )

    print(f"Training finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    # 1. Setup Data
    full_dataset = generate_dummy_data()

    # 2. Setup Frozen Backbone
    print("Loading Backbone...")
    backbone = GoogleViTBackbone(freeze_backbone=True, backbone_grad=False).to(DEVICE)

    # 3. Smart Dataset (Handles Caching Automatically)
    # We pass the backbone/loader ONLY if needed for generation.
    # Otherwise, it loads from disk and backbone remains unused/idle.
    backbone_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cached_dataset = CachedFeatureDataset(
        cache_path=CACHE_PATH,
        backbone=backbone,
        source_dataloader=backbone_loader,
        device=DEVICE,
    )

    # 4. Free up memory
    # (In a real script, if cache was loaded, we never even needed to load the backbone to GPU!)
    del backbone
    torch.cuda.empty_cache()

    # 5. Setup Classifier Head
    print("Initializing Head...")
    # Using dummy backbone just to init the config
    dummy_backbone = GoogleViTBackbone(freeze_backbone=True)
    full_model = GoogleViTClassifier(dummy_backbone).to(DEVICE)

    # 6. Train Head
    train_head_only(full_model, cached_dataset, DEVICE)

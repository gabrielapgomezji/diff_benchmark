import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification
from diff_benchmark.models.base import LightningModel
from diff_benchmark.models.utils import create_trainer
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import json, csv
from diff_benchmark.utils.logger import TrainLogger

def collate_with_augmentation(batch, transform=None):
    """Custom collate function that applies 2D augmentations to each slice of 3D volumes in the batch."""
    xs, ys, gs = zip(*batch)  # separate batch components
    xs_aug = []
    for x in xs:  # x shape: (D,H,W)
        slices = []
        for i in range(x.shape[0]):
            slice_2d = x[i, :, :].unsqueeze(0)  # (1,H,W)
            if transform:
                slice_2d = transform(slice_2d)
            slices.append(slice_2d)
        x_aug = torch.stack(slices, dim=0)  # (D,1,H,W)
        x_aug = x_aug.permute(1, 0, 2, 3)  # (C=1,D,H,W)
        xs_aug.append(x_aug)

    xs_aug = torch.stack(xs_aug, dim=0)
    ys = torch.stack(ys)
    gs = torch.stack(gs)
    return xs_aug, ys, gs


train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
val_transforms = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

def extract_pseudo_rgb(volume):
    """
    volume: (B, 1, S, H, W)
    Returns: (B, 3, H, W)
    """
    B, C, S, H, W = volume.shape
    assert C == 1, "Expected 1-channel MRI volume"
    
    mid = S // 2

    # Safe padding if mid-1 or mid+1 out of range
    s1 = max(mid - 1, 0)
    s2 = mid
    s3 = min(mid + 1, S - 1)

    rgb = torch.stack([
        volume[:, 0, s1],
        volume[:, 0, s2],
        volume[:, 0, s3],
    ], dim=1)  # (B, 3, H, W)

    return rgb

class HFViTWrapper(nn.Module):
    def __init__(self, num_classes=2, model_name="google/vit-large-patch16-224"):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # allows head replacement
        )

    def forward(self, images):
        # images: (B, C, H, W) tensors in standard torch format
        # Convert to PIL for HF processor
        images = extract_pseudo_rgb(images)
        images_cpu = [
            ((img.permute(1,2,0).cpu().numpy() + 1.0) / 2.0).clip(0,1) 
            for img in images
        ] # [-1,1] -> [0,1]
        pixel_values = self.processor(images=images_cpu, return_tensors="pt", do_rescale=False)["pixel_values"]
        pixel_values = pixel_values.to(images.device)

        outputs = self.model(pixel_values)
        return outputs.logits
    
class ViTBase(LightningModel):
    data_type = "images"
    def __init__(self, lr=3e-4, num_classes=2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        self.device_str = kwargs.get("device", "cuda")
        self.epochs = kwargs.get("epochs", 100)
        self.run_id = kwargs.get("run_id", "default_run")
        self.fold_idx = kwargs.get("fold_idx", 0)
        
        # self.logger = None
        self.history = {
            "train": {"epoch": [], "batch": [], "loss": [], "metrics": []},
            "val": {"epoch": [], "loss": [], "metrics": [], "batch_train_idx": []},
        }
        
        # Will be created in build_model()
        self.build_model()
        self.criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------
    # Build model (calls HFViTWrapper)
    # ----------------------------------------------------
    def build_model(self):
        self.model = HFViTWrapper(num_classes=self.num_classes)

    # ----------------------------------------------------
    # Forward just routes to HFViTWrapper
    # ----------------------------------------------------
    def forward(self, x):
        return self.model(x)
    
    def _save_logs(self, history, save_path):
        """Utility for saving training logs as JSON or CSV."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f)
        elif path.suffix == ".csv":
            keys = history[0].keys()
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(history)
        else:
            raise ValueError("Save path must end with .json or .csv")

    def _train_val_loader_split(self, train_loader, val_ratio=0.3):
        """Split the incoming dataloader's dataset into train and validation subsets."""
        dataset = train_loader.dataset  # access the underlying dataset
        n = len(dataset)
        genders = np.asarray(dataset.dataset.gender[dataset.indices])

        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=genders, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Transform-aware collation (optional)
        train_loader_new = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=19,  # 0,#
            pin_memory=False,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=train_transforms
            ),
        )
        val_loader_new = DataLoader(
            val_subset,
            batch_size=128,
            shuffle=False,
            num_workers=19,  # 0,#10,
            pin_memory=False,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=val_transforms
            ),
        )
        return train_loader_new, val_loader_new
    
    def fit(self, dataloader):
        """
        Lightning-based fit function to preserve compatibility with the old API.
        Handles: train/val split, trainer setup, checkpointing.
        """
        print(f"Device: {self.device_str}")
        print(f"Fold index: {self.fold_idx}")

        # Split train/val
        train_loader, val_loader = self._train_val_loader_split(dataloader)
        print("Dataloaders created.")

        trainer = create_trainer(
            max_epochs=self.epochs,
            monitor="val_accuracy",  # you use this everywhere else
            mode="max",
            patience=10,
            accelerator="gpu" if "cuda" in self.device_str else "cpu",
            devices=1,
            save_dir=f"./data/results/checkpoints/{self.run_id}/fold_{self.fold_idx}",
        )
        self.logger = None
        self.logger = TrainLogger(
            fold_idx=self.fold_idx,
            run_id=self.run_id,
            save_dir=f"./data/results/logger/{self.run_id}/fold_{self.fold_idx}",
            monitor="val_accuracy",
            mode="max",
        )

        # Train the model
        trainer.fit(self, train_loader, val_loader)

        # Save trainer for later prediction
        self.trainer = trainer

        print(
            f"[INFO] Training finished. Best model: {trainer.checkpoint_callback.best_model_path}"
        )
        
        if self.logger is not None:
            try:
                self.logger.save_checkpoint(self.model, self.epochs, 0, is_last=True)
                self.logger.save_logs()
            except Exception:
                # swallow logger errors to avoid breaking training completion
                pass
        # save history JSON
        try:
            self._save_logs(self.history, f"./data/results/logs/{self.run_id}_training_log.json")
        except Exception:
            pass
    
    def predict(self, dataloader):
        """
        Lightning-based predict function compatible with the old API.
        Loads the best checkpoint automatically.
        """
        dataset = dataloader.dataset

        # Build a fresh dataloader with collate + transforms
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=19,
            pin_memory=False,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=val_transforms
            ),
        )

        # If fit() has not been called → create default trainer
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            trainer = create_trainer(
                accelerator="gpu" if "cuda" in self.device_str else "cpu",
                devices=1,
                max_epochs=1,
            )

        # best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        best_path = None
        if self.logger is not None:
            best_path = getattr(self.logger, "best_path", None)
        if not best_path:
            best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)

        self.eval()

        # If there is a checkpoint → use it
        if best_path and Path(best_path).exists():
            preds_all = trainer.predict(
                self,
                dataloaders=dataloader,
                ckpt_path=best_path
            )
        else:
            # Fallback: no checkpoint, so use x-only loader
            preds_all = trainer.predict(
                self,
                dataloaders=self.x_only_loader(dataloader)
            )

        preds = torch.cat([p.cpu() for p in preds_all])
        return preds.numpy()

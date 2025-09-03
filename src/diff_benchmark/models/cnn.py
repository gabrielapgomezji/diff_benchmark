import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from diff_benchmark.models.base import TorchAbstractModel
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import csv

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove final FC
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.out_dim = 512

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, 512)
        """
        feats = self.feature_extractor(x)  # (B, 512, 1, 1)
        return feats.view(feats.size(0), -1)  # (B, 512)
    
class ResNet3SliceClassifier(nn.Module):
    def __init__(self, input_slices, num_classes=2, freeze_backbone=True, **kwargs):
        super().__init__()
        self.backbone = ResNet18Backbone(**kwargs)
        self.num_subvols = input_slices // 3
        self.fc = nn.Linear(self.num_subvols * self.backbone.out_dim, num_classes)
        
        # freeze_backbone = kwargs.get("freeze_backbone", True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: (B, S, H, W) where S is slice dimension
        """
        B, S, H, W = x.shape
        # assert S % 3 == 0, "Slice dimension must be divisible by 3"

        subvols = x.unfold(dimension=1, size=3, step=3)  # (B, num_subvols, 3, H, W)
        subvols = subvols.permute(0, 1, 4, 2, 3)
        num_subvols = subvols.size(1)

        feats = []
        for i in range(num_subvols):
            sv = subvols[:, i]              # (B, 3, H, W)
            f = self.backbone(sv)           # (B, 512)
            feats.append(f)

        feats = torch.cat(feats, dim=1)     # (B, num_subvols*512)
        # Add dropout
        # Normalization layer
        out = self.fc(feats)                # (B, num_classes)
        return out

class ResNet3SliceModel(TorchAbstractModel, nn.Module):
    def __init__(self, input_slices=145, num_classes=2, device="cuda", **kwargs): # change "cpu" to "cuda" if GPU is available
        super(ResNet3SliceModel, self).__init__()
        self.device = device
        self.run_id = kwargs.get("run_id", "unnamed_run")
        self.model = ResNet3SliceClassifier(input_slices=input_slices, num_classes=num_classes, **kwargs).to(device)
        self.criterion = nn.CrossEntropyLoss()
        lr = kwargs.get("learning_rate", 1e-5)
        if kwargs.get("freeze_backbone", True):
            self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=1e-5)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        self.best_val_model = 0
        self.history = {
            "train": {"epoch": [], "batch": [], "loss": [], "accuracy": []},
            "val": {"epoch": [], "batch": [], "loss": [], "accuracy": [], "batch_train_idx": []}
        }

    def _dataloader_to_numpy(self, dataloader):
        X, y, g = [], [], []
        for xb, yb, gb in dataloader:
            X.append(xb.numpy())
            y.append(yb.numpy())
            g.append(gb.numpy())
        return np.concatenate(X), np.concatenate(y), np.concatenate(g)
    
    def _train_val_loader_split(self, train_loader, val_ratio=0.3):
        dataset = train_loader.dataset  # access the underlying dataset
        n = len(dataset)
        
        genders = []
        for i in range(n):
            _, _, g = dataset[i]
            genders.append(g)
        genders = np.array(genders)
        
        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, stratify=genders, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader_new = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader_new = DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=False)
        return train_loader_new, val_loader_new
    
    def _save_logs(self, history, save_path):
        path = Path(save_path)
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(history, f)
        elif path.suffix == ".csv":
            keys = history[0].keys()
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(history)
        else:
            raise ValueError("Save path must end with .json or .csv")

    def fit(self, dataloader, epochs=1): #epochs=100
        print(f"Device: {self.device}")
        self.model.train()
        train_loader, val_loader = self._train_val_loader_split(dataloader)
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            train_accuracy = 0
            for batch_train_idx, (xb, yb, gb) in enumerate(train_loader):
                xb, yb = xb.to(self.device), yb.long().to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                train_current_loss = loss.item()
                train_accuracy += (preds.argmax(dim=1) == yb).float().mean().item()
                train_current_accuracy = (preds.argmax(dim=1) == yb).float().mean().item()
                avg_train_accuracy = train_accuracy / len(train_loader)
                
                self.history["train"]["epoch"].append(epoch)
                self.history["train"]["batch"].append(batch_train_idx)
                self.history["train"]["loss"].append(loss.item())
                self.history["train"]["accuracy"].append((preds.argmax(dim=1) == yb).float().mean().item())
                
                
                # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
                if batch_train_idx % 3 == 0:
                    self.model.eval()
                    val_loss = 0
                    val_accuracy = 0
                    with torch.no_grad():
                        for batch_val_idx, (xb, yb, gb) in enumerate(val_loader):
                            xb, yb = xb.to(self.device), yb.long().to(self.device)
                            preds = self.model(xb)
                            loss = self.criterion(preds, yb)
                            val_loss += loss.item()
                            val_current_loss = loss.item()
                            val_accuracy += (preds.argmax(dim=1) == yb).float().mean().item()
                            val_current_accuracy = (preds.argmax(dim=1) == yb).float().mean().item()
                            
                            self.history["val"]["epoch"].append(epoch)
                            self.history["val"]["batch"].append(batch_val_idx)
                            self.history["val"]["loss"].append(loss.item())
                            self.history["val"]["accuracy"].append((preds.argmax(dim=1) == yb).float().mean().item())
                            self.history["val"]["batch_train_idx"].append(batch_train_idx)
                        
                    avg_val_loss = val_loss / len(val_loader)
                    avg_val_accuracy = val_accuracy / len(val_loader)
                    if avg_val_accuracy > self.best_val_model:
                        self.best_val_model = avg_val_accuracy
                        torch.save(self.model.state_dict(), f"./data/models/best_{self.run_id}_model.pth")
                    self.model.train()  # switch back to train mode
            
                avg_train_loss = total_loss / (batch_train_idx + 1) #len(train_loader)
                    
                # self.scheduler.step()
                # print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_accuracy:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_accuracy:.4f}")
                print(f"Epoch {epoch+1}: Train Loss={train_current_loss:.4f}, Train Acc={train_current_accuracy:.4f}, Val Loss={val_current_loss:.4f}, Val Acc={val_current_accuracy:.4f}")
        self._save_logs(self.history, f"./data/results/{self.run_id}_training_log.json")
            

    def predict(self, dataloader):
        self.model.eval()
        preds_all = []
        with torch.no_grad():
            for xb, yb, gb in dataloader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                preds = torch.argmax(logits, dim=1)
                preds_all.append(preds.cpu())
        return torch.cat(preds_all).numpy()


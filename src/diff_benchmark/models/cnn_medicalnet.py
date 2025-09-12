import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm

from diff_benchmark.models.base import TorchAbstractModel

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
    
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()

class ResNet3DModel(TorchAbstractModel, nn.Module):
    def __init__(self, input_volumes=1, num_classes=2, device="cuda", **kwargs):
        super(ResNet3DModel, self).__init__()
        self.device = device
        self.run_id = kwargs.get("run_id", "unnamed_run")
        self.epochs = kwargs.get("epochs", 100)

        self.model = resnet10(num_classes=num_classes).to(device)
        
        # Load pretrained if provided
        pretrain_path = kwargs.get("pretrain_path", None)
        if pretrain_path:
            print(f"Loading pretrained weights from {pretrain_path}")
            state_dict = torch.load(pretrain_path, map_location=device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
            
        self.criterion = nn.CrossEntropyLoss()
        lr = kwargs.get("learning_rate", 1e-5)
        weight_decay = kwargs.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=lr, weight_decay=weight_decay)
        
        self.best_val_model = 0
        self.history = {
            "train": {"epoch": [], "batch": [], "loss": [], "accuracy": []},
            "val": {
                "epoch": [],
                "batch": [],
                "loss": [],
                "accuracy": [],
                "batch_train_idx": [],
            },
        }
    
    def _dataloader_to_numpy(self, dataloader):
        x, y, g = [], [], []
        for xb, yb, gb in dataloader:
            x.append(xb.numpy())
            y.append(yb.numpy())
            g.append(gb.numpy())
        return np.concatenate(x), np.concatenate(y), np.concatenate(g)

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

        train_loader_new = DataLoader(
            train_subset, batch_size=train_loader.batch_size, shuffle=True
        )
        val_loader_new = DataLoader(
            val_subset, batch_size=train_loader.batch_size, shuffle=False
        )
        return train_loader_new, val_loader_new

    def _save_logs(self, history, save_path):
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
        
    def fit(self, dataloader):
        print(f"Device: {self.device}")
        self.model.train()
        patience_counter = 0
        patience = 10
        train_loader, val_loader = self._train_val_loader_split(dataloader)
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0
            train_accuracy = 0
            for batch_train_idx, (xb, yb, _) in enumerate(train_loader):
                xb, yb = xb.to(self.device), yb.long().to(self.device)
                self.optimizer.zero_grad()
                xb = xb.unsqueeze(1) 
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                train_current_loss = loss.item()
                train_accuracy += (preds.argmax(dim=1) == yb).float().mean().item()
                train_current_accuracy = (
                    (preds.argmax(dim=1) == yb).float().mean().item()
                )
                # avg_train_accuracy = train_accuracy / len(train_loader) # NOT USED FOR THE MOMENT

                self.history["train"]["epoch"].append(epoch)
                self.history["train"]["batch"].append(batch_train_idx)
                self.history["train"]["loss"].append(loss.item())
                self.history["train"]["accuracy"].append(
                    (preds.argmax(dim=1) == yb).float().mean().item()
                )

                # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
                if batch_train_idx % 3 == 0:
                    self.model.eval()
                    val_loss = 0
                    val_accuracy = 0
                    with torch.no_grad():
                        for batch_val_idx, (xb, yb, _) in enumerate(val_loader):
                            xb, yb = xb.to(self.device), yb.long().to(self.device)
                            xb = xb.unsqueeze(1) 
                            preds = self.model(xb)
                            loss = self.criterion(preds, yb)
                            val_loss += loss.item()
                            val_current_loss = loss.item()
                            val_accuracy += (
                                (preds.argmax(dim=1) == yb).float().mean().item()
                            )
                            val_current_accuracy = (
                                (preds.argmax(dim=1) == yb).float().mean().item()
                            )

                            self.history["val"]["epoch"].append(epoch)
                            self.history["val"]["batch"].append(batch_val_idx)
                            self.history["val"]["loss"].append(loss.item())
                            self.history["val"]["accuracy"].append(
                                (preds.argmax(dim=1) == yb).float().mean().item()
                            )
                            self.history["val"]["batch_train_idx"].append(
                                batch_train_idx
                            )

                    # avg_val_loss = val_loss / len(val_loader)   # NOT USED FOR THE MOMENT
                    avg_val_accuracy = val_accuracy / len(val_loader)
                    if avg_val_accuracy > self.best_val_model:
                        patience_counter = 0
                        save_path = Path("./data/models") / f"{self.run_id}_best.pth"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        self.best_val_model = avg_val_accuracy
                        torch.save(
                            self.model.state_dict(),
                            save_path,
                        )
                    else:
                        patience_counter += 1

                    # Early stopping trigger
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                    self.model.train()  # switch back to train mode

                # avg_train_loss = total_loss / (batch_train_idx + 1)  # len(train_loader) #NOT USED FOR THE MOMENT

                # self.scheduler.step()
                # print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_accuracy:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_accuracy:.4f}")
                print(
                    f"Epoch {epoch+1}: Train Loss={train_current_loss:.4f}, Train Acc={train_current_accuracy:.4f}, Val Loss={val_current_loss:.4f}, Val Acc={val_current_accuracy:.4f}"
                )
        self._save_logs(self.history, f"./data/results/{self.run_id}_training_log.json")
    
    def predict(self, dataloader):
        self.model.eval()
        preds_all = []
        with torch.no_grad():
            for xb, _, _ in dataloader:
                xb = xb.to(self.device)
                xb = xb.unsqueeze(1) 
                logits = self.model(xb)
                preds = torch.argmax(logits, dim=1)
                preds_all.append(preds.cpu())
        return torch.cat(preds_all).numpy()

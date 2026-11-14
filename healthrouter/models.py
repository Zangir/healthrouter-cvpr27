"""Model construction and deterministic inference helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_transform(size: int, variant: str = "standard"):
    ops = [transforms.Resize((size, size))]
    if variant == "grayscale":
        ops.append(transforms.Grayscale(num_output_channels=3))
    elif variant == "blur":
        ops.append(transforms.GaussianBlur(kernel_size=7, sigma=2.0))
    elif variant == "low_resolution":
        ops = [transforms.Resize((64, 64)), transforms.Resize((size, size))]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class FileDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = files, labels, transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        return self.transform(image), self.labels[index]


class ArrayDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


def convnext_model(outputs: int, checkpoint: str | Path | None = None,
                   imagenet: bool = False):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if imagenet else None
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(768, outputs)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model.to(device()).eval()


def predict(model, dataset, batch_size: int = 32, multiclass: bool = False):
    scores, labels = [], []
    with torch.inference_mode():
        for images, target in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            logits = model(images.to(device()))
            out = logits.softmax(1) if multiclass else logits.sigmoid().squeeze(1)
            scores.append(out.cpu().numpy())
            labels.append(np.asarray(target))
    return np.concatenate(scores), np.concatenate(labels)


def imagenet_features(files: list[str], size: int = 384, batch_size: int = 32):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device()).eval()
    dataset = FileDataset(files, np.zeros(len(files)), image_transform(size))
    features = []
    with torch.inference_mode():
        for images, _ in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            pooled = model.avgpool(model.features(images.to(device()))).flatten(1)
            features.append(pooled.cpu().numpy())
    return np.concatenate(features)

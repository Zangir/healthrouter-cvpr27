"""Frozen representation packages: one loader per released encoder + preprocessing."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .models import ArrayDataset, FileDataset, device, image_transform

RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"

# name -> (huggingface repo or "convnext", revision, parameter count in millions)
PACKAGES = {
    "imagenet_convnext": ("convnext", None, 28.6),
    "dinov2_base": ("facebook/dinov2-base", None, 86.6),
    "dinov2_large": ("facebook/dinov2-large", None, 304.4),
    "rad_dino": ("microsoft/rad-dino", RAD_DINO_REVISION, 86.6),
}


def _convnext_features(loader) -> np.ndarray:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device()).eval()
    out = []
    with torch.inference_mode():
        for images, _ in loader:
            pooled = model.avgpool(model.features(images.to(device()))).flatten(1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out)


def _hf_features(name: str, revision, images, batch: int = 8) -> np.ndarray:
    """Pool a Hugging Face vision encoder with its own pinned upstream processor."""
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(name, revision=revision)
    model = AutoModel.from_pretrained(name, revision=revision).to(device()).eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            chunk = [im() if callable(im) else im for im in images[start:start + batch]]
            inputs = processor(images=chunk, return_tensors="pt").to(device())
            output = model(**inputs)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out)


def file_features(package: str, files: list[str], convnext_size: int = 384,
                  batch: int = 32) -> np.ndarray:
    """Frozen features for image files under the package's declared preprocessing."""
    repo, revision, _ = PACKAGES[package]
    if repo == "convnext":
        dataset = FileDataset(files, np.zeros(len(files)), image_transform(convnext_size))
        return _convnext_features(DataLoader(dataset, batch_size=batch, num_workers=0))
    lazy = [(lambda p=p: Image.open(p).convert("RGB")) for p in files]
    return _hf_features(repo, revision, lazy, batch=min(batch, 8))


def array_features(package: str, arrays: np.ndarray, batch: int = 64) -> np.ndarray:
    """Frozen features for an in-memory uint8 HWC image stack (PatchCamelyon)."""
    repo, revision, _ = PACKAGES[package]
    if repo == "convnext":
        from torchvision import transforms
        from .models import MEAN, STD
        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                 transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        loader = DataLoader(ArrayDataset(arrays, np.zeros(len(arrays)), tf),
                            batch_size=batch, num_workers=0)
        return _convnext_features(loader)
    lazy = [(lambda a=a: Image.fromarray(a)) for a in arrays]
    return _hf_features(repo, revision, lazy, batch=8)

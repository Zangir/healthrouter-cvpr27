"""Route scores for four gates under five acquisition conditions.

Every gate reads the same 224-px square pixels per condition, decoded once from the
originals and cached outside the repo. Gates: the supplied supervised ConvNeXt gate, a
frozen ImageNet probe, the same probe trained with grayscale augmentation, and zero-shot
BiomedCLIP text-prompt routing (no routing labels at all).
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import external, pools
from .models import MEAN, STD, convnext_model, device

CONDITIONS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]
PROMPTS = ["a chest X-ray radiograph", "an endoscopy image of the colon",
           "an H&E stained histopathology patch", "a color fundus photograph of the retina",
           "a mammogram"]
SIZE = 224


def _condition(image: Image.Image, condition: str) -> np.ndarray:
    """Mirror the released transforms: resize first, then the perturbation."""
    if condition == "low_resolution":
        out = image.resize((64, 64), Image.BILINEAR).resize((SIZE, SIZE), Image.BILINEAR)
    else:
        out = image.resize((SIZE, SIZE), Image.BILINEAR)
    if condition == "grayscale":
        out = out.convert("L").convert("RGB")
    elif condition == "blur":
        out = out.filter(ImageFilter.GaussianBlur(radius=2.0))
    elif condition == "jpeg":
        buffer = io.BytesIO()
        out.save(buffer, format="JPEG", quality=10)
        out = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return np.asarray(out, np.uint8)


def condition_arrays(name: str, items, conditions=CONDITIONS) -> dict[str, np.ndarray]:
    key = hashlib.sha256(f"{name}{len(items)}".encode()).hexdigest()[:10]
    paths = {c: pools.cache_dir() / f"gate_{name}_{c}_{key}.npy" for c in conditions}
    if all(p.exists() for p in paths.values()):
        return {c: np.load(p, mmap_mode="r") for c, p in paths.items()}
    tmp = {c: p.with_suffix(".partial.npy") for c, p in paths.items()}
    arrays = {c: np.lib.format.open_memmap(tmp[c], mode="w+", dtype=np.uint8,
                                           shape=(len(items), SIZE, SIZE, 3)) for c in conditions}
    for i, item in enumerate(items):
        image = external.decode(item)
        for c in conditions:
            arrays[c][i] = _condition(image, c)
    for c, p in paths.items():  # memory-mapped: RAM stays flat; rename when complete
        arrays[c].flush()
        del arrays[c]
        tmp[c].rename(p)
    return {c: np.load(p, mmap_mode="r") for c, p in paths.items()}


def _tensor(pixels: np.ndarray, mean, std) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(pixels)).permute(0, 3, 1, 2).float().div(255.0)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def _batched(fn, pixels, mean, std, batch=32) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for s in range(0, len(pixels), batch):
            out.append(fn(_tensor(pixels[s:s + batch], mean, std).to(device())).float().cpu().numpy())
    return np.concatenate(out)


def supervised_gate(root: str, pixels: np.ndarray) -> np.ndarray:
    model = convnext_model(5, Path(root) / "router_convnext_tiny.pt")
    return _batched(lambda x: model(x).softmax(1), pixels, MEAN, STD)


def imagenet_features(pixels: np.ndarray) -> np.ndarray:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device()).eval()
    return _batched(lambda x: model.avgpool(model.features(x)).flatten(1), pixels, MEAN, STD)


def probe_gate(train_x, train_y, seed: int = 0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3_000,
                                                              random_state=seed)).fit(train_x, train_y)


def zero_shot_gate(pixels_by_condition: dict) -> dict:
    import open_clip
    name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _ = open_clip.create_model_from_pretrained(name)
    tokenizer = open_clip.get_tokenizer(name)
    model = model.to(device()).eval()
    with torch.inference_mode():
        text = model.encode_text(tokenizer(PROMPTS).to(device()))
        text = text / text.norm(dim=-1, keepdim=True)
        scale = model.logit_scale.exp()

        def route(x):
            image = model.encode_image(x)
            image = image / image.norm(dim=-1, keepdim=True)
            return (scale * image @ text.T).softmax(1)
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        return {c: _batched(route, p, mean, std) for c, p in pixels_by_condition.items()}

"""Frozen encoders for the matched-resolution grid.

Every package receives the same square uint8 pixels at the same resolution; only the
weights and the package's own channel normalisation differ. Parameter counts are measured
from the loaded vision tower, never quoted.
"""
from __future__ import annotations

import numpy as np
import torch

from .models import device

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"

# name -> (pretraining domain, objective, loader key, normalisation)
PACKAGES = {
    "imagenet_convnext": ("natural images", "supervised", "convnext", IMAGENET),
    "dinov2_base": ("natural images", "self-distillation", "hf:facebook/dinov2-base", IMAGENET),
    "dinov2_large": ("natural images", "self-distillation", "hf:facebook/dinov2-large", IMAGENET),
    "siglip2_base": ("web image-text", "sigmoid contrastive", "siglip:google/siglip2-base-patch16-224",
                     ((0.5,) * 3, (0.5,) * 3)),
    "biomedclip": ("biomedical figures (PMC)", "contrastive", "biomedclip", CLIP),
    "rad_dino": ("chest radiographs", "self-distillation", "hf:microsoft/rad-dino",
                 ((0.5307,) * 3, (0.2583,) * 3)),
    "phikon": ("histopathology", "iBOT", "hfcls:owkin/phikon", IMAGENET),
    "phikon_v2": ("histopathology", "self-distillation", "hfcls:owkin/phikon-v2", IMAGENET),
    "retfound": ("retinal photographs", "MAE", "retfound", IMAGENET),
    "mammo_clip": ("mammograms", "contrastive", "mammoclip",
                   ((0.3089279,) * 3, (0.2505356,) * 3)),  # released pre-train config
}


def _hf(name: str):
    from transformers import AutoModel
    revision = RAD_DINO_REVISION if name == "microsoft/rad-dino" else None
    model = AutoModel.from_pretrained(name, revision=revision)

    def run(x, cls_only=False):
        out = model(pixel_values=x)
        pooled = getattr(out, "pooler_output", None)
        return out.last_hidden_state[:, 0] if cls_only or pooled is None else pooled
    return model, run


def _load(key: str):
    if key == "convnext":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        return model, lambda x: model.avgpool(model.features(x)).flatten(1)
    if key.startswith("hf:"):
        return _hf(key[3:])
    if key.startswith("hfcls:"):  # model cards for Phikon use the CLS token
        model, run = _hf(key[6:])
        kwargs = {"interpolate_pos_encoding": True} if "phikon-v2" not in key else {}
        return model, lambda x: model(pixel_values=x, **kwargs).last_hidden_state[:, 0]
    if key.startswith("siglip:"):
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained(key[7:])
        return model, lambda x: model(pixel_values=x, interpolate_pos_encoding=True).pooler_output
    if key == "biomedclip":
        import open_clip
        from timm.layers import Format
        clip, _ = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        trunk = clip.visual.trunk
        trunk.dynamic_img_size = True
        trunk.patch_embed.strict_img_size = False
        trunk.patch_embed.flatten = False
        trunk.patch_embed.output_fmt = Format.NHWC
        return clip.visual, lambda x: clip.visual(x)
    if key == "retfound":
        import timm
        model = timm.create_model("hf-hub:bitfount/RETFound_MAE", pretrained=True,
                                  num_classes=0, dynamic_img_size=True, global_pool="avg",
                                  fc_norm=False)  # learned final norm, then token mean
        return model, lambda x: model(x)
    if key == "mammoclip":
        from .mammoclip import load_image_encoder
        model = load_image_encoder()
        return model, lambda x: model(x)
    raise KeyError(key)


def parameter_count(model) -> float:
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 1)


def embed(package: str, pixels: np.ndarray, batch: int = 16) -> tuple[np.ndarray, float]:
    """Features for uint8 (n, s, s, 3) pixels; returns (features, params in millions)."""
    _, _, key, (mean, std) = PACKAGES[package]
    model, run = _load(key)
    model = model.to(device()).eval()
    mean_t = torch.tensor(mean, device=device()).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device()).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for start in range(0, len(pixels), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(pixels[start:start + batch]))
            x = chunk.to(device()).permute(0, 3, 1, 2).float().div(255.0)
            out.append(run((x - mean_t) / std_t).float().cpu().numpy())
    params = parameter_count(model)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return np.concatenate(out), params

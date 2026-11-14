"""Is the chest gain of RAD-DINO a pretraining effect or an input-resolution effect?

Shenzhen pool, fixed folds and tuned head (grid2.cross_validate). Four packages are scored
(a) under each package's released processor (RAD-DINO 518 px, DINOv2 224 px centre crop,
ConvNeXt 384 px), exactly as in the released-preprocessing grid, and (b) at matched square
resolutions 224, 336, 448 and 518 px.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import roc_auc_score

from . import encoders, packages, pools
from .data import shenzhen
from .grid2 import cross_validate
from .metrics import paired_metric_difference, rounded_ci

SIZES = (224, 336, 448, 518)
NAMES = ["imagenet_convnext", "dinov2_base", "dinov2_large", "rad_dino"]
PAIRS = [("rad_dino", "dinov2_base"), ("rad_dino", "imagenet_convnext"),
         ("dinov2_large", "dinov2_base")]


def _auc(y, p):
    return roc_auc_score(y, p[:, 1])


def _score(features: dict, y: np.ndarray) -> dict:
    probs = {n: cross_validate(x, y, "auroc")["prob"] for n, x in features.items()}
    block = {"oof_auroc": {n: round(float(_auc(y, p)), 5) for n, p in probs.items()}}
    block["contrasts"] = {f"{a}_minus_{b}": rounded_ci(paired_metric_difference(
        y, probs[a], probs[b], _auc, n_boot=2_000), 5) for a, b in PAIRS}
    return block


def evaluate(root: str) -> dict:
    start = time.time()
    files, y = shenzhen(root)
    out = {}
    released = {n: encoders.file_features(n, files, convnext_size=384, batch=8) for n in NAMES}
    out["released_processor"] = _score(released, y)
    out["released_processor"]["input_px"] = {"imagenet_convnext": 384, "dinov2_base": 224,
                                             "dinov2_large": 224, "rad_dino": 518}
    cached = pools.build("chest_xray", files, "files", sizes=SIZES)
    for size in SIZES:
        feats = {n: packages.embed(n, cached[size], batch=4 if size >= 448 else 8)[0]
                 for n in NAMES}
        out[f"matched_{size}"] = _score(feats, y)
        print(f"  chest matched {size}: {out[f'matched_{size}']['oof_auroc']}", flush=True)
    return {"provenance": "run", "pool": "Shenzhen, 662 images", "protocol": __doc__.strip(),
            "variants": out, "runtime_seconds": round(time.time() - start, 2)}

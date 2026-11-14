"""Paired five-fold TB probes for generic and radiology-aligned encoders."""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data import shenzhen, tb_folds
from .metrics import bootstrap_auc, paired_auc_difference, rounded_ci
from .models import device, imagenet_features

RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"


def _rad_dino_features(files: list[str]) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino",
                                                   revision=RAD_DINO_REVISION)
    model = AutoModel.from_pretrained("microsoft/rad-dino",
                                      revision=RAD_DINO_REVISION).to(device()).eval()
    features = []
    with torch.inference_mode():
        for start in range(0, len(files), 8):
            images = [Image.open(path).convert("RGB") for path in files[start:start + 8]]
            inputs = processor(images=images, return_tensors="pt").to(device())
            output = model(**inputs)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            features.append(pooled.cpu().numpy())
    return np.concatenate(features)


def _cross_validated(features: np.ndarray, labels: np.ndarray):
    folds, oof = [], np.zeros(len(labels), dtype=float)
    for train, test in tb_folds(labels):
        head = LogisticRegression(max_iter=2_000).fit(features[train], labels[train])
        probability = head.predict_proba(features[test])[:, 1]
        oof[test] = probability
        folds.append(float(roc_auc_score(labels[test], probability)))
    return np.asarray(folds), oof


def _summary(folds: np.ndarray, labels: np.ndarray, oof: np.ndarray):
    return {
        "fold_aurocs": [round(float(x), 5) for x in folds],
        "fold_mean": round(float(folds.mean()), 5),
        "fold_std": round(float(folds.std()), 5),
        "oof": rounded_ci(bootstrap_auc(labels, oof), 5),
    }


def evaluate(root: str) -> dict:
    start = time.time()
    files, labels = shenzhen(root)
    generic = imagenet_features(files, size=384)
    radiology = _rad_dino_features(files)
    generic_folds, generic_oof = _cross_validated(generic, labels)
    radiology_folds, radiology_oof = _cross_validated(radiology, labels)
    difference = paired_auc_difference(labels, radiology_oof, generic_oof)
    return {
        "provenance": "run", "protocol": "paired five-fold out-of-fold evaluation",
        "n_images": len(files), "n_folds": 5, "rad_dino_revision": RAD_DINO_REVISION,
        "imagenet_convnext_probe": _summary(generic_folds, labels, generic_oof),
        "rad_dino_probe": _summary(radiology_folds, labels, radiology_oof),
        "paired_oof_auroc_difference": rounded_ci(difference, 5),
        "runtime_seconds": round(time.time() - start, 2),
    }

"""Cached frozen features, task registry, costs and the frozen cross-fitting splits."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ENCODERS = ["imagenet_convnext", "dinov2_base", "dinov2_large", "siglip2_base", "biomedclip",
            "rad_dino", "phikon", "phikon_v2", "retfound", "mammo_clip"]
ROUTER_ENCODER = "imagenet_convnext"
MED_TASKS = ["chest_xray", "pathology", "fundus", "mammography"]
NONMED_TASKS = ["eurosat", "dtd"]
METRIC = {"chest_xray": "auroc", "pathology": "auroc", "fundus": "qwk",
          "mammography": "auroc", "eurosat": "accuracy", "dtd": "accuracy"}
SPLIT_SEEDS = (0, 1, 2)
N_OUTER = 5


def cache_dir() -> Path:
    return Path(os.environ.get("HEALTHROUTER_CACHE", "~/.cache/healthrouter")).expanduser()


def feature_path(task: str, encoder: str, size: int = 224, condition: str | None = None) -> Path:
    if condition is None:
        return cache_dir() / "features" / f"{task}_{encoder}_{size}.npz"
    return cache_dir() / "features_external" / f"{task}_{encoder}_{size}_{condition}.npz"


def load(task: str, encoder: str, size: int = 224, condition: str | None = None):
    d = np.load(feature_path(task, encoder, size, condition))
    return d["x"].astype(np.float32), d["y"].astype(int)


def candidates(sizes=(224,)) -> list[tuple[str, int]]:
    return [(e, s) for s in sizes for e in ENCODERS]


def costs_ms(sizes=(224,)) -> np.ndarray:
    """Measured per-image extraction time (ms), mean over the four in-source pools."""
    out = []
    for e, s in candidates(sizes):
        per = []
        for t in MED_TASKS:
            d = np.load(feature_path(t, e, s))
            per.append(float(d["seconds"]) / len(d["y"]) * 1000.0)
        out.append(np.mean(per))
    return np.asarray(out)


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        members = rng.permutation(np.where(y == c)[0])
        offset = rng.randint(k)
        for j, i in enumerate(members):
            folds[(j + offset) % k].append(i)
    return [np.sort(np.asarray(f, dtype=int)) for f in folds]


def stratified_holdout(y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    hold = []
    for c in np.unique(y):
        members = rng.permutation(np.where(y == c)[0])
        hold.extend(members[:max(1, int(round(frac * len(members))))])
    hold = np.sort(np.asarray(hold, dtype=int))
    return np.setdiff1d(np.arange(len(y)), hold), hold


def outer_split(y: np.ndarray, seed: int, fold: int):
    """(inner_train, val, test) index arrays for one outer fold of one split seed."""
    folds = stratified_folds(y, N_OUTER, 1000 + seed)
    test = folds[fold]
    rest = np.setdiff1d(np.arange(len(y)), test)
    tr_rel, va_rel = stratified_holdout(y[rest], 0.2, 2000 + 10 * seed + fold)
    return rest[tr_rel], rest[va_rel], test

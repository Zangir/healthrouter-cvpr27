"""Decode-once caches of the fixed evaluation pools at matched square resolutions.

The cache holds derived pixels of restricted data, so it lives outside the repository
(``HEALTHROUTER_CACHE``) and is never committed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

SIZES = (224, 448)


def cache_dir() -> Path:
    path = Path(os.environ.get("HEALTHROUTER_CACHE", "~/.cache/healthrouter")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _square(image: Image.Image, size: int) -> np.ndarray:
    """Square resize (no crop, so no lesion is cut away); PIL antialiases downscales."""
    return np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC), np.uint8)


def build(name: str, items, kind: str, sizes=SIZES) -> dict[int, np.ndarray]:
    """Return {size: uint8 array (n, size, size, 3)} for one pool, caching to disk."""
    cache = cache_dir()
    key = hashlib.sha256(json.dumps([name, kind, len(items),
                                     [str(x) for x in items[:5]] if kind == "files" else
                                     int(np.asarray(items[:50]).sum())]).encode()).hexdigest()[:12]
    paths = {s: cache / f"pool_{name}_{s}_{key}.npy" for s in sizes}
    if all(p.exists() for p in paths.values()):
        return {s: np.load(p, mmap_mode="r") for s, p in paths.items()}
    tmp = {s: p.with_suffix(".partial.npy") for s, p in paths.items()}
    arrays = {s: np.lib.format.open_memmap(tmp[s], mode="w+", dtype=np.uint8,
                                           shape=(len(items), s, s, 3)) for s in sizes}
    for i, item in enumerate(items):
        image = Image.open(item) if kind == "files" else Image.fromarray(np.asarray(item))
        image.load()
        for s in sizes:
            arrays[s][i] = _square(image, s)
        if i % 250 == 0:
            print(f"  cache {name}: {i}/{len(items)}", flush=True)
    for s, p in paths.items():  # memory-mapped writes keep RAM flat; rename when complete
        arrays[s].flush()
        del arrays[s]
        tmp[s].rename(p)
    return {s: np.load(p, mmap_mode="r") for s, p in paths.items()}


def grid_pools(root: str) -> dict:
    """The four Q1 pools, identical items to the released-preprocessing grid."""
    from .grid import _domain_pools
    pools = _domain_pools(root)
    for name, pool in pools.items():
        pool["cached"] = build(name, pool["items"], pool["kind"])
    return pools

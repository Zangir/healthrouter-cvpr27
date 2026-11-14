#!/usr/bin/env python3
"""Checkpointed frozen-feature extraction for the routing experiments.

  --what external   the four classification external cohorts (Montgomery, Camelyon17-WILDS
                    OOD hospital, IDRiD, CMMD) from the cached 224 px gate images, under the five
                    acquisition conditions; endoscopy (segmentation lane) is skipped
  --what nonmedical EuroSAT and DTD class-balanced subsamples (square 224 px resize)

Each (encoder, condition/task) unit is written atomically and skipped if present, so the
script can be re-run in short guarded chunks. --budget stops before starting a unit that
would likely exceed the remaining time.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
from PIL import Image

from healthrouter import packages
from healthrouter.collection import data

LANES = {0: "chest_xray", 2: "pathology", 3: "fundus", 4: "mammography"}
CONDITIONS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]
NONMED_ROOT = Path(__file__).resolve().parents[3] / "external_data" / "nonmedical"


def external_units(encoders, conditions):
    items = np.load(data.cache_dir() / "dispatch_items.npz")
    lane = items["lane_ids"]
    out_dir = data.cache_dir() / "features_external"
    out_dir.mkdir(exist_ok=True)
    for cond in conditions:
        imgs = None
        for enc in encoders:
            targets = {t: out_dir / f"{t}_{enc}_224_{cond}.npz" for t in LANES.values()}
            if all(p.exists() for p in targets.values()):
                continue
            if imgs is None:
                path = glob.glob(str(data.cache_dir() / f"gate_external5_{cond}_*.npy"))[0]
                imgs = np.load(path, mmap_mode="r")
            yield (enc, cond), lambda enc=enc, imgs=imgs, targets=targets: _ext(enc, imgs, lane, items, targets)


def _ext(enc, imgs, lane, items, targets):
    keep = np.isin(lane, list(LANES))
    t0 = time.time()
    feats, params = packages.embed(enc, np.asarray(imgs[keep]), batch=16)
    secs = time.time() - t0
    lane_k = lane[keep]
    for lid, task in LANES.items():
        x = feats[lane_k == lid]
        y = items[f"y_{task}"].astype(int)
        assert len(x) == len(y), (task, len(x), len(y))
        tmp = targets[task].with_suffix(".partial.npz")
        np.savez(tmp, x=x, y=y, params=params, seconds=secs * len(x) / len(feats))
        tmp.rename(targets[task])


def _square(img, size=224):
    return np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), np.uint8)


def nonmed_pool(task):
    path = data.cache_dir() / f"pool_{task}_224.npz"
    if path.exists():
        d = np.load(path)
        return d["x"], d["y"]
    rng = np.random.RandomState(0)
    if task == "eurosat":
        root = NONMED_ROOT / "eurosat" / "2750"
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        per = 200
        files = {c: sorted(glob.glob(str(root / c / "*.jpg"))) for c in classes}
    else:
        root = NONMED_ROOT / "dtd" / "dtd" / "images"
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        per = 50
        files = {c: sorted(glob.glob(str(root / c / "*.jpg"))) for c in classes}
    xs, ys = [], []
    for k, c in enumerate(classes):
        pick = rng.choice(len(files[c]), per, replace=False)
        for i in sorted(pick):
            xs.append(_square(Image.open(files[c][i])))
            ys.append(k)
    x, y = np.stack(xs), np.asarray(ys)
    np.savez(path, x=x, y=y, classes=np.asarray(classes))
    return x, y


def nonmed_units(encoders, tasks):
    for task in tasks:
        for enc in encoders:
            target = data.feature_path(task, enc, 224)
            if target.exists():
                continue
            yield (enc, task), lambda enc=enc, task=task, target=target: _nm(enc, task, target)


def _nm(enc, task, target):
    x, y = nonmed_pool(task)
    t0 = time.time()
    feats, params = packages.embed(enc, x, batch=16)
    tmp = target.with_suffix(".partial.npz")
    np.savez(tmp, x=feats, y=y, params=params, seconds=time.time() - t0)
    tmp.rename(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--what", choices=["external", "nonmedical"], required=True)
    p.add_argument("--encoders", nargs="+", default=data.ENCODERS)
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--tasks", nargs="+", default=data.NONMED_TASKS)
    p.add_argument("--budget", type=float, default=520.0, help="seconds")
    a = p.parse_args()
    start = time.time()
    units = (external_units(a.encoders, a.conditions) if a.what == "external"
             else nonmed_units(a.encoders, a.tasks))
    last = 0.0
    for name, fn in units:
        if time.time() - start + 1.3 * last > a.budget:
            print("budget reached; re-run to continue", flush=True)
            return
        t0 = time.time()
        fn()
        last = time.time() - t0
        print(f"done {name} {last:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

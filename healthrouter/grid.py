"""Encoder x modality grid: which frozen representation package wins where."""
from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import encoders
from .data import (aptos_split, cbis_split, nonoverlap_mask, pcam_arrays, shenzhen,
                   stratified_folds, tb_folds)
from .metrics import paired_metric_difference, qwk, rounded_ci

PCAM_SUBSAMPLE = 2_000
REFERENCE = "imagenet_convnext"
ARCHITECTURE_CONTROL = "dinov2_base"


def _pcam_pool(root: str, seed: int = 2026):
    """Balanced subsample; reads only the chosen patches from disk (same items as before)."""
    import h5py
    from .data import data_dir
    base = data_dir(root)
    with h5py.File(base / "camelyonpatch_level_2_split_test_y.h5") as handle:
        labels = handle["y"][:].reshape(-1).astype(np.float32)
    rng = np.random.RandomState(seed)
    per_class = PCAM_SUBSAMPLE // 2
    chosen = np.concatenate([rng.choice(np.where(labels == c)[0], per_class, replace=False)
                             for c in (0, 1)])
    chosen.sort()
    with h5py.File(base / "camelyonpatch_level_2_split_test_x.h5") as handle:
        images = handle["x"][chosen]
    return images, labels[chosen].astype(int)


def _domain_pools(root: str) -> dict:
    """Fixed, encoder-independent evaluation pools with their native endpoints."""
    chest_files, chest_y = shenzhen(root)
    aptos_train, _, aptos_files, aptos_y = aptos_split(root)
    aptos_keep = nonoverlap_mask(aptos_train, aptos_files)
    cbis_train, _, cbis_files, cbis_y = cbis_split(root)
    cbis_keep = nonoverlap_mask(cbis_train, cbis_files)
    pcam_images, pcam_y = _pcam_pool(root)
    return {
        "chest_xray": {"kind": "files", "items": chest_files, "y": chest_y,
                       "metric": "auroc", "convnext_size": 384,
                       "pool": "Shenzhen, all images"},
        "pathology": {"kind": "arrays", "items": pcam_images, "y": pcam_y,
                      "metric": "auroc", "convnext_size": 224,
                      "pool": f"PatchCamelyon official test, balanced {PCAM_SUBSAMPLE}-patch subsample"},
        "fundus": {"kind": "files",
                   "items": [f for f, k in zip(aptos_files, aptos_keep) if k],
                   "y": aptos_y[aptos_keep].astype(int), "metric": "qwk",
                   "convnext_size": 384,
                   "pool": "APTOS mirror holdout, exact-deduplicated"},
        "mammography": {"kind": "files",
                        "items": [f for f, k in zip(cbis_files, cbis_keep) if k],
                        "y": cbis_y[cbis_keep].astype(int), "metric": "auroc",
                        "convnext_size": 512,
                        "pool": "CBIS-DDSM local mirror test, exact-deduplicated"},
    }


def _cross_validate(features: np.ndarray, y: np.ndarray, metric: str):
    """Same fixed folds and logistic head for every package; only features change."""
    folds, oof = [], np.zeros(len(y), dtype=float if metric == "auroc" else int)
    splitter = tb_folds(y) if len(np.unique(y)) == 2 else stratified_folds(y)
    for train, test in splitter:
        head = LogisticRegression(max_iter=2_000).fit(features[train], y[train])
        if metric == "auroc":
            prediction = head.predict_proba(features[test])[:, 1]
            folds.append(float(roc_auc_score(y[test], prediction)))
        else:
            prediction = head.predict(features[test])
            folds.append(float(qwk(y[test], prediction)))
        oof[test] = prediction
    return np.asarray(folds), oof


def evaluate(root: str) -> dict:
    start = time.time()
    pools = _domain_pools(root)
    domains: dict = {}
    for name, pool in pools.items():
        scorer = roc_auc_score if pool["metric"] == "auroc" else qwk
        oof_by_package, entries = {}, {}
        for package in encoders.PACKAGES:
            if pool["kind"] == "files":
                features = encoders.file_features(package, pool["items"],
                                                  convnext_size=pool["convnext_size"])
            else:
                features = encoders.array_features(package, pool["items"])
            folds, oof = _cross_validate(features, pool["y"], pool["metric"])
            oof_by_package[package] = oof
            entries[package] = {
                "parameters_millions": encoders.PACKAGES[package][2],
                "fold_scores": [round(float(x), 5) for x in folds],
                "fold_mean": round(float(folds.mean()), 5),
                "fold_std": round(float(folds.std()), 5),
                "oof_score": round(float(scorer(pool["y"], oof)), 5),
            }
            print(f"  {name}/{package}: oof={entries[package]['oof_score']}")
        best = max(entries, key=lambda k: entries[k]["oof_score"])
        contrasts = {}
        for target, baseline in [("rad_dino", REFERENCE),
                                 ("rad_dino", ARCHITECTURE_CONTROL),
                                 ("dinov2_large", ARCHITECTURE_CONTROL)]:
            contrasts[f"{target}_minus_{baseline}"] = rounded_ci(
                paired_metric_difference(pool["y"], oof_by_package[target],
                                        oof_by_package[baseline], scorer), 5)
        domains[name] = {"metric": pool["metric"], "n_items": int(len(pool["y"])),
                         "pool": pool["pool"], "packages": entries,
                         "best_package": best, "paired_contrasts": contrasts}
    return {"provenance": "run", "protocol": "shared seed-42 five-fold logistic probe; "
            "only the frozen representation package changes",
            "reference_package": REFERENCE, "architecture_control": ARCHITECTURE_CONTROL,
            "rad_dino_revision": encoders.RAD_DINO_REVISION,
            "domains": domains,
            "best_package_by_domain": {k: v["best_package"] for k, v in domains.items()},
            "distinct_winning_packages": len({v["best_package"] for v in domains.values()}),
            "runtime_seconds": round(time.time() - start, 2)}

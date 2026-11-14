"""Inputs from a modality with no lane: does the closed-set gate dispatch them anyway?

300 breast-ultrasound images (BUSI) and 300 dermoscopy images (HAM10000) are shown to every
gate. We report how many are dispatched at the threshold calibrated in abstain.py and to
which lane, and compare max-softmax rejection with a k-nearest-neighbour distance detector
on frozen ImageNet features, thresholded to keep 95% of the in-scope calibration half.
"""
from __future__ import annotations

import glob
import time

import numpy as np

from . import external, gates, pools
from .abstain import split_halves
from .data import router_sources

K, KEEP = 10, 0.95


def ood_items() -> dict:
    root = external.external_root()
    busi = sorted(glob.glob(str(root / "theshoaibme__busi" / "*" / "*.png")))
    ham = sorted(glob.glob(str(root / "BMN95__ham10000" / "HAM10000_images_part_1" / "*.jpg")))
    return {"breast_ultrasound": busi, "dermoscopy": ham}


def _knn_distance(bank: np.ndarray, query: np.ndarray) -> np.ndarray:
    bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    sims = query @ bank.T
    return 1.0 - np.sort(sims, axis=1)[:, -K:].mean(1)


def evaluate(root: str, abstention: dict) -> dict:
    start = time.time()
    data = np.load(pools.cache_dir() / "dispatch_items.npz")
    lane_ids, groups = data["lane_ids"], data["groups"]
    calib = split_halves(lane_ids, groups)
    src = external.sources(root)
    ext = gates.condition_arrays("external5", [x for l in external.LANES for x in src[l]["items"]])
    tr_files, tr_y, _, _, _ = router_sources(root)
    train = gates.condition_arrays("insource_train", tr_files, ["standard", "grayscale"])
    bank = gates.imagenet_features(train["standard"])
    ext_f = gates.imagenet_features(ext["standard"])
    probe = gates.probe_gate(bank, tr_y)
    out = {"n_per_modality": {}, "gates": {}, "knn": {}}
    ood = ood_items()
    ood_px = {m: gates.condition_arrays(f"ood_{m}", f, ["standard"])["standard"] for m, f in ood.items()}
    ood_scores = {"supervised": {m: gates.supervised_gate(root, px) for m, px in ood_px.items()},
                  "probe": {m: probe.predict_proba(gates.imagenet_features(px))
                            for m, px in ood_px.items()}}
    zs = gates.zero_shot_gate(ood_px)
    ood_scores["zero_shot_biomedclip"] = zs
    for gate, by_mod in ood_scores.items():
        tau = abstention["gates"][gate]["clean_to_clean"]["tau"]
        out["gates"][gate] = {"tau_clean": tau}
        for m, prob in by_mod.items():
            out["n_per_modality"][m] = int(len(prob))
            conf, route = prob.max(1), prob.argmax(1)
            sent = conf >= tau
            out["gates"][gate][m] = {
                "dispatched_fraction": round(float(sent.mean()), 5),
                "mean_confidence": round(float(conf.mean()), 5),
                "lane_counts_when_dispatched": np.bincount(route[sent], minlength=5).tolist()}
    d_ext = _knn_distance(bank, ext_f)
    threshold = float(np.quantile(d_ext[calib], KEEP))
    out["knn"] = {"k": K, "keep_target": KEEP, "threshold": round(threshold, 6),
                  "in_scope_test_kept": round(float((d_ext[~calib] <= threshold).mean()), 5)}
    for m, px in ood_px.items():
        d = _knn_distance(bank, gates.imagenet_features(px))
        out["knn"][f"{m}_rejected"] = round(float((d > threshold).mean()), 5)
    return {"provenance": "run", "lanes": external.LANES, **out,
            "runtime_seconds": round(time.time() - start, 2)}

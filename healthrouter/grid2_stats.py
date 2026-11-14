"""Score the matched-resolution grid from cached features.

Per (domain, size): seed-averaged out-of-fold score for every package, paired item
bootstrap contrasts with Holm correction, and a bootstrap probability that each package
is the best one in that domain.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import roc_auc_score

from . import packages, pools
from .grid2 import FOLD_SEEDS, cross_validate
from .metrics import qwk

N_BOOT = 2_000
METRIC = {"chest_xray": "auroc", "pathology": "auroc", "fundus": "qwk", "mammography": "auroc"}
IN_DOMAIN = {"chest_xray": "rad_dino", "pathology": "phikon_v2", "fundus": "retfound",
             "mammography": "mammo_clip"}
# (target, control) pairs: in-domain package vs its closest generic counterpart
CONTRASTS = [("rad_dino", "dinov2_base"), ("phikon_v2", "dinov2_large"),
             ("phikon", "dinov2_base"), ("retfound", "dinov2_large"),
             ("biomedclip", "siglip2_base"), ("dinov2_large", "dinov2_base"),
             ("mammo_clip", "imagenet_convnext")]


def _metric(name, y, prob):
    return roc_auc_score(y, prob[:, 1]) if name == "auroc" else qwk(y, prob.argmax(1))


def _boot_indices(y, n_boot, seed=2026):
    rng = np.random.RandomState(seed)
    classes = [np.where(y == c)[0] for c in np.unique(y)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(m, len(m), replace=True) for m in classes])


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def score_domain(domain: str, size: int, names) -> dict:
    feat_dir = pools.cache_dir() / "features"
    metric = METRIC[domain]
    probs, entries = {}, {}
    for package in names:
        path = feat_dir / f"{domain}_{package}_{size}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        y = data["y"].astype(int)
        cv = cross_validate(data["x"], y, metric)
        probs[package] = cv["prob"]
        entries[package] = {"oof_score": round(float(_metric(metric, y, cv["prob"])), 5),
                            "seed_scores": cv["seed_scores"],
                            "seed_sd": round(float(np.std(cv["seed_scores"])), 5),
                            "parameters_millions": float(data["params"]),
                            "median_C": float(np.median(cv["chosen_C"]))}
        print(f"  {domain}/{size}/{package}: {entries[package]['oof_score']}", flush=True)
    names = list(probs)
    wins = dict.fromkeys(names, 0)
    diffs = {f"{t}_minus_{c}": [] for t, c in CONTRASTS if t in probs and c in probs}
    for idx in _boot_indices(y, N_BOOT):
        values = {n: _metric(metric, y[idx], probs[n][idx]) for n in names}
        wins[max(values, key=values.get)] += 1
        for t, c in CONTRASTS:
            if f"{t}_minus_{c}" in diffs:
                diffs[f"{t}_minus_{c}"].append(values[t] - values[c])
    contrasts, pvals = {}, {}
    for key, values in diffs.items():
        t, c = key.split("_minus_")
        values = np.asarray(values)
        point = entries[t]["oof_score"] - entries[c]["oof_score"]
        pvals[key] = min(1.0, 2 * min((values <= 0).mean(), (values >= 0).mean()) + 1 / N_BOOT)
        contrasts[key] = {"value": round(float(point), 5),
                          "ci_low": round(float(np.percentile(values, 2.5)), 5),
                          "ci_high": round(float(np.percentile(values, 97.5)), 5),
                          "p_boot": round(float(pvals[key]), 5)}
    best = max(entries, key=lambda k: entries[k]["oof_score"])
    return {"metric": metric, "n_items": int(len(y)), "packages": entries,
            "best_package": best, "in_domain_package": IN_DOMAIN[domain],
            "prob_best": {n: round(wins[n] / N_BOOT, 4) for n in names},
            "contrasts": contrasts, "_pvals": pvals}


def evaluate(root: str, sizes=pools.SIZES) -> dict:
    start = time.time()
    names = list(packages.PACKAGES)
    out = {}
    for size in sizes:
        block = {d: score_domain(d, size, names) for d in METRIC}
        family = {f"{d}:{k}": v for d, b in block.items() for k, v in b.pop("_pvals").items()}
        adjusted = holm(family)
        for key, value in adjusted.items():
            d, k = key.split(":")
            block[d]["contrasts"][k]["p_holm"] = round(float(value), 5)
        out[str(size)] = {"domains": block,
                          "best_package_by_domain": {d: b["best_package"] for d, b in block.items()},
                          "distinct_winning_packages": len({b["best_package"] for b in block.values()})}
    return {"provenance": "run",
            "protocol": "matched square resolution for every package; standardised logistic "
                        "head, C in {0.001,0.01,0.1,1} by inner 3-fold CV; folds repeated over seeds "
                        f"{list(FOLD_SEEDS)} and probabilities averaged; paired stratified item "
                        f"bootstrap ({N_BOOT}) with Holm correction over all contrasts",
            "packages": {k: {"pretraining_domain": v[0], "objective": v[1]}
                         for k, v in packages.PACKAGES.items()},
            "sizes": out, "runtime_seconds": round(time.time() - start, 2)}

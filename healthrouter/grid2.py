"""Matched-resolution encoder x modality grid with tuned heads and repeated folds.

Fixes two protocol confounds of the released-preprocessing grid (grid.py): every package
sees identical square pixels at one resolution, and the linear head is standardised with
its regularisation chosen by inner cross-validation, repeated over three fold seeds.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import packages, pools
from .metrics import qwk

FOLD_SEEDS = (42, 43, 44)
C_GRID = (0.001, 0.01, 0.1, 1.0)


def _folds(y: np.ndarray, seed: int, k: int = 5):
    rng = np.random.RandomState(seed)
    by_class = [rng.permutation(np.where(y == c)[0]) for c in np.unique(y)]
    for fold in range(k):
        test = np.concatenate([m[fold::k] for m in by_class])
        yield np.setdiff1d(np.arange(len(y)), test), test


def _score(metric, y, prob):
    return roc_auc_score(y, prob[:, 1]) if metric == "auroc" else qwk(y, prob.argmax(1))


def _head(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3_000))


def _fit_fold(x, y, train, test, metric, seed):
    """Choose C by inner 3-fold CV on the training part only, then predict the test part."""
    inner = list(_folds(y[train], seed + 1000, k=3))
    best_c, best = C_GRID[0], -np.inf
    for C in C_GRID:
        scores = []
        for itr, ite in inner:
            head = _head(C).fit(x[train][itr], y[train][itr])
            scores.append(_score(metric, y[train][ite], head.predict_proba(x[train][ite])))
        if np.mean(scores) > best:
            best_c, best = C, float(np.mean(scores))
    return test, _head(best_c).fit(x[train], y[train]).predict_proba(x[test]), best_c


def cross_validate(x: np.ndarray, y: np.ndarray, metric: str) -> dict:
    """Seed-averaged out-of-fold class probabilities plus per-seed scores."""
    n_classes = len(np.unique(y))
    jobs = [(seed, tr, te) for seed in FOLD_SEEDS for tr, te in _folds(y, seed)]
    results = Parallel(n_jobs=4)(delayed(_fit_fold)(x, y, tr, te, metric, seed)
                                 for seed, tr, te in jobs)
    per_seed = {s: np.zeros((len(y), n_classes)) for s in FOLD_SEEDS}
    chosen = []
    for (seed, _, _), (test, prob, c) in zip(jobs, results):
        per_seed[seed][test] = prob
        chosen.append(c)
    mean_prob = np.mean([per_seed[s] for s in FOLD_SEEDS], axis=0)
    return {"prob": mean_prob,
            "seed_scores": [round(float(_score(metric, y, per_seed[s])), 5) for s in FOLD_SEEDS],
            "chosen_C": chosen}


def features(root: str, sizes=pools.SIZES, names=None) -> Path:
    """Extract and cache frozen features for every (pool, package, size)."""
    out = pools.cache_dir() / "features"
    out.mkdir(exist_ok=True)
    grid = pools.grid_pools(root)
    for package in names or packages.PACKAGES:
        for size in sizes:
            for domain, pool in grid.items():
                target = out / f"{domain}_{package}_{size}.npz"
                if target.exists():
                    continue
                start = time.time()
                feats, params = packages.embed(package, pool["cached"][size],
                                               batch=8 if size > 224 else 16)
                np.savez(target, x=feats, y=pool["y"], params=params,
                         seconds=time.time() - start)
                print(f"  {domain}/{package}/{size}: {feats.shape} "
                      f"{time.time() - start:.0f}s", flush=True)
    return out

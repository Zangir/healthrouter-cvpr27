"""Task metrics and the paired, stratified, seed-averaged bootstrap."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .data import METRIC

EPS = 1e-4


def auroc(y, s):
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def qwk(y, pred, k):
    O = np.zeros((k, k))
    np.add.at(O, (y, pred), 1)
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > 0 else 0.0


def primary(task: str, y: np.ndarray, P: np.ndarray) -> float:
    m = METRIC[task]
    if m == "auroc":
        return auroc(y, P[:, 1])
    if m == "qwk":
        return qwk(y, P.argmax(1), P.shape[1])
    return float((P.argmax(1) == y).mean())


def nll(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))))


def accuracy(y, P):
    return float((P.argmax(1) == y).mean())


def strat_boot_indices(y: np.ndarray, n_boot: int, seed: int, groups=None) -> np.ndarray:
    """(n_boot, n) resampled indices; class-stratified, or cluster bootstrap over groups."""
    rng = np.random.RandomState(seed)
    if groups is not None:
        uniq = np.unique(groups)
        members = [np.where(groups == g)[0] for g in uniq]
        out = []
        for _ in range(n_boot):
            pick = rng.randint(len(uniq), size=len(uniq))
            out.append(np.concatenate([members[i] for i in pick])[:len(y)])
        lens = min(len(o) for o in out)
        return np.stack([o[:lens] for o in out])
    idx_by_class = [np.where(y == c)[0] for c in np.unique(y)]
    out = np.empty((n_boot, len(y)), dtype=int)
    for b in range(n_boot):
        out[b] = np.concatenate([rng.choice(m, len(m), replace=True) for m in idx_by_class])
    return out


def seed_mean_metric(task, y, preds_by_seed, idx=None):
    """Metric averaged over seeds; each seed's predictions are full-length (n, K)."""
    if idx is None:
        return float(np.mean([primary(task, y, P) for P in preds_by_seed]))
    return float(np.mean([primary(task, y[idx], P[idx]) for P in preds_by_seed]))


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

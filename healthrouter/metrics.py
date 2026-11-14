"""Metrics and deterministic item-bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


def _stratified_indices(y: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    chunks = []
    for value in np.unique(y):
        members = np.where(y == value)[0]
        chunks.append(rng.choice(members, len(members), replace=True))
    return np.concatenate(chunks)


def bootstrap(y, pred, metric: Callable, n_boot: int = 10_000, seed: int = 2026):
    y = np.asarray(y)
    pred = np.asarray(pred)
    rng = np.random.RandomState(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _stratified_indices(y, rng)
        values[i] = metric(y[idx], pred[idx])
    point = float(metric(y, pred))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_auc(y, score, n_boot: int = 10_000):
    return bootstrap(y, score, roc_auc_score, n_boot)


def bootstrap_accuracy(y, pred, n_boot: int = 10_000):
    return bootstrap(y, pred, accuracy_score, n_boot)


def bootstrap_qwk(y, pred, n_boot: int = 10_000):
    fn = lambda a, b: cohen_kappa_score(a, b, weights="quadratic")
    return bootstrap(y, pred, fn, n_boot)


def paired_auc_difference(y, first, second, n_boot: int = 10_000, seed: int = 2026):
    y = np.asarray(y)
    first = np.asarray(first)
    second = np.asarray(second)
    rng = np.random.RandomState(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _stratified_indices(y, rng)
        values[i] = roc_auc_score(y[idx], first[idx]) - roc_auc_score(y[idx], second[idx])
    point = roc_auc_score(y, first) - roc_auc_score(y, second)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mean_interval(values, n_boot: int = 10_000, seed: int = 2026):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, len(values), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def paired_metric_difference(y, first, second, metric: Callable,
                             n_boot: int = 10_000, seed: int = 2026):
    """Paired item bootstrap for metric(first) - metric(second) on shared items."""
    y, first, second = np.asarray(y), np.asarray(first), np.asarray(second)
    rng = np.random.RandomState(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _stratified_indices(y, rng)
        values[i] = metric(y[idx], first[idx]) - metric(y[idx], second[idx])
    point = float(metric(y, first) - metric(y, second))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, float(lo), float(hi)


def qwk(y, pred):
    return cohen_kappa_score(y, pred, weights="quadratic")


def rounded_ci(values, digits: int = 4):
    return {"value": round(values[0], digits), "ci_low": round(values[1], digits),
            "ci_high": round(values[2], digits)}

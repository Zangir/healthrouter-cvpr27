"""Candidate models: one standardised logistic head per (encoder, resolution) and task.

For one outer fold the module returns out-of-fold candidate probabilities on the inner-train
part (the routers' meta-training data) and refit-on-inner-train probabilities on the
validation and test parts. Results are cached per (task, seed, fold, pool).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import data

C_GRID = (0.001, 0.01, 0.1, 1.0)
EPS = 1e-4


def _fit(x, y, C, n_classes):
    scaler = StandardScaler().fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        head = LogisticRegression(C=C, max_iter=3000).fit(scaler.transform(x), y)
    return scaler, head


def _predict(model, x, n_classes):
    scaler, head = model
    p = np.full((len(x), n_classes), 0.0)
    p[:, head.classes_] = head.predict_proba(scaler.transform(x))
    return p


def _nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def fit_candidate(x, y, train, others, n_classes, seed):
    """OOF probs on `train` (C chosen by OOF log-loss) + refit probs on each array in `others`."""
    folds = data.stratified_folds(y[train], 5, 3000 + seed)
    best = None
    for C in C_GRID:
        oof = np.zeros((len(train), n_classes))
        for f in folds:
            fit_idx = np.setdiff1d(np.arange(len(train)), f)
            model = _fit(x[train][fit_idx], y[train][fit_idx], C, n_classes)
            oof[f] = _predict(model, x[train][f], n_classes)
        loss = _nll(oof, y[train])
        if best is None or loss < best[0]:
            best = (loss, C, oof)
    model = _fit(x[train], y[train], best[1], n_classes)
    return best[2], [_predict(model, xo, n_classes) for xo in others], best[1]


def outer_fold(task: str, seed: int, fold: int, sizes=(224,)) -> dict:
    tag = "x".join(str(s) for s in sizes)
    path = data.cache_dir() / "router" / f"cand_{task}_s{seed}_f{fold}_{tag}.npz"
    if path.exists():
        return dict(np.load(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    cands = data.candidates(sizes)
    _, y = data.load(task, cands[0][0], cands[0][1])
    n_classes = int(y.max()) + 1
    tr, va, te = data.outer_split(y, seed, fold)
    P = {k: np.zeros((len(i), len(cands), n_classes), np.float32)
         for k, i in (("tr", tr), ("va", va), ("te", te))}
    chosen = []
    for c, (enc, size) in enumerate(cands):
        x, y2 = data.load(task, enc, size)
        assert np.array_equal(y, y2), (task, enc, size)
        oof, (pva, pte), C = fit_candidate(x, y, tr, [x[va], x[te]], n_classes, seed * 10 + fold)
        P["tr"][:, c], P["va"][:, c], P["te"][:, c] = oof, pva, pte
        chosen.append(C)
    zx, _ = data.load(task, data.ROUTER_ENCODER, 224)
    out = {"P_tr": P["tr"], "P_va": P["va"], "P_te": P["te"], "idx_tr": tr, "idx_va": va,
           "idx_te": te, "y_tr": y[tr], "y_va": y[va], "y_te": y[te], "Z_tr": zx[tr],
           "Z_va": zx[va], "Z_te": zx[te], "C": np.asarray(chosen)}
    np.savez_compressed(path, **out)
    return out


def external(task: str, conditions, sizes=(224,), seed: int = 0) -> dict:
    """Train on the in-source pool (80% inner-train / 20% val), predict external cohorts."""
    path = data.cache_dir() / "router" / f"ext_{task}_s{seed}.npz"
    if path.exists():
        return dict(np.load(path))
    cands = data.candidates(sizes)
    _, y = data.load(task, cands[0][0], cands[0][1])
    n_classes = int(y.max()) + 1
    tr, va = data.stratified_holdout(y, 0.2, 4000 + seed)
    out = {"idx_tr": tr, "idx_va": va, "y_tr": y[tr], "y_va": y[va]}
    _, y_ext = data.load(task, cands[0][0], 224, conditions[0])
    out["y_ext"] = y_ext
    P_tr = np.zeros((len(tr), len(cands), n_classes), np.float32)
    P_va = np.zeros((len(va), len(cands), n_classes), np.float32)
    P_ext = {cond: np.zeros((len(y_ext), len(cands), n_classes), np.float32) for cond in conditions}
    for c, (enc, size) in enumerate(cands):
        x, _ = data.load(task, enc, size)
        xe = [data.load(task, enc, 224, cond)[0] for cond in conditions]
        oof, preds, _ = fit_candidate(x, y, tr, [x[va]] + xe, n_classes, 777 + seed)
        P_tr[:, c], P_va[:, c] = oof, preds[0]
        for cond, p in zip(conditions, preds[1:]):
            P_ext[cond][:, c] = p
    zx, _ = data.load(task, data.ROUTER_ENCODER, 224)
    out.update({"P_tr": P_tr, "P_va": P_va, "Z_tr": zx[tr], "Z_va": zx[va]})
    for cond in conditions:
        out[f"P_ext_{cond}"] = P_ext[cond]
        out[f"Z_ext_{cond}"] = data.load(task, data.ROUTER_ENCODER, 224, cond)[0]
    np.savez_compressed(path, **out)
    return out

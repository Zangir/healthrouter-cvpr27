#!/usr/bin/env python3
"""Post-plan analysis R3b, step 2 (plan amendment 9, external review request): grouped re-run.

Mammography (CBIS-DDSM) is grouped by patient and pathology (PatchCamelyon) by whole-slide image
(groups from scripts/router_review_groups.py). For those two tasks the outer 5-fold split, the
inner validation hold-out and the candidates' inner out-of-fold folds are all grouped
(StratifiedGroupKFold); chest and fundus keep their frozen folds and cached candidates.
Candidates are refit on the frozen features with the plan's protocol (standardised logistic
head, C by out-of-fold log-loss); the methods select their configurations on validation with
the common rule and are scored once on the grouped test folds.

One resumable run per split seed: cache/router/review/grouped_s<seed>.pkl + .json.
`--aggregate` writes results/router_review_grouped.json from the finished seeds.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_aggregate import r, summarize  # noqa: E402
from router_experiment import choose, families, load_run, text_embeddings  # noqa: E402

from healthrouter.collection import candidates, data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUPED = ("mammography", "pathology")
METHODS = ["healthrouter", "healthrouter_harness", "best_single_per_task", "uniform_ensemble", "soft_moe",
           "sparse_topk_moe", "knora_u", "knora_e", "meta_des", "knn_router"]


def review_dir() -> Path:
    d = data.cache_dir() / "router" / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sgkf(y, g, seed, k=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [te for _, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g)]


def grouped_split(y, g, seed, fold):
    test = np.sort(sgkf(y, g, 1000 + seed)[fold])
    rest = np.setdiff1d(np.arange(len(y)), test)
    va_rel = np.sort(sgkf(y[rest], g[rest], 2000 + 10 * seed + fold)[0])
    va = rest[va_rel]
    tr = np.setdiff1d(rest, va)
    return tr, va, test


def fit_candidate_grouped(x, y, g, train, others, n_classes, seed):
    folds = sgkf(y[train], g[train], 3000 + seed)
    best = None
    for C in candidates.C_GRID:
        oof = np.zeros((len(train), n_classes))
        for f in folds:
            fit_idx = np.setdiff1d(np.arange(len(train)), f)
            model = candidates._fit(x[train][fit_idx], y[train][fit_idx], C, n_classes)
            oof[f] = candidates._predict(model, x[train][f], n_classes)
        loss = candidates._nll(oof, y[train])
        if best is None or loss < best[0]:
            best = (loss, C, oof)
    model = candidates._fit(x[train], y[train], best[1], n_classes)
    return best[2], [candidates._predict(model, xo, n_classes) for xo in others], best[1]


def grouped_outer_fold(task, groups, seed, fold):
    path = review_dir() / f"gcand_{task}_s{seed}_f{fold}.npz"
    if path.exists():
        return dict(np.load(path))
    cands = data.candidates((224,))
    _, y = data.load(task, cands[0][0], 224)
    n_classes = int(y.max()) + 1
    tr, va, te = grouped_split(y, groups, seed, fold)
    P = {k: np.zeros((len(i), len(cands), n_classes), np.float32) for k, i in (("tr", tr), ("va", va), ("te", te))}
    chosen = []
    for c, (enc, size) in enumerate(cands):
        x, y2 = data.load(task, enc, size)
        assert np.array_equal(y, y2)
        oof, (pva, pte), C = fit_candidate_grouped(x, y, groups, tr, [x[va], x[te]], n_classes, seed * 10 + fold)
        P["tr"][:, c], P["va"][:, c], P["te"][:, c] = oof, pva, pte
        chosen.append(C)
    zx, _ = data.load(task, data.ROUTER_ENCODER, 224)
    out = {"P_tr": P["tr"], "P_va": P["va"], "P_te": P["te"], "idx_tr": tr, "idx_va": va, "idx_te": te,
           "y_tr": y[tr], "y_va": y[va], "y_te": y[te], "Z_tr": zx[tr], "Z_va": zx[va], "Z_te": zx[te],
           "C": np.asarray(chosen)}
    np.savez_compressed(path, **out)
    return out


def build(seed, fold, groups):
    run = load_run(seed, fold, data.MED_TASKS)          # frozen folds; grouped tasks replaced below
    for t in GROUPED:
        d = grouped_outer_fold(t, groups[t], seed, fold)
        B, _ = data.load(t, "biomedclip", 224)
        A, _ = data.load(t, "dinov2_large", 224)
        run[t] = dict(d)
        for sp in ("tr", "va", "te"):
            run[t][f"B_{sp}"] = B[d[f"idx_{sp}"]]
            run[t][f"A_{sp}"] = A[d[f"idx_{sp}"]]
    for i, t in enumerate(run["tasks"]):
        for sp in ("tr", "va", "te"):
            run[t][f"T_{sp}"] = np.repeat(np.eye(len(run["tasks"]), dtype=np.float32)[[i]], len(run[t][f"y_{sp}"]), 0)
    return run


def run_seed(seed):
    base = review_dir() / f"grouped_s{seed}"
    if base.with_suffix(".json").exists():
        print("exists, skipped: seed", seed)
        return
    g = np.load(review_dir() / "groups.npz")
    groups = {t: g[t] for t in GROUPED}
    assert all(len(groups[t]) for t in GROUPED), "group mapping incomplete; R3b not run"
    text_emb = text_embeddings(data.MED_TASKS)
    cost = data.costs_ms((224,))
    store, leak = {}, {}
    for fold in range(data.N_OUTER):
        t0 = time.time()
        run = build(seed, fold, groups)
        for t in GROUPED:  # no group may straddle training and test
            tr = set(groups[t][np.concatenate([run[t]["idx_tr"], run[t]["idx_va"]])].tolist())
            leak[(t, fold)] = len(tr & set(groups[t][run[t]["idx_te"]].tolist()))
        fams = families(run, ["va", "te"], text_emb, cost)
        for name in METHODS:
            _, cfg, out, sc, per, inv = choose(run, fams[name]())
            store[(name, seed, fold)] = {"cfg": cfg, "val_macro": sc,
                                         "test": {t: (run[t]["idx_te"], out["te"][t][0].astype(np.float32), out["te"][t][1])
                                                  for t in data.MED_TASKS}}
        print(f"seed {seed} fold {fold} done in {time.time() - t0:.0f}s; shared groups {[leak[(t, fold)] for t in GROUPED]}",
              flush=True)
    base.with_suffix(".pkl").write_bytes(pickle.dumps(store))
    base.with_suffix(".json").write_text(json.dumps({"seed": seed, "shared_groups_train_test": sum(leak.values()),
                                                     "methods": METHODS}, indent=1) + "\n")
    print("wrote", base.with_suffix(".json"), flush=True)


def aggregate():
    seeds = [s for s in data.SPLIT_SEEDS if (review_dir() / f"grouped_s{s}.pkl").exists()]
    if not seeds:
        print("no finished seed")
        return
    store = {}
    for s in seeds:
        store.update(pickle.loads((review_dir() / f"grouped_s{s}.pkl").read_bytes()))
    res = summarize(store, data.MED_TASKS, (224,))
    meta = json.loads((ROOT / "results" / "router_review_groups.json").read_text())
    out = {"amendment": "router_plan.json amendment 9, R3b (post-plan, external review request)",
           "provenance": "scripts/router_review_grouped.py", "seeds": seeds,
           "status": "complete" if len(seeds) == len(data.SPLIT_SEEDS) else "partial",
           "groups": {"mammography": meta["mammography_cbis_ddsm"]["patients"], "pathology": meta["pathology_pcam"]["slides"]},
           "shared_groups_train_test": sum(json.loads((review_dir() / f"grouped_s{s}.json").read_text())["shared_groups_train_test"]
                                           for s in seeds),
           "methods": {n: {"per_task": {t: v["per_task"][t]["value"] for t in data.MED_TASKS}, "macro": v["macro"],
                           "invoked_mean": v["invoked_mean"], "latency_ms_mean": v["latency_ms_mean"]}
                       for n, v in res["methods"].items()},
           "differences_vs_healthrouter": res["differences_vs_healthrouter"]}
    (ROOT / "results" / "router_review_grouped.json").write_text(json.dumps(out, indent=1) + "\n")
    for n, d in out["differences_vs_healthrouter"].items():
        print(f"HR - {n:24s}", {k: (v["value"], v["ci"]) for k, v in d.items() if k in ("mammography", "pathology", "macro")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    if a.seed is not None:
        run_seed(a.seed)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()

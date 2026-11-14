#!/usr/bin/env python3
"""Validation-only design exploration (seed 0, five outer folds). Never touches test."""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from router_experiment import GRID, load_run, text_embeddings  # noqa: E402

from healthrouter.collection import data, methods, metrics, router  # noqa: E402


def val_macro(run, W_va):
    Wt = methods.split_by_task(run, "va", W_va)
    s = [metrics.primary(t, run[t]["y_va"], router.fuse(run[t]["P_va"], Wt[t])) for t in run["tasks"]]
    return float(np.mean(s)), s


def coll_variants(run, max_size, epochs=300, wd=1e-4, hidden=256):
    Ztr = methods.pooled(run, "tr")
    std = router.Standardizer().fit(Ztr)
    U = np.concatenate([methods.true_prob(run[t]["P_tr"], run[t]["y_tr"]) for t in run["tasks"]])
    lib = router.collection_library(run["n_cand"], max_size)
    out = {}
    for tau in (0.25, 1.0):
        for beta in (0.0, 1.0):
            g = router.train_collection_gate(std(Ztr), U, lib, tau=tau, beta=beta, epochs=epochs,
                                             wd=wd, hidden=hidden, seed=run["seed"])
            S = router.scores(g, std(methods.pooled(run, "va")))
            for T in (0.5, 1.0, 2.0):
                for m in (1, 3, 5, 10):
                    out[(max_size, tau, beta, T, m, epochs, wd)] = router.select_collection(S, lib, T, m)
    return out


def main():
    import json
    folds = [int(x) for x in sys.argv[1].split(",")]
    tasks = sys.argv[2:] or data.MED_TASKS
    store = data.cache_dir() / "router" / "dev_explore.json"
    saved = json.loads(store.read_text()) if store.exists() else {}
    res = {}
    text_emb = text_embeddings(tasks)
    for fold in folds:
        t0 = time.time()
        run = load_run(0, fold, tasks)
        # reference: per-task top-k (oracle-free domain lane), modality router
        for name, fam in (("modality_sup", methods.modality_router_supervised(run, ["va"])),
                          ("modality_zs", methods.modality_router_zeroshot(run, ["va"], text_emb)),
                          ("knn", methods.knn_router(run, ["va"])),
                          ("lane_oof", methods.modality_router_supervised(run, ["va"], source="tr")),
                          ("single_oof", methods.best_single_per_task(run, ["va"], source="tr")),
                          ("single_va", methods.best_single_per_task(run, ["va"])),
                          ("hr", methods.healthrouter(run, ["va"], GRID)),
                          ("hr_uniform", methods.healthrouter(run, ["va"], GRID, uniform_weights=True))):
            for cfg, out in fam:
                sc = np.mean([metrics.primary(t, run[t]["y_va"], out["va"][t][0]) for t in tasks])
                res.setdefault((name, str(cfg)), []).append(sc)
        for ms in (2, 3):
            for key, W in coll_variants(run, ms).items():
                res.setdefault(("coll", str(key)), []).append(val_macro(run, W)[0])
        print(f"fold {fold} {time.time() - t0:.0f}s", flush=True)
        for (fam, cfg), v in res.items():
            saved.setdefault(f"{fam}|{cfg}", {})[str(fold)] = v[-1]
        store.write_text(json.dumps(saved))
    by_family = {}
    for k, v in saved.items():
        fam, cfg = k.split("|", 1)
        if len(v) < 5:
            continue
        by_family.setdefault(fam, []).append((np.mean(list(v.values())), cfg))
    for fam, lst in by_family.items():
        lst.sort(reverse=True)
        print(fam, "best fixed config (mean val over folds):")
        for sc, cfg in lst[:6]:
            print(f"   {sc:.4f} {cfg}")


if __name__ == "__main__":
    main()

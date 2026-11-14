"""Validation-only: neighbourhood collection-utility variants (seed 0)."""
import numpy as np
from router_experiment import load_run, GRID
from healthrouter.collection import data, methods, metrics, router
res = {}
for fold in range(5):
    run = load_run(0, fold, data.MED_TASKS)
    Ztr = methods.pooled(run, "tr"); std = router.Standardizer().fit(Ztr)
    A = std(Ztr); A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = std(methods.pooled(run, "va")); B /= np.linalg.norm(B, axis=1, keepdims=True)
    U = np.concatenate([methods.true_prob(run[t]["P_tr"], run[t]["y_tr"]) for t in run["tasks"]])
    sim = B @ A.T
    order = np.argsort(-sim, 1)[:, :200]
    for ms in (1, 2, 3, 5):
        lib = router.collection_library(run["n_cand"], ms)
        CU = router.collection_utility(U, lib)          # (ntr, nS)
        for nn in (10, 25, 50, 100, 200):
            est = CU[order[:, :nn]].mean(1)            # (nva, nS)
            for m in (1, 3):
                W = router.select_collection(est, lib, 1e-3 if m == 1 else 0.05, m)
                Wt = methods.split_by_task(run, "va", W)
                s = [metrics.primary(t, run[t]["y_va"], router.fuse(run[t]["P_va"], Wt[t])) for t in run["tasks"]]
                res.setdefault(("knn_coll", ms, nn, m), []).append(s)
        # mean-of-probabilities utility instead of mean of log
        for nn in (25, 50, 100):
            avgp = (U @ lib.T) / lib.sum(1)
            est = np.log(np.clip(avgp[order[:, :nn]].mean(1), 1e-4, 1))
            W = router.select_collection(est, lib, 1e-3, 1)
            Wt = methods.split_by_task(run, "va", W)
            s = [metrics.primary(t, run[t]["y_va"], router.fuse(run[t]["P_va"], Wt[t])) for t in run["tasks"]]
            res.setdefault(("knn_coll_meanp", ms, nn), []).append(s)
for k, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1]))[:25]:
    print(f"{np.mean(v):.4f} {np.round(np.mean(v,0),3)} {k}")

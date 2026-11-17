#!/usr/bin/env python3
"""Post-plan analysis R3a (plan amendment 9, external review request): near-duplicate diagnostic.

For every in-source task, split seed and outer fold: cosine similarity of each test image to its
nearest image in the fold's training part (inner-train + validation), on the standardised
ConvNeXt-T router features (standardisation fitted on the pooled meta-training features, as in
HealthRouter). Reports the share of test evaluations above 0.95 / 0.98 / 0.99, and HealthRouter
minus soft MoE, sparse MoE, KNORA-U, KNORA-E, META-DES and the neighbourhood router on all test
images and with above-threshold images excluded (excluded if above threshold in any seed), with
the paired stratified bootstrap of router_aggregate.py (2,000 replicates, seeds 17 + task index)
on the retained images. Writes results/router_review_neardup.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_aggregate import assemble, r  # noqa: E402

from healthrouter.collection import candidates, data, metrics, router  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.95, 0.98, 0.99)
BASELINES = ["soft_moe", "sparse_topk_moe", "knora_u", "knora_e", "meta_des", "knn_router"]


def max_similarity(tasks):
    """{task: (n_seeds, n) max cosine similarity of each pool image, as a test image, to training}."""
    out = {t: np.full((len(data.SPLIT_SEEDS), len(data.load(t, data.ROUTER_ENCODER, 224)[1])), np.nan) for t in tasks}
    for si, s in enumerate(data.SPLIT_SEEDS):
        for f in range(data.N_OUTER):
            d = {t: candidates.outer_fold(t, s, f) for t in tasks}
            std = router.Standardizer().fit(np.concatenate([d[t]["Z_tr"] for t in tasks]))
            norm = lambda z: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(std(z))  # noqa: E731
            for t in tasks:
                A = norm(np.concatenate([d[t]["Z_tr"], d[t]["Z_va"]]))
                B = norm(d[t]["Z_te"])
                out[t][si, d[t]["idx_te"]] = (B @ A.T).max(1)
    for t in tasks:
        assert not np.isnan(out[t]).any(), t
    return out


def main():
    tasks = data.MED_TASKS
    sim = max_similarity(tasks)
    store = pickle.loads((data.cache_dir() / "router" / "final_main.pkl").read_bytes())
    store = {k: v for k, v in store.items() if k[0] in ["healthrouter"] + BASELINES}
    names, seeds, ys, preds, _ = assemble(store, tasks, (224,))
    res = {"amendment": "router_plan.json amendment 9, R3a (post-plan, external review request)",
           "provenance": "scripts/router_review_neardup.py", "thresholds": list(THRESHOLDS),
           "share_above": {}, "excluded_images": {}, "similarity_quantiles": {}, "differences": {}}
    for t in tasks:
        res["share_above"][t] = {str(th): r((sim[t] > th).mean()) for th in THRESHOLDS}
        res["similarity_quantiles"][t] = {q: r(np.percentile(sim[t], q)) for q in (50, 90, 99)}
        res["similarity_quantiles"][t]["max"] = r(sim[t].max())
    for th in (None,) + THRESHOLDS:
        key = "all" if th is None else str(th)
        keep = {t: np.ones(len(ys[t]), bool) if th is None else ~(sim[t] > th).any(0) for t in tasks}
        res["excluded_images"][key] = {t: int((~keep[t]).sum()) for t in tasks}
        boots = {t: metrics.strat_boot_indices(ys[t][keep[t]], 2000, 17 + i) for i, t in enumerate(tasks)}
        sub = lambda n, t: [P[keep[t]] for P in preds[n][t]]  # noqa: E731
        rep = {n: {t: np.array([metrics.seed_mean_metric(t, ys[t][keep[t]], sub(n, t), b) for b in boots[t]])
                   for t in tasks} for n in names}
        val = {n: {t: metrics.seed_mean_metric(t, ys[t][keep[t]], sub(n, t)) for t in tasks} for n in names}
        res["differences"][key] = {}
        for b in BASELINES:
            d = {t: {"value": r(val["healthrouter"][t] - val[b][t]),
                     "ci": [r(x) for x in metrics.ci(rep["healthrouter"][t] - rep[b][t])]} for t in tasks}
            md = np.mean([rep["healthrouter"][t] for t in tasks], 0) - np.mean([rep[b][t] for t in tasks], 0)
            d["macro"] = {"value": r(np.mean([val["healthrouter"][t] - val[b][t] for t in tasks])),
                          "ci": [r(x) for x in metrics.ci(md)]}
            res["differences"][key][b] = d
        print(key, res["excluded_images"][key], {b: res["differences"][key][b]["macro"] for b in BASELINES}, flush=True)
    (ROOT / "results" / "router_review_neardup.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["share_above"]), json.dumps(res["similarity_quantiles"]))


if __name__ == "__main__":
    main()

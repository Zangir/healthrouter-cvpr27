#!/usr/bin/env python3
"""Post-plan analysis R1 (plan amendment 7, external review request).

Paired HealthRouter minus neighbourhood-router differences on the four external cohorts under
the five acquisition conditions, from the stored validation-chosen predictions, with the same
stratified paired bootstrap as router_external.json (2,000 replicates, seeds 31 + task index);
and, in distribution, harness minus neighbourhood router with the bootstrap of
router_aggregate.py (2,000 replicates, seeds 17 + task index). No model is refit.
Writes results/router_review_ext_nr.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_aggregate import r, summarize  # noqa: E402
from router_external import CONDITIONS  # noqa: E402

from healthrouter.collection import candidates, data, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A, B = "healthrouter", "knn_router"


def external():
    tasks = data.MED_TASKS
    store = pickle.loads((data.cache_dir() / "router" / "final_external.pkl").read_bytes())
    ys = {t: candidates.external(t, CONDITIONS)["y_ext"] for t in tasks}
    boots = {t: metrics.strat_boot_indices(ys[t], 2000, 31 + i) for i, t in enumerate(tasks)}
    out = {}
    for cond in CONDITIONS:
        res, reps = {}, {}
        for t in tasks:
            Pa, Pb = store[(A, cond)]["test"][t][0], store[(B, cond)]["test"][t][0]
            ra = np.array([metrics.primary(t, ys[t][b], Pa[b]) for b in boots[t]])
            rb = np.array([metrics.primary(t, ys[t][b], Pb[b]) for b in boots[t]])
            reps[t] = (ra, rb)
            va, vb = metrics.primary(t, ys[t], Pa), metrics.primary(t, ys[t], Pb)
            res[t] = {"healthrouter": r(va), "neighbourhood_router": r(vb), "diff": r(va - vb),
                      "ci": [r(x) for x in metrics.ci(ra - rb)], "n": int(len(ys[t]))}
        d = np.mean([reps[t][0] for t in tasks], 0) - np.mean([reps[t][1] for t in tasks], 0)
        res["macro"] = {"healthrouter": r(np.mean([res[t]["healthrouter"] for t in tasks])),
                        "neighbourhood_router": r(np.mean([res[t]["neighbourhood_router"] for t in tasks])),
                        "ci": [r(x) for x in metrics.ci(d)]}
        res["macro"]["diff"] = r(np.mean([metrics.primary(t, ys[t], store[(A, cond)]["test"][t][0])
                                          - metrics.primary(t, ys[t], store[(B, cond)]["test"][t][0]) for t in tasks]))
        res["calls"] = {n: r(np.concatenate([store[(n, cond)]["test"][t][1] for t in tasks]).sum(1).mean(), 3)
                        for n in (A, B)}
        out[cond] = res
        print(cond, res["macro"], {t: (res[t]["diff"], res[t]["ci"]) for t in tasks}, flush=True)
    return out


def in_distribution():
    store = pickle.loads((data.cache_dir() / "router" / "final_main.pkl").read_bytes())
    sub = {k: v for k, v in store.items() if k[0] in ("healthrouter_harness", B)}
    res = summarize(sub, data.MED_TASKS, (224,), ours="healthrouter_harness")
    d = res["differences_vs_healthrouter_harness"][B]
    m = res["methods"]
    out = {"harness": {"macro": m["healthrouter_harness"]["macro"], "calls": m["healthrouter_harness"]["invoked_mean"],
                       "ms": m["healthrouter_harness"]["latency_ms_mean"]},
           "neighbourhood_router": {"macro": m[B]["macro"], "calls": m[B]["invoked_mean"], "ms": m[B]["latency_ms_mean"]},
           "diff_harness_minus_neighbourhood_router": d}
    print("in-distribution", json.dumps(out), flush=True)
    return out


def main():
    out = {"amendment": "router_plan.json amendment 7 (post-plan, external review request)",
           "provenance": "scripts/router_review_extnr.py on cache/router/final_external.pkl and final_main.pkl",
           "bootstrap": "stratified paired, 2,000 replicates; external seeds 31+task index (as router_external.json); "
                        "in distribution seeds 17+task index, seed-averaged (as router_aggregate.py)",
           "external_hr_minus_nr": external(), "in_distribution": in_distribution()}
    (ROOT / "results" / "router_review_ext_nr.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

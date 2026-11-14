#!/usr/bin/env python3
"""External transfer under acquisition shift.

Candidates and every router are trained on the in-source pools (80% inner-train with
out-of-fold meta-training predictions, 20% validation for configuration choice) and applied
unchanged to the four external classification cohorts under five acquisition conditions.
Writes results/router_external.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_experiment import choose, families, text_embeddings  # noqa: E402
from router_aggregate import r  # noqa: E402

from healthrouter.collection import candidates, data, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]
MAIN = ["healthrouter", "healthrouter_harness", "healthrouter_fallback", "harness_fallback", "best_single_per_task", "best_single_per_task_oof", "best_single_overall",
        "uniform_ensemble", "stacking", "greedy_ensemble_selection", "modality_router_supervised",
        "modality_router_supervised_oof", "modality_router_zeroshot_biomedclip", "soft_moe",
        "sparse_topk_moe", "knn_router", "knora_u", "knora_e", "meta_des", "routerbench_mlp",
        "routellm_mf", "logme_selection", "hr_candidate_mlp", "oracle_single"]


def build_run(tasks, cond, seed=0):
    run = {"tasks": tasks, "seed": 900 + seed}
    for t in tasks:
        d = candidates.external(t, CONDITIONS, seed=seed)
        B, _ = data.load(t, "biomedclip", 224)
        A, _ = data.load(t, "dinov2_large", 224)
        e = {"P_tr": d["P_tr"], "P_va": d["P_va"], "P_te": d[f"P_ext_{cond}"],
             "y_tr": d["y_tr"], "y_va": d["y_va"], "y_te": d["y_ext"],
             "Z_tr": d["Z_tr"], "Z_va": d["Z_va"], "Z_te": d[f"Z_ext_{cond}"],
             "B_tr": B[d["idx_tr"]], "B_va": B[d["idx_va"]],
             "B_te": data.load(t, "biomedclip", 224, cond)[0],
             "A_tr": A[d["idx_tr"]], "A_va": A[d["idx_va"]],
             "A_te": data.load(t, "dinov2_large", 224, cond)[0],
             "idx_te": np.arange(len(d["y_ext"]))}
        e["X_tr"] = [data.load(t, enc, s)[0][d["idx_tr"]] for enc, s in data.candidates()]
        run[t] = e
    for i, t in enumerate(tasks):
        for sp in ("tr", "va", "te"):
            run[t][f"T_{sp}"] = np.repeat(np.eye(len(tasks), dtype=np.float32)[[i]], len(run[t][f"y_{sp}"]), 0)
    run["n_cand"] = run[tasks[0]]["P_tr"].shape[1]
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    tasks = data.MED_TASKS
    path = data.cache_dir() / "router" / "final_external.pkl"
    store = pickle.loads(path.read_bytes()) if path.exists() else {}
    if not a.aggregate:
        text_emb = text_embeddings(tasks)
        cost = data.costs_ms()
        for cond in a.conditions:
            run = build_run(tasks, cond)
            fams = families(run, ["va", "te"], text_emb, cost)
            for name in MAIN:
                if (name, cond) in store:
                    continue
                _, cfg, out, sc, per, inv = choose(run, fams[name]())
                store[(name, cond)] = {"cfg": cfg, "val_macro": sc,
                                       "test": {t: (out["te"][t][0].astype(np.float32), out["te"][t][1]) for t in tasks}}
                print(cond, name, round(sc, 4), cfg, flush=True)
            path.write_bytes(pickle.dumps(store))
        return
    ys = {t: candidates.external(t, CONDITIONS)["y_ext"] for t in tasks}
    cost = data.costs_ms()
    boots = {t: metrics.strat_boot_indices(ys[t], 2000, 31 + i) for i, t in enumerate(tasks)}
    out = {"tasks": tasks, "conditions": CONDITIONS, "n": {t: int(len(ys[t])) for t in tasks},
           "provenance": "scripts/router_external.py", "results": {}}
    for cond in CONDITIONS:
        names = [n for n in MAIN if (n, cond) in store]
        rep = {n: {t: np.array([metrics.primary(t, ys[t][b], store[(n, cond)]["test"][t][0][b]) for b in boots[t]])
                   for t in tasks} for n in names}
        res = {}
        for n in names:
            m = {"per_task": {}, "cfg": store[(n, cond)]["cfg"]}
            for t in tasks:
                P, M = store[(n, cond)]["test"][t]
                m["per_task"][t] = {"value": r(metrics.primary(t, ys[t], P)), "ci": [r(x) for x in metrics.ci(rep[n][t])]}
            mac = np.mean([rep[n][t] for t in tasks], 0)
            m["macro"] = {"value": r(np.mean([m["per_task"][t]["value"] for t in tasks])), "ci": [r(x) for x in metrics.ci(mac)]}
            M_all = np.concatenate([store[(n, cond)]["test"][t][1] for t in tasks])
            m["invoked_mean"] = r(M_all.sum(1).mean(), 3)
            if "healthrouter" in names and n != "healthrouter":
                d = np.mean([rep["healthrouter"][t] for t in tasks], 0) - mac
                hr = np.mean([metrics.primary(t, ys[t], store[("healthrouter", cond)]["test"][t][0]) for t in tasks])
                m["diff_healthrouter_minus_this"] = {"value": r(hr - m["macro"]["value"]),
                                                     "ci": [r(x) for x in metrics.ci(d)]}
            res[n] = m
        out["results"][cond] = res
        print(cond)
        for n, m in sorted(res.items(), key=lambda kv: -kv[1]["macro"]["value"]):
            print(f"   {n:38s} {m['macro']['value']:.4f} {m['macro']['ci']} "
                  f"{[m['per_task'][t]['value'] for t in tasks]} inv={m['invoked_mean']} "
                  f"{m.get('diff_healthrouter_minus_this', '')}")
    (ROOT / "results" / "router_external.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

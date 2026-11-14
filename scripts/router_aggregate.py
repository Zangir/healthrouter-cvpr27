#!/usr/bin/env python3
"""Aggregate cross-fitted test predictions: seed-averaged metrics, paired stratified bootstrap
CIs, differences against HealthRouter, invoked models and measured latency per sample."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from healthrouter.collection import data, metrics

ROOT = Path(__file__).resolve().parents[1]
OURS = "healthrouter"
ROUTER_ENC = {"healthrouter_fallback": "imagenet_convnext", "harness_fallback": "imagenet_convnext",
              "healthrouter": "imagenet_convnext", "healthrouter_harness": "imagenet_convnext", "hr_candidate_mlp": "imagenet_convnext",
              "hr_collection_mlp": "imagenet_convnext", "modality_router_supervised_oof": "imagenet_convnext", "soft_moe": "imagenet_convnext",
              "sparse_topk_moe": "imagenet_convnext", "knn_router": "imagenet_convnext",
              "knora_u": "imagenet_convnext", "knora_e": "imagenet_convnext",
              "meta_des": "imagenet_convnext", "routerbench_mlp": "imagenet_convnext",
              "routellm_mf": "imagenet_convnext", "modality_router_supervised": "imagenet_convnext",
              "modality_router_zeroshot_biomedclip": "biomedclip"}


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def assemble(store, tasks, sizes):
    names = sorted({k[0] for k in store})
    seeds = sorted({k[1] for k in store})
    ys = {t: data.load(t, data.candidates(sizes)[0][0], data.candidates(sizes)[0][1])[1] for t in tasks}
    preds, masks = {}, {}
    for n in names:
        preds[n], masks[n] = {}, {}
        for s in seeds:
            for t in tasks:
                K = int(ys[t].max()) + 1
                P = np.full((len(ys[t]), K), np.nan, np.float32)
                M = None
                for f in range(data.N_OUTER):
                    idx, Pt, Mt = store[(n, s, f)]["test"][t]
                    P[idx] = Pt
                    if M is None:
                        M = np.zeros((len(ys[t]), Mt.shape[1]), bool)
                    M[idx] = Mt
                assert not np.isnan(P).any(), (n, s, t)
                preds[n].setdefault(t, []).append(P)
                masks[n].setdefault(t, []).append(M)
    return names, seeds, ys, preds, masks


def summarize(store, tasks, sizes, n_boot=2000, ours=OURS):
    names, seeds, ys, preds, masks = assemble(store, tasks, sizes)
    cands = data.candidates(sizes)
    cost = data.costs_ms(sizes)
    boots = {t: metrics.strat_boot_indices(ys[t], n_boot, 17 + i) for i, t in enumerate(tasks)}
    # per-replicate metric arrays
    rep = {n: {t: np.array([metrics.seed_mean_metric(t, ys[t], preds[n][t], b) for b in boots[t]])
               for t in tasks} for n in names}
    out = {"tasks": tasks, "seeds": seeds, "n_boot": n_boot, "candidates": [f"{e}@{s}" for e, s in cands],
           "candidate_cost_ms": [r(c, 2) for c in cost], "methods": {}}
    for n in names:
        m = {"per_task": {}, "chosen_configs": [store[(n, s, f)]["cfg"] for s in seeds
                                                for f in range(data.N_OUTER)]}
        for t in tasks:
            pt = metrics.seed_mean_metric(t, ys[t], preds[n][t])
            per_seed = [metrics.primary(t, ys[t], P) for P in preds[n][t]]
            m["per_task"][t] = {"metric": data.METRIC[t], "value": r(pt), "ci": [r(x) for x in metrics.ci(rep[n][t])],
                                "seed_values": [r(x) for x in per_seed],
                                "nll": r(np.mean([metrics.nll(ys[t], P) for P in preds[n][t]])),
                                "accuracy": r(np.mean([metrics.accuracy(ys[t], P) for P in preds[n][t]])),
                                "n": int(len(ys[t]))}
        macro_rep = np.mean([rep[n][t] for t in tasks], 0)
        macro_seed = np.mean([[metrics.primary(t, ys[t], preds[n][t][i]) for t in tasks]
                              for i in range(len(seeds))], 1)
        m["macro"] = {"value": r(np.mean([m["per_task"][t]["value"] for t in tasks])),
                      "ci": [r(x) for x in metrics.ci(macro_rep)],
                      "seed_values": [r(x) for x in macro_seed], "seed_std": r(np.std(macro_seed))}
        # compute: invoked candidates and latency per sample (ms), averaged over samples
        enc = ROUTER_ENC.get(n) or ("imagenet_convnext" if n.startswith(("harness_", "cost_", "abl_")) else None)
        inv, lat = [], []
        for t in tasks:
            for M in masks[n][t]:
                inv.append(M.sum(1))
                c = (M * cost[None, :]).sum(1)
                if enc is not None:
                    ridx = [i for i, (e, s) in enumerate(cands) if e == enc and s == 224]
                    c = c - (M[:, ridx[0]] * cost[ridx[0]] if ridx else 0) + data.costs_ms((224,))[data.ENCODERS.index(enc)]
                lat.append(c)
        m["invoked_mean"] = r(np.mean(np.concatenate(inv)), 3)
        m["latency_ms_mean"] = r(np.mean(np.concatenate(lat)), 2)
        m["invoked_hist"] = {t: [r(x, 4) for x in np.bincount(np.concatenate([M.sum(1) for M in masks[n][t]]),
                                                              minlength=len(cands) + 1)[1:] / (len(seeds) * len(ys[t]))]
                             for t in tasks}
        m["invoked_per_task"] = {t: r(np.mean([M.sum(1).mean() for M in masks[n][t]]), 3) for t in tasks}
        m["selection_frequency"] = {t: [r(x, 4) for x in np.mean([M.mean(0) for M in masks[n][t]], 0)]
                                    for t in tasks}
        out["methods"][n] = m
    if ours in names:
        out["differences_vs_" + ours] = {}
        for n in names:
            if n == ours:
                continue
            d = {}
            for t in tasks:
                diff = rep[ours][t] - rep[n][t]
                d[t] = {"value": r(out["methods"][ours]["per_task"][t]["value"] - out["methods"][n]["per_task"][t]["value"]),
                        "ci": [r(x) for x in metrics.ci(diff)], "p_le_0": r(np.mean(diff <= 0), 4)}
            md = np.mean([rep[ours][t] for t in tasks], 0) - np.mean([rep[n][t] for t in tasks], 0)
            d["macro"] = {"value": r(out["methods"][ours]["macro"]["value"] - out["methods"][n]["macro"]["value"]),
                          "ci": [r(x) for x in metrics.ci(md)], "p_le_0": r(np.mean(md <= 0), 4)}
            out["differences_vs_" + ours][n] = d
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main")
    p.add_argument("--tasks", nargs="+", default=data.MED_TASKS)
    p.add_argument("--sizes", nargs="+", type=int, default=[224])
    p.add_argument("--ours", default=OURS)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    store = pickle.loads((data.cache_dir() / "router" / f"final_{a.tag}.pkl").read_bytes())
    res = summarize(store, a.tasks, tuple(a.sizes), ours=a.ours)
    res["provenance"] = f"scripts/router_experiment.py --phase final --tag {a.tag}; scripts/router_aggregate.py"
    path = Path(a.out or ROOT / "results" / f"router_{a.tag}.json")
    path.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", path)
    rows = sorted(res["methods"].items(), key=lambda kv: -kv[1]["macro"]["value"])
    for n, m in rows:
        pt = " ".join(f"{m['per_task'][t]['value']:.4f}" for t in a.tasks)
        print(f"{n:38s} macro {m['macro']['value']:.4f} {m['macro']['ci']}  [{pt}] inv={m['invoked_mean']} lat={m['latency_ms_mean']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired differences against each table column's best method (bold rule of Tables 1, 3, 4).

Bold rule (user request 2026-11-16): a cell is bold when its method is the column's best or is
statistically tied with it, i.e. the paired 95% bootstrap interval of (best minus method)
includes 0. router_main.json and router_external.json hold differences against HealthRouter
only; this script adds the pairs against every column's best. It reads the stored test outputs
(cache/router/final_main.pkl, final_external.pkl) and reuses the bootstrap of the aggregators
(same class-stratified indices and seeds: 17+i in-source, 31+i external), so a pair that also
appears in router_main.json reproduces its interval (checked below). CPU only, no refitting.

Writes results/router_ties.json:
  {"main": {col: {"best": name, "pairs": {name: {"value", "ci", "tied"}}}},     Table 1
   "ablation": {"macro": {...}},                                                  Table 3
   "external": {cond|"mean": {...}}}                                              Table 4
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_aggregate import assemble, r  # noqa: E402
import router_descriptive as RD  # noqa: E402

from healthrouter.collection import data, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
T1 = ["best_single_overall", "best_single_per_task", "best_single_per_task_oof", "uniform_ensemble", "stacking",
      "greedy_ensemble_selection", "logme_selection", "modality_router_supervised", "modality_router_supervised_oof",
      "modality_router_zeroshot_biomedclip", "soft_moe", "sparse_topk_moe", "knora_u", "knora_e", "meta_des",
      "knn_router", "routerbench_mlp", "routellm_mf", "healthrouter", "healthrouter_harness"]
T3 = ["healthrouter", "abl_size1", "abl_size2", "abl_size3", "abl_size4", "abl_picks1", "abl_picks3",
      "abl_geometric_fusion", "abl_feat_biomedclip", "abl_feat_dinov2_large", "abl_feat_task_id",
      "healthrouter_labelfree", "hr_collection_mlp", "hr_candidate_mlp", "healthrouter_fallback"]
T4 = ["best_single_per_task", "uniform_ensemble", "stacking", "greedy_ensemble_selection",
      "modality_router_supervised", "modality_router_zeroshot_biomedclip", "soft_moe", "sparse_topk_moe",
      "knora_u", "meta_des", "knn_router", "routellm_mf", "healthrouter", "healthrouter_harness",
      "healthrouter_labelfree"]
CONDS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]


def labelfree_insource(store, tasks):
    """HealthRouter with the neighbours' voted task (router_descriptive.label_free_insource, N0a)."""
    out = {}
    for s in data.SPLIT_SEEDS:
        for f in range(data.N_OUTER):
            C = {t: RD.cand(t, s, f) for t in tasks}
            Ztr = np.concatenate([C[t]["Z_tr"] for t in tasks])
            Zte = np.concatenate([C[t]["Z_te"] for t in tasks])
            task_tr = np.concatenate([np.full(len(C[t]["y_tr"]), i) for i, t in enumerate(tasks)])
            rec = store[("healthrouter", s, f)]
            inferred = RD.vote(RD.neighbours(Ztr, Zte, int(rec["cfg"]["neighbours"])), task_tr, len(tasks))
            test, i0 = {}, 0
            for i, t in enumerate(tasks):
                idx, P, M = rec["test"][t]
                ok = inferred[i0:i0 + len(C[t]["y_te"])] == i
                i0 += len(C[t]["y_te"])
                test[t] = (idx, np.where(ok[:, None], P, RD.prior(C[t]["y_tr"], P.shape[1])[None, :]).astype(np.float32), M)
            out[("healthrouter_labelfree", s, f)] = {"cfg": rec["cfg"], "test": test}
    return out


def pairs(rep, value, names):
    """rep[name] -> (n_boot,) replicate array; value[name] -> point estimate."""
    best = max(names, key=lambda n: (round(value[n], 4), n == "healthrouter"))
    res = {"best": best, "best_value": r(value[best]), "pairs": {}}
    for n in names:
        if n == best:
            continue
        d = rep[best] - rep[n]
        lo, hi = metrics.ci(d)
        res["pairs"][n] = {"value": r(value[best] - value[n]), "ci": [r(lo), r(hi)], "tied": bool(r(lo) <= 0 <= r(hi))}
    return res


def insource():
    tasks = data.MED_TASKS
    store = pickle.loads((data.cache_dir() / "router" / "final_main.pkl").read_bytes())
    need = sorted(set(T1 + T3) - {"healthrouter_labelfree"})
    sub = {k: v for k, v in store.items() if k[0] in need}
    sub.update(labelfree_insource(store, tasks))
    names, seeds, ys, preds, _ = assemble(sub, tasks, (224,))
    boots = {t: metrics.strat_boot_indices(ys[t], N_BOOT, 17 + i) for i, t in enumerate(tasks)}
    rep = {n: {t: np.array([metrics.seed_mean_metric(t, ys[t], preds[n][t], b) for b in boots[t]]) for t in tasks}
           for n in names}
    pt = {n: {t: r(metrics.seed_mean_metric(t, ys[t], preds[n][t])) for t in tasks} for n in names}
    for n in names:
        rep[n]["macro"] = np.mean([rep[n][t] for t in tasks], 0)
        pt[n]["macro"] = r(np.mean([pt[n][t] for t in tasks]))
    main = {c: pairs({n: rep[n][c] for n in T1}, {n: pt[n][c] for n in T1}, T1) for c in tasks + ["macro"]}
    abl = {"macro": pairs({n: rep[n]["macro"] for n in T3}, {n: pt[n]["macro"] for n in T3}, T3)}
    # consistency with router_main.json (same bootstrap -> same interval for HealthRouter pairs)
    ref = json.loads((ROOT / "results" / "router_main.json").read_text())["differences_vs_healthrouter"]
    for n in ("greedy_ensemble_selection", "stacking", "knn_router", "abl_picks3"):
        d = rep["healthrouter"]["macro"] - rep[n]["macro"]
        mine = [r(x) for x in metrics.ci(d)]
        assert mine == ref[n]["macro"]["ci"], (n, mine, ref[n]["macro"]["ci"])
    return main, abl


def external():
    tasks = data.MED_TASKS
    store = RD.load_pkl("external")
    E = {t: dict(np.load(RD.CACHE / f"ext_{t}_s0.npz")) for t in tasks}
    ys = {t: E[t]["y_ext"] for t in tasks}
    boots = {t: metrics.strat_boot_indices(ys[t], N_BOOT, 31 + i) for i, t in enumerate(tasks)}
    Ztr = np.concatenate([E[t]["Z_tr"] for t in tasks])
    task_tr = np.concatenate([np.full(len(E[t]["y_tr"]), i) for i, t in enumerate(tasks)])
    rep, pt = {}, {}
    for cond in CONDS:
        P = {n: {t: store[(n, cond)]["test"][t][0] for t in tasks} for n in T4 if n != "healthrouter_labelfree"}
        rec = store[("healthrouter", cond)]
        Zte = np.concatenate([E[t][f"Z_ext_{cond}"] for t in tasks])
        inferred = RD.vote(RD.neighbours(Ztr, Zte, int(rec["cfg"]["neighbours"])), task_tr, len(tasks))
        lf, i0 = {}, 0
        for i, t in enumerate(tasks):
            Pt = rec["test"][t][0]
            ok = inferred[i0:i0 + len(ys[t])] == i
            i0 += len(ys[t])
            lf[t] = np.where(ok[:, None], Pt, RD.prior(E[t]["y_tr"], Pt.shape[1])[None, :])
        P["healthrouter_labelfree"] = lf
        rep[cond] = {n: np.mean([[metrics.primary(t, ys[t][b], P[n][t][b]) for b in boots[t]] for t in tasks], 0)
                     for n in T4}
        pt[cond] = {n: r(np.mean([r(metrics.primary(t, ys[t], P[n][t])) for t in tasks])) for n in T4}
    out = {c: pairs(rep[c], pt[c], T4) for c in CONDS}
    out["mean"] = pairs({n: np.mean([rep[c][n] for c in CONDS], 0) for n in T4},
                        {n: float(np.mean([pt[c][n] for c in CONDS])) for n in T4}, T4)
    ref = json.loads((ROOT / "results" / "router_external.json").read_text())["results"]
    for c in CONDS:
        for n in ("uniform_ensemble", "best_single_per_task"):
            d = rep[c]["healthrouter"] - rep[c][n]
            assert [r(x) for x in metrics.ci(d)] == ref[c][n]["diff_healthrouter_minus_this"]["ci"], (c, n)
    return out


def main():
    main_t, abl = insource()
    ext = external()
    out = {"rule": "best or statistically tied with the column's best: paired 95% bootstrap interval of "
                   "(best minus method) includes 0",
           "provenance": "scripts/router_tie_pairs.py (stored test outputs of final_main.pkl / final_external.pkl; "
                         "bootstrap as router_aggregate.py / router_external.py)",
           "main": main_t, "ablation": abl, "external": ext}
    path = ROOT / "results" / "router_ties.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)
    for tab, cols in (("main", main_t), ("ablation", abl), ("external", ext)):
        for c, v in cols.items():
            tied = [n for n, p in v["pairs"].items() if p["tied"]]
            print(f"{tab:9s} {c:15s} best={v['best']} ({v['best_value']}) tied={tied}")


if __name__ == "__main__":
    main()

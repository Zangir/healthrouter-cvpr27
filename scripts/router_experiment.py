#!/usr/bin/env python3
"""Run HealthRouter and all baselines on the frozen cross-fitting splits.

  --phase dev     validation only (seed 0), prints the best validation config per method
  --phase final   validation selection, then a single scoring of the test folds; writes
                  cache/router/final_<tag>.pkl for the aggregation script
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from healthrouter.collection import candidates, data, methods, metrics

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "results" / "router_plan.json").read_text())
GRID = PLAN["method"]["grid"]
DES_K = (5, 7, 10, 25, 50, 100)  # plan amendment 2026-11-17 (was k = 7)


def load_run(seed, fold, tasks, sizes=(224,), need_logme=False, zkey_encoder="dinov2_large"):
    run = {"tasks": tasks, "seed": seed * 10 + fold}
    for t in tasks:
        d = candidates.outer_fold(t, seed, fold, sizes)
        run[t] = dict(d)
        B, _ = data.load(t, "biomedclip", 224)
        for sp in ("tr", "va", "te"):
            run[t][f"B_{sp}"] = B[d[f"idx_{sp}"]]
        if data.feature_path(t, "ijepa", 224).exists():
            J, _ = data.load(t, "ijepa", 224)
            for sp in ("tr", "va", "te"):
                run[t][f"J_{sp}"] = J[d[f"idx_{sp}"]]
        if zkey_encoder:
            A, _ = data.load(t, zkey_encoder, 224)
            for sp in ("tr", "va", "te"):
                run[t][f"A_{sp}"] = A[d[f"idx_{sp}"]]
        if need_logme:
            run[t]["X_tr"] = [data.load(t, e, s)[0][d["idx_tr"]] for e, s in data.candidates(sizes)]
    for i, t in enumerate(tasks):
        for sp in ("tr", "va", "te"):
            run[t][f"T_{sp}"] = np.repeat(np.eye(len(tasks), dtype=np.float32)[[i]], len(run[t][f"y_{sp}"]), 0)
    run["n_cand"] = run[tasks[0]]["P_tr"].shape[1]
    return run


def families(run, splits, text_emb, cost):
    return {
        "healthrouter": lambda: methods.collection_router(run, splits, "knn"),
        "healthrouter_harness": lambda: methods.smart_harness(run, splits),
        **{f"harness_theta{th}": (lambda th=th: methods.smart_harness(run, splits, thetas=(th,)))
           for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01)},
        "healthrouter_fallback": lambda: methods.with_fallback(run, splits, methods.collection_router(run, splits, "knn")),
        "harness_fallback": lambda: methods.with_fallback(run, splits, methods.smart_harness(run, splits)),
        "hr_candidate_mlp": lambda: methods.healthrouter(run, splits, GRID),
        "hr_collection_mlp": lambda: methods.collection_router(run, splits, "mlp", neighbours=(0,), grid=GRID),
        # ablations of HealthRouter (each re-selects its remaining settings on validation)
        "abl_size1": lambda: methods.collection_router(run, splits, "knn", sizes=(1,)),
        "abl_size2": lambda: methods.collection_router(run, splits, "knn", sizes=(2,)),
        "abl_size3": lambda: methods.collection_router(run, splits, "knn", sizes=(3,)),
        "abl_size4": lambda: methods.collection_router(run, splits, "knn", sizes=(4,)),
        "abl_picks1": lambda: methods.collection_router(run, splits, "knn", picks=(1,)),
        "abl_picks3": lambda: methods.collection_router(run, splits, "knn", picks=(3,)),
        "abl_geometric_fusion": lambda: methods.collection_router(run, splits, "knn", fusion="geo"),
        "abl_feat_biomedclip": lambda: methods.collection_router(run, splits, "knn", Zkey="B"),
        "abl_feat_dinov2_large": lambda: methods.collection_router(run, splits, "knn", Zkey="A"),
        "abl_feat_ijepa": lambda: methods.collection_router(run, splits, "knn", Zkey="J"),
        "abl_harness_ijepa": lambda: methods.smart_harness(run, splits, Zkey="J"),
        "abl_feat_task_id": lambda: methods.collection_router(run, splits, "knn", Zkey="T"),
        **{f"cost_lam{lam}": (lambda lam=lam: methods.collection_router(
            run, splits, "knn", cost=cost / cost.max(), lams=(lam,))) for lam in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)},
        "best_single_per_task_oof": lambda: methods.best_single_per_task(run, splits, source="tr"),
        "modality_router_supervised_oof": lambda: methods.modality_router_supervised(run, splits, source="tr"),
        "best_single_per_task": lambda: methods.best_single_per_task(run, splits),
        "best_single_overall": lambda: methods.best_single_overall(run, splits),
        "uniform_ensemble": lambda: methods.uniform(run, splits),
        "stacking": lambda: methods.stacking(run, splits),
        "greedy_ensemble_selection": lambda: methods.greedy_ensemble(run, splits),
        "modality_router_supervised": lambda: methods.modality_router_supervised(run, splits),
        "modality_router_zeroshot_biomedclip": lambda: methods.modality_router_zeroshot(run, splits, text_emb),
        "soft_moe": lambda: methods.soft_moe(run, splits, GRID),
        "sparse_topk_moe": lambda: methods.sparse_moe(run, splits, GRID),
        "knn_router": lambda: methods.knn_router(run, splits),
        "knora_u": lambda: methods.des(run, splits, "knora_u", ks=DES_K),
        "knora_e": lambda: methods.des(run, splits, "knora_e", ks=DES_K),
        "meta_des": lambda: methods.des(run, splits, "meta_des", ks=DES_K),
        "fixed_order_harness": lambda: methods.fixed_order_harness(run, splits),
        "routerbench_mlp": lambda: methods.routerbench_mlp(run, splits, GRID),
        "routellm_mf": lambda: methods.routellm_mf(run, splits, GRID),
        "logme_selection": lambda: methods.logme_selection(run, splits),
        "oracle_single": lambda: methods.oracle_single(run, splits),
    }


def val_score(run, outputs):
    s = [metrics.primary(t, run[t]["y_va"], outputs["va"][t][0]) for t in run["tasks"]]
    inv = np.mean([outputs["va"][t][1].sum(1).mean() for t in run["tasks"]])
    return float(np.mean(s)), float(inv), s


def choose(run, results):
    best = None
    for order, (cfg, out) in enumerate(results):
        sc, inv, per = val_score(run, out)
        key = (round(sc, 6), -inv, -order)
        if best is None or key > best[0]:
            best = (key, cfg, out, sc, per, inv)
    return best


def dev_cross_score(run, results):
    """Unbiased dev estimate of a family *including* its validation selection: select the
    config on one stratified half of validation, score it on the other half, and swap."""
    halves = {t: data.stratified_holdout(run[t]["y_va"], 0.5, 99) for t in run["tasks"]}
    def score(out, t, h):
        idx = halves[t][h]
        return metrics.primary(t, run[t]["y_va"][idx], out["va"][t][0][idx])
    tot = []
    for sel, ev in ((0, 1), (1, 0)):
        best = max(range(len(results)), key=lambda i: (round(np.mean([score(results[i][1], t, sel)
                                                                        for t in run["tasks"]]), 6), -i))
        tot.append([score(results[best][1], t, ev) for t in run["tasks"]])
    return np.mean(tot, 0)


def text_embeddings(tasks):
    d = np.load(data.cache_dir() / "router" / "biomedclip_text.npz")
    names = list(d["tasks"])
    return d["emb"][[names.index(t) for t in tasks]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["dev", "final"], default="dev")
    p.add_argument("--tasks", nargs="+", default=data.MED_TASKS)
    p.add_argument("--sizes", nargs="+", type=int, default=[224])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--tag", default="main")
    a = p.parse_args()
    sizes = tuple(a.sizes)
    seeds = a.seeds if a.seeds is not None else ([0] if a.phase == "dev" else list(data.SPLIT_SEEDS))
    splits = ["va"] if a.phase == "dev" else ["va", "te"]
    text_emb = text_embeddings(a.tasks)
    cost = data.costs_ms(sizes)
    out_path = data.cache_dir() / "router" / f"final_{a.tag}.pkl"
    dev_path = data.cache_dir() / "router" / f"dev_{a.tag}.pkl"
    store = pickle.loads(out_path.read_bytes()) if (a.phase == "final" and out_path.exists()) else \
        (pickle.loads(dev_path.read_bytes()) if (a.phase == "dev" and dev_path.exists()) else {})
    for seed in seeds:
        for fold in range(data.N_OUTER):
            run = None
            fams_needed = None
            for name in (a.only or list(families({}, [], None, None))):
                key = (name, seed, fold)
                if key in store:
                    continue
                if run is None:
                    run = load_run(seed, fold, a.tasks, sizes, need_logme=True)
                    fams_needed = families(run, splits, text_emb, cost)
                t0 = time.time()
                results = fams_needed[name]()
                best = choose(run, results)
                _, cfg, out, sc, per, inv = best
                rec = {"cfg": cfg, "val_macro": sc, "val_per_task": per, "val_invoked": inv,
                       "seconds": time.time() - t0}
                if a.phase == "dev":
                    rec["dev_cross"] = dev_cross_score(run, results).tolist()
                if a.phase == "final":
                    rec["test"] = {t: (run[t]["idx_te"], out["te"][t][0].astype(np.float32),
                                       out["te"][t][1]) for t in a.tasks}
                store[key] = rec
                print(f"s{seed} f{fold} {name:38s} val={sc:.4f} "
                      f"[{' '.join(f'{x:.3f}' for x in per)}] inv={inv:.2f} cfg={cfg} "
                      f"({time.time() - t0:.1f}s)", flush=True)
            (out_path if a.phase == "final" else dev_path).write_bytes(pickle.dumps(store))
    if a.phase == "dev":
        names = sorted({k[0] for k in store}, key=lambda n: -np.mean([np.mean(store[k]["dev_cross"]) for k in store if k[0] == n]))
        print("\nDEV SUMMARY (validation only; 'honest' = select on one half, score the other)")
        for n in names:
            v = [store[k]["val_macro"] for k in store if k[0] == n]
            pt = np.mean([store[k]["val_per_task"] for k in store if k[0] == n], 0)
            inv = np.mean([store[k]["val_invoked"] for k in store if k[0] == n])
            dc = np.mean([store[k]["dev_cross"] for k in store if k[0] == n], 0)
            print(f"  {n:38s} sel-val {np.mean(v):.4f} | honest {np.mean(dc):.4f} {np.round(dc, 3)}  inv={inv:.2f}")


if __name__ == "__main__":
    main()

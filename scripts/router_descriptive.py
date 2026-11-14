#!/usr/bin/env python3
"""Descriptive analyses of plan amendment 4 (logged 2026-11-14, after all test scoring).

Every analysis reads cached arrays only: the cross-fitted candidate predictions
(cache/router/cand_<task>_s<seed>_f<fold>_224.npz, ext_<task>_s0.npz) and the stored
test outputs of the validation-chosen configurations (cache/router/final_*.pkl).
No configuration is chosen here and no main-table number changes.

  N0a  label-free end-to-end inference: the task is inferred from HealthRouter's own
       K retrieved neighbours (majority vote) and the inferred task's heads are used;
       an image with a wrong inferred task gets the null output (true task's meta-training
       class prior).
  N1   every candidate used alone, per task; per-image variation of the best candidate.
  N2   neighbourhood-estimated vs realised log-likelihood advantage of HealthRouter's
       selection over the task's validation-best single candidate.
  N6   per-image difficulty vs the number of models the harness calls.

Writes results/router_descriptive.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_aggregate import r, summarize  # noqa: E402

from healthrouter.collection import candidates, data, metrics, router  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = data.cache_dir() / "router"
EPS = 1e-4
CONDITIONS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]
N_BOOT = 2000


def load_pkl(tag):
    return pickle.loads((CACHE / f"final_{tag}.pkl").read_bytes())


def cand(task, seed, fold):
    return dict(np.load(CACHE / f"cand_{task}_s{seed}_f{fold}_224.npz"))


def unit(std, Z):
    B = std(Z)
    return B / np.linalg.norm(B, axis=1, keepdims=True)


def neighbours(Ztr, Zte, k):
    std = router.Standardizer().fit(Ztr)
    A, B = unit(std, Ztr), unit(std, Zte)
    return np.argsort(-(B @ A.T), 1)[:, :k]


def vote(order, task_tr, n_tasks):
    """Majority task among neighbours; ties go to the nearest neighbour's task."""
    lab = task_tr[order]
    counts = np.stack([(lab == i).sum(1) for i in range(n_tasks)], 1)
    best = counts.max(1, keepdims=True)
    tied = counts == best
    out = counts.argmax(1)
    for i in np.where(tied.sum(1) > 1)[0]:
        for j in lab[i]:
            if tied[i, j]:
                out[i] = j
                break
    return out


def prior(y_tr, k):
    return np.bincount(y_tr, minlength=k).astype(np.float32) / len(y_tr)


# ----------------------------------------------------------------------------- N0a, in-source
def label_free_insource(tag, tasks):
    store = load_pkl(tag)
    new = {}
    acc = {t: [] for t in tasks}
    for s in data.SPLIT_SEEDS:
        for f in range(data.N_OUTER):
            C = {t: cand(t, s, f) for t in tasks}
            Ztr = np.concatenate([C[t]["Z_tr"] for t in tasks])
            Zte = np.concatenate([C[t]["Z_te"] for t in tasks])
            task_tr = np.concatenate([np.full(len(C[t]["y_tr"]), i) for i, t in enumerate(tasks)])
            n_te = [len(C[t]["y_te"]) for t in tasks]
            for base, lf in (("healthrouter", "healthrouter_labelfree"),
                             ("healthrouter_harness", "harness_labelfree")):
                rec = store[(base, s, f)]
                K = int(rec["cfg"]["neighbours"])
                inferred = vote(neighbours(Ztr, Zte, K), task_tr, len(tasks))
                test = {}
                i0 = 0
                for i, t in enumerate(tasks):
                    idx, P, M = rec["test"][t]
                    assert np.array_equal(idx, C[t]["idx_te"]), (t, s, f)
                    ok = inferred[i0:i0 + n_te[i]] == i
                    i0 += n_te[i]
                    pri = prior(C[t]["y_tr"], P.shape[1])
                    P2 = np.where(ok[:, None], P, pri[None, :]).astype(np.float32)
                    test[t] = (idx, P2, M)
                    if base == "healthrouter":
                        acc[t].append((int(ok.sum()), int(len(ok))))
                new[(lf, s, f)] = {"cfg": {**rec["cfg"], "task_inference": "neighbour_vote"}, "test": test}
            for base in ("healthrouter", "healthrouter_harness"):
                new[(base, s, f)] = store[(base, s, f)]
    summ = summarize(new, tasks, (224,), ours="healthrouter")
    out = {"task_inference_accuracy": {t: r(sum(a for a, _ in acc[t]) / sum(n for _, n in acc[t]), 5)
                                       for t in tasks},
           "misinferred_images": {t: int(sum(n - a for a, n in acc[t])) for t in tasks},
           "image_evaluations": {t: int(sum(n for _, n in acc[t])) for t in tasks}}
    tot_ok = sum(a for t in tasks for a, _ in acc[t])
    tot = sum(n for t in tasks for _, n in acc[t])
    out["task_inference_accuracy"]["all"] = r(tot_ok / tot, 5)
    out["methods"] = {n: {"macro": summ["methods"][n]["macro"],
                          "per_task": {t: {k: summ["methods"][n]["per_task"][t][k] for k in ("value", "ci")}
                                       for t in tasks}}
                      for n in ("healthrouter", "healthrouter_labelfree", "healthrouter_harness", "harness_labelfree")}
    out["differences_vs_healthrouter"] = {n: summ["differences_vs_healthrouter"][n]
                                          for n in ("healthrouter_labelfree", "harness_labelfree")}
    # harness label-free vs harness with task heads
    return out


# ----------------------------------------------------------------------------- N0a, external
def label_free_external():
    tasks = data.MED_TASKS
    store = load_pkl("external")
    E = {t: dict(np.load(CACHE / f"ext_{t}_s0.npz")) for t in tasks}
    ys = {t: E[t]["y_ext"] for t in tasks}
    boots = {t: metrics.strat_boot_indices(ys[t], N_BOOT, 31 + i) for i, t in enumerate(tasks)}
    Ztr = np.concatenate([E[t]["Z_tr"] for t in tasks])
    task_tr = np.concatenate([np.full(len(E[t]["y_tr"]), i) for i, t in enumerate(tasks)])
    res = {}
    for cond in CONDITIONS:
        Zte = np.concatenate([E[t][f"Z_ext_{cond}"] for t in tasks])
        n_te = [len(ys[t]) for t in tasks]
        res[cond] = {"task_inference_accuracy": {}, "methods": {}}
        for base, lf in (("healthrouter", "healthrouter_labelfree"), ("healthrouter_harness", "harness_labelfree")):
            rec = store[(base, cond)]
            K = int(rec["cfg"]["neighbours"])
            inferred = vote(neighbours(Ztr, Zte, K), task_tr, len(tasks))
            preds = {}
            i0 = 0
            for i, t in enumerate(tasks):
                P, _ = rec["test"][t]
                ok = inferred[i0:i0 + n_te[i]] == i
                i0 += n_te[i]
                if base == "healthrouter":
                    res[cond]["task_inference_accuracy"][t] = r(ok.mean(), 5)
                preds[t] = (P, np.where(ok[:, None], P, prior(E[t]["y_tr"], P.shape[1])[None, :]))
            rep_b = {t: np.array([metrics.primary(t, ys[t][b], preds[t][0][b]) for b in boots[t]]) for t in tasks}
            rep_l = {t: np.array([metrics.primary(t, ys[t][b], preds[t][1][b]) for b in boots[t]]) for t in tasks}
            for name, which, rep in ((base, 0, rep_b), (lf, 1, rep_l)):
                pt = {t: metrics.primary(t, ys[t], preds[t][which]) for t in tasks}
                res[cond]["methods"][name] = {
                    "macro": {"value": r(np.mean(list(pt.values()))),
                              "ci": [r(x) for x in metrics.ci(np.mean([rep[t] for t in tasks], 0))]},
                    "per_task": {t: r(pt[t]) for t in tasks}}
            d = np.mean([rep_b[t] for t in tasks], 0) - np.mean([rep_l[t] for t in tasks], 0)
            res[cond]["methods"][lf]["diff_base_minus_this"] = {
                "value": r(res[cond]["methods"][base]["macro"]["value"] - res[cond]["methods"][lf]["macro"]["value"]),
                "ci": [r(x) for x in metrics.ci(d)]}
        allok = []
        rec = store[("healthrouter", cond)]
        inferred = vote(neighbours(Ztr, Zte, int(rec["cfg"]["neighbours"])), task_tr, len(tasks))
        truth = np.concatenate([np.full(n, i) for i, n in enumerate(n_te)])
        res[cond]["task_inference_accuracy"]["all"] = r((inferred == truth).mean(), 5)
    return res


# ----------------------------------------------------------------------------- N1
def single_candidates(tasks):
    store = load_pkl("main")
    new = {}
    beats = {t: [] for t in tasks}
    for s in data.SPLIT_SEEDS:
        for f in range(data.N_OUTER):
            new[("healthrouter", s, f)] = store[("healthrouter", s, f)]
            for t in tasks:
                C = cand(t, s, f)
                P_te, y_te = C["P_te"], C["y_te"]
                for c, enc in enumerate(data.ENCODERS):
                    key = (f"single_{enc}", s, f)
                    new.setdefault(key, {"cfg": {"candidate": enc}, "test": {}})
                    M = np.zeros((len(y_te), len(data.ENCODERS)), bool)
                    M[:, c] = True
                    new[key]["test"][t] = (C["idx_te"], P_te[:, c].astype(np.float32), M)
                _, _, Mb = store[("best_single_per_task", s, f)]["test"][t]
                cb = int(Mb[0].argmax())
                tp = P_te[np.arange(len(y_te)), :, y_te]
                beats[t].append(((tp > tp[:, [cb]] + 1e-9).any(1).sum(), len(y_te)))
    summ = summarize(new, tasks, (224,), ours="healthrouter")
    out = {"candidates": {}, "per_image_other_candidate_better": {}}
    for enc in data.ENCODERS:
        m = summ["methods"][f"single_{enc}"]
        out["candidates"][enc] = {"macro": m["macro"],
                                  "per_task": {t: {k: m["per_task"][t][k] for k in ("value", "ci")} for t in tasks},
                                  "latency_ms": m["latency_ms_mean"]}
    best_by_task = {t: max(data.ENCODERS, key=lambda e: out["candidates"][e]["per_task"][t]["value"]) for t in tasks}
    out["best_candidate_per_task"] = best_by_task
    out["n_distinct_best"] = len(set(best_by_task.values()))
    for t in tasks:
        a = sum(x for x, _ in beats[t])
        n = sum(x for _, x in beats[t])
        out["per_image_other_candidate_better"][t] = r(a / n, 4)
    a = sum(x for t in tasks for x, _ in beats[t])
    n = sum(x for t in tasks for _, x in beats[t])
    out["per_image_other_candidate_better"]["all"] = r(a / n, 4)
    ranked = sorted(data.ENCODERS, key=lambda e: -out["candidates"][e]["macro"]["value"])
    out["strongest"] = ranked[0]
    out["cheapest"] = data.ENCODERS[int(np.argmin(data.costs_ms()))]
    return out


# ----------------------------------------------------------------------------- N2 and N6
def boot_mean_ci(x, rng_seed=5):
    rng = np.random.RandomState(rng_seed)
    x = np.asarray(x, float)
    reps = [x[rng.randint(len(x), size=len(x))].mean() for _ in range(N_BOOT)]
    return [r(v) for v in metrics.ci(np.asarray(reps))]


def boot_spearman_ci(x, y, groups, rng_seed=7):
    """Cluster bootstrap over images (an image appears once per seed)."""
    rng = np.random.RandomState(rng_seed)
    uniq = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in uniq}
    reps = []
    for _ in range(400):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([pos[g] for g in pick])
        reps.append(spearmanr(x[ii], y[ii]).correlation)
    return [r(v) for v in metrics.ci(np.asarray(reps))]


def gain_and_difficulty(tasks):
    store = load_pkl("main")
    xs = {t: [] for t in tasks}
    ys_ = {t: [] for t in tasks}
    gid = {t: [] for t in tasks}
    calls = {t: [] for t in tasks}
    loss = {t: [] for t in tasks}
    err = {t: [] for t in tasks}
    hid = {t: [] for t in tasks}
    for s in data.SPLIT_SEEDS:
        for f in range(data.N_OUTER):
            C = {t: cand(t, s, f) for t in tasks}
            rec = store[("healthrouter", s, f)]
            cfg = rec["cfg"]
            K, size = int(cfg["neighbours"]), int(cfg["max_size"])
            Ztr = np.concatenate([C[t]["Z_tr"] for t in tasks])
            Zte = np.concatenate([C[t]["Z_te"] for t in tasks])
            U = np.concatenate([C[t]["P_tr"][np.arange(len(C[t]["y_tr"])), :, C[t]["y_tr"]] for t in tasks])
            lib = router.collection_library(len(data.ENCODERS), size)
            CU = router.collection_utility(U, lib)
            est = CU[neighbours(Ztr, Zte, K)].mean(1)                     # (n_te, |L|)
            single_col = {int(np.argmax(row)): j for j, row in enumerate(lib) if row.sum() == 1}
            i0 = 0
            harn = store[("healthrouter_harness", s, f)]
            for t in tasks:
                idx, P, _ = rec["test"][t]
                y = C[t]["y_te"]
                n = len(y)
                _, _, Mb = store[("best_single_per_task", s, f)]["test"][t]
                cb = int(Mb[0].argmax())
                e = est[i0:i0 + n]
                i0 += n
                xs[t].append(e.max(1) - e[:, single_col[cb]])
                p_hr = np.clip(P[np.arange(n), y], EPS, 1)
                pb = C[t]["P_te"][:, cb]
                p_b = np.clip(pb[np.arange(n), y], EPS, 1)
                ys_[t].append(np.log(p_hr) - np.log(p_b))
                gid[t].append(idx)
                _, _, Mh = harn["test"][t]
                calls[t].append(Mh.sum(1))
                loss[t].append(-np.log(p_b))
                err[t].append((pb.argmax(1) != y).astype(float))
                hid[t].append(idx)
    gain, diff = {}, {}
    allx, ally, allg = [], [], []
    for ti, t in enumerate(tasks):
        x, y, g = np.concatenate(xs[t]), np.concatenate(ys_[t]), np.concatenate(gid[t])
        allx.append(x)
        ally.append(y)
        allg.append(g + 100000 * ti)
        q = np.quantile(x, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, 9)
        bins = []
        for k in range(10):
            sel = b == k
            if sel.sum() == 0:
                continue
            bins.append({"x_mean": r(x[sel].mean()), "y_mean": r(y[sel].mean()),
                         "y_ci": boot_mean_ci(y[sel], 11 + k), "n": int(sel.sum())})
        slope, icpt = np.polyfit(x, y, 1)
        gain[t] = {"spearman": r(spearmanr(x, y).correlation), "spearman_ci": boot_spearman_ci(x, y, g),
                   "slope": r(slope), "intercept": r(icpt), "bins": bins, "n_pairs": int(len(x)),
                   "mean_realised_gain": r(y.mean()), "mean_realised_gain_ci": boot_mean_ci(y, 3)}
        c, lo, er, h = (np.concatenate(calls[t]), np.concatenate(loss[t]), np.concatenate(err[t]),
                        np.concatenate(hid[t]))
        one, more = c == 1, c >= 2
        diff[t] = {"share_one_call": r(one.mean()),
                   "logloss_one_call": r(lo[one].mean()) if one.any() else None,
                   "logloss_multi_call": r(lo[more].mean()) if more.any() else None,
                   "error_one_call": r(er[one].mean()) if one.any() else None,
                   "error_multi_call": r(er[more].mean()) if more.any() else None,
                   "logloss_one_call_ci": boot_mean_ci(lo[one], 21) if one.any() else None,
                   "logloss_multi_call_ci": boot_mean_ci(lo[more], 22) if more.any() else None,
                   "error_one_call_ci": boot_mean_ci(er[one], 23) if one.any() else None,
                   "error_multi_call_ci": boot_mean_ci(er[more], 24) if more.any() else None,
                   "spearman_calls_logloss": r(spearmanr(c, lo).correlation),
                   "spearman_calls_logloss_ci": boot_spearman_ci(c.astype(float), lo, h, 9),
                   "n_pairs": int(len(c))}
    x, y, g = np.concatenate(allx), np.concatenate(ally), np.concatenate(allg)
    gain["pooled"] = {"spearman": r(spearmanr(x, y).correlation), "spearman_ci": boot_spearman_ci(x, y, g),
                      "n_pairs": int(len(x))}
    return gain, diff


def main():
    tasks = data.MED_TASKS
    out = {"amendment": "router_plan.json amendment 4 (2026-11-14T18:02:18+04:00), descriptive, after test scoring",
           "provenance": "scripts/router_descriptive.py"}
    out["single_candidates"] = single_candidates(tasks)
    print("N1 done", out["single_candidates"]["best_candidate_per_task"], flush=True)
    out["label_free"] = {"in_distribution": label_free_insource("main", tasks)}
    print("N0a main", out["label_free"]["in_distribution"]["task_inference_accuracy"], flush=True)
    out["label_free"]["six_task"] = label_free_insource("general", tasks + data.NONMED_TASKS)
    print("N0a general", out["label_free"]["six_task"]["task_inference_accuracy"], flush=True)
    out["label_free"]["external"] = label_free_external()
    print("N0a external", {c: v["task_inference_accuracy"]["all"] for c, v in out["label_free"]["external"].items()},
          flush=True)
    gain, diff = gain_and_difficulty(tasks)
    out["gain_calibration"] = gain
    out["harness_difficulty"] = diff
    path = ROOT / "results" / "router_descriptive.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()

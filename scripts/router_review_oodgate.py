#!/usr/bin/env python3
"""Post-plan, post-hoc analysis R2 (plan amendment 8, external review request): OOD gate.

Two per-image scores decide whether HealthRouter's output is replaced by a fallback ensemble:
  maha     Mahalanobis distance of the standardised router features to the image's task, with
           a Ledoit-Wolf covariance fitted per task on in-source meta-training features;
  entropy  entropy of HealthRouter's fused output after per-task temperature scaling (NLL fit
           on in-source validation).
Per task the threshold is the q-th percentile of the score on in-source validation,
q in {90, 95, 99}; q is chosen on in-source validation with the common selection rule.
Fallbacks: the uniform ten-model ensemble, or the task's validation-chosen top-3 ensemble.
HealthRouter keeps its stored validation-chosen configuration.

Short resumable runs (one per external condition or in-source split seed), each written to
cache/router/review/oodgate_<run>.pkl + .json; `--aggregate` writes
results/router_review_oodgate.json from whatever runs exist (partial runs are listed).

  python scripts/router_review_oodgate.py --run ext_jpeg
  python scripts/router_review_oodgate.py --run ins_s0
  python scripts/router_review_oodgate.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_aggregate import r  # noqa: E402
from router_experiment import choose, load_run  # noqa: E402
from router_external import CONDITIONS, build_run  # noqa: E402

from healthrouter.collection import candidates, data, methods, metrics, router  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QS = (90, 95, 99)
SCORES = ("maha", "entropy")
FALLBACKS = ("uniform", "top3")
EPS = 1e-6


def review_dir() -> Path:
    d = data.cache_dir() / "router" / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- scores
def maha_scores(run, splits):
    """Per-task Ledoit-Wolf Mahalanobis distance on standardised router features."""
    std = router.Standardizer().fit(methods.pooled(run, "tr"))
    out = {sp: {} for sp in splits}
    for t in run["tasks"]:
        lw = LedoitWolf().fit(std(run[t]["Z_tr"]).astype(np.float64))
        for sp in splits:
            out[sp][t] = np.sqrt(np.maximum(lw.mahalanobis(std(run[t][f"Z_{sp}"]).astype(np.float64)), 0))
    return out


def fit_temperature(P, y):
    logp = np.log(np.clip(P, EPS, 1.0))

    def nll(logT):
        z = logp / np.exp(logT)
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        return float(np.mean(lse - z[np.arange(len(y)), y]))
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded").x))


def entropy_scores(run, hr, splits):
    out = {sp: {} for sp in splits}
    temps = {}
    for t in run["tasks"]:
        T = fit_temperature(hr["va"][t][0], run[t]["y_va"])
        temps[t] = T
        for sp in splits:
            z = np.log(np.clip(hr[sp][t][0], EPS, 1.0)) / T
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            out[sp][t] = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)
    return out, temps


# ----------------------------------------------------------------------------- fallbacks
def fallback_outputs(run, splits):
    uni = methods.uniform(run, splits)[0][1]
    w = {}
    for t in run["tasks"]:
        v = np.nan_to_num(methods.val_metric_per_candidate(run, t, "va"), nan=-1.0)
        w[t] = np.zeros(run["n_cand"])
        w[t][np.argsort(-v)[:3]] = 1.0
    top3 = methods.per_task_const(run, splits, w)
    return {"uniform": uni, "top3": top3}, {t: np.flatnonzero(w[t]).tolist() for t in run["tasks"]}


def gated(run, hr, score, fb, splits):
    """One (cfg, outputs) per q, as in methods.with_fallback but with per-task thresholds."""
    res = []
    for q in QS:
        thr = {t: float(np.percentile(score["va"][t], q)) for t in run["tasks"]}
        out = {}
        for sp in splits:
            out[sp] = {}
            for t in run["tasks"]:
                far = score[sp][t] > thr[t]
                P = np.where(far[:, None], fb[sp][t][0], hr[sp][t][0])
                M = np.where(far[:, None], fb[sp][t][1], hr[sp][t][1])
                out[sp][t] = (P, M)
        res.append(({"q": q, "thresholds": thr}, out))
    return res


def hr_outputs(run, cfg, splits):
    res = methods.collection_router(run, splits, "knn", sizes=(cfg["max_size"],),
                                    neighbours=(cfg["neighbours"],), picks=(cfg["picks"],))
    assert len(res) == 1 and res[0][0] == cfg, (res[0][0], cfg)
    return res[0][1]


def evaluate(run, hr, splits):
    """All gate variants for one run; returns {variant: {cfg, val_macro, test outputs, flags}}."""
    fbs, top3 = fallback_outputs(run, splits)
    ent, temps = entropy_scores(run, hr, splits)
    scores = {"maha": maha_scores(run, splits), "entropy": ent}
    out = {}
    for s in SCORES:
        for f in FALLBACKS:
            _, cfg, o, sc, per, inv = choose(run, gated(run, hr, scores[s], fbs[f], splits))
            flags = {t: (scores[s]["te"][t] > cfg["thresholds"][t]) for t in run["tasks"]}
            out[f"{s}_{f}"] = {"cfg": cfg, "val_macro": sc,
                               "test": {t: o["te"][t][0].astype(np.float32) for t in run["tasks"]},
                               "flags": flags}
    out["_meta"] = {"temperatures": temps, "top3": top3,
                    "flag_rate_all_q": {s: {q: {t: float((scores[s]["te"][t] > np.percentile(scores[s]["va"][t], q)).mean())
                                                for t in run["tasks"]} for q in QS} for s in SCORES}}
    return out


# ----------------------------------------------------------------------------- runs
def run_external(cond):
    tasks = data.MED_TASKS
    store = pickle.loads((data.cache_dir() / "router" / "final_external.pkl").read_bytes())
    run = build_run(tasks, cond)
    cfg = store[("healthrouter", cond)]["cfg"]
    hr = hr_outputs(run, cfg, ["va", "te"])
    for t in tasks:  # the recomputed HealthRouter output must equal the stored one
        assert np.allclose(hr["te"][t][0], store[("healthrouter", cond)]["test"][t][0], atol=1e-5), t
    return evaluate(run, hr, ["va", "te"])


def run_insource(seed):
    tasks = data.MED_TASKS
    store = pickle.loads((data.cache_dir() / "router" / "final_main.pkl").read_bytes())
    out = {}
    for fold in range(data.N_OUTER):
        run = load_run(seed, fold, tasks)
        cfg = store[("healthrouter", seed, fold)]["cfg"]
        hr = hr_outputs(run, cfg, ["va", "te"])
        for t in tasks:
            idx, P, _ = store[("healthrouter", seed, fold)]["test"][t]
            assert np.array_equal(idx, run[t]["idx_te"]) and np.allclose(hr["te"][t][0], P, atol=1e-5), (fold, t)
        ev = evaluate(run, hr, ["va", "te"])
        ev["_idx"] = {t: run[t]["idx_te"] for t in tasks}
        out[fold] = ev
        print(f"seed {seed} fold {fold}:", {k: (v["cfg"]["q"], round(v["val_macro"], 4)) for k, v in ev.items()
                                            if not k.startswith("_")}, flush=True)
    return out


def run_one(name):
    base = review_dir() / f"oodgate_{name}"
    if base.with_suffix(".json").exists():
        print("exists, skipped:", name)
        return
    res = run_external(name[4:]) if name.startswith("ext_") else run_insource(int(name[len("ins_s"):]))
    base.with_suffix(".pkl").write_bytes(pickle.dumps(res))
    if name.startswith("ext_"):
        summ = {k: {"q": v["cfg"]["q"], "val_macro": r(v["val_macro"]),
                    "flag_rate": {t: r(f.mean()) for t, f in v["flags"].items()}}
                for k, v in res.items() if not k.startswith("_")}
    else:
        summ = {f: {k: {"q": v["cfg"]["q"], "flag_rate": {t: r(x.mean()) for t, x in v["flags"].items()}}
                    for k, v in ev.items() if not k.startswith("_")} for f, ev in res.items()}
    base.with_suffix(".json").write_text(json.dumps({"run": name, "summary": summ}, indent=1) + "\n")
    print("wrote", base.with_suffix(".json"), flush=True)


# ----------------------------------------------------------------------------- aggregate
def boot_stats(t, y, boots, P):
    return np.array([metrics.primary(t, y[b], P[b]) for b in boots])


def aggregate():
    tasks = data.MED_TASKS
    rd = review_dir()
    variants = [f"{s}_{f}" for s in SCORES for f in FALLBACKS]
    out = {"amendment": "router_plan.json amendment 8 (post-plan, post-hoc, external review request)",
           "provenance": "scripts/router_review_oodgate.py", "variants": variants, "q_grid": list(QS),
           "external": {}, "in_source": {}, "missing_runs": []}
    # external
    ext = pickle.loads((data.cache_dir() / "router" / "final_external.pkl").read_bytes())
    ys = {t: candidates.external(t, CONDITIONS)["y_ext"] for t in tasks}
    boots = {t: metrics.strat_boot_indices(ys[t], 2000, 31 + i) for i, t in enumerate(tasks)}
    refs = {"healthrouter": "healthrouter", "distance_fallback": "healthrouter_fallback", "uniform": "uniform_ensemble"}
    for cond in CONDITIONS:
        p = rd / f"oodgate_ext_{cond}.pkl"
        if not p.exists():
            out["missing_runs"].append(f"ext_{cond}")
            continue
        res = pickle.loads(p.read_bytes())
        preds = {k: res[k]["test"] for k in variants}
        preds.update({k: {t: ext[(v, cond)]["test"][t][0] for t in tasks} for k, v in refs.items()})
        rep = {k: {t: boot_stats(t, ys[t], boots[t], preds[k][t]) for t in tasks} for k in preds}
        val = {k: {t: metrics.primary(t, ys[t], preds[k][t]) for t in tasks} for k in preds}
        c = {"scores": {}, "flag_rate": {}, "q": {}, "diffs": {}}
        for k in preds:
            mac = np.mean([rep[k][t] for t in tasks], 0)
            c["scores"][k] = {**{t: r(val[k][t]) for t in tasks}, "macro": r(np.mean(list(val[k].values()))),
                              "macro_ci": [r(x) for x in metrics.ci(mac)]}
        dist_flag = {t: ext[("healthrouter_fallback", cond)]["test"][t][1].all(1) for t in tasks}
        c["flag_rate"]["distance_fallback"] = {**{t: r(dist_flag[t].mean()) for t in tasks},
                                               "all": r(np.concatenate(list(dist_flag.values())).mean())}
        for k in variants:
            c["q"][k] = res[k]["cfg"]["q"]
            fl = res[k]["flags"]
            c["flag_rate"][k] = {**{t: r(fl[t].mean()) for t in tasks}, "all": r(np.concatenate([fl[t] for t in tasks]).mean())}
            c["diffs"][k] = {}
            for ref in refs:
                d = {t: {"value": r(val[k][t] - val[ref][t]), "ci": [r(x) for x in metrics.ci(rep[k][t] - rep[ref][t])]}
                     for t in tasks}
                md = np.mean([rep[k][t] for t in tasks], 0) - np.mean([rep[ref][t] for t in tasks], 0)
                d["macro"] = {"value": r(np.mean([val[k][t] - val[ref][t] for t in tasks])), "ci": [r(x) for x in metrics.ci(md)]}
                c["diffs"][k][f"minus_{ref}"] = d
        c["temperatures"] = {t: r(x, 3) for t, x in res["_meta"]["temperatures"].items()}
        c["top3_candidates"] = {t: [data.ENCODERS[i] for i in v] for t, v in res["_meta"]["top3"].items()}
        out["external"][cond] = c
    # in source
    main = pickle.loads((data.cache_dir() / "router" / "final_main.pkl").read_bytes())
    seeds = [s for s in data.SPLIT_SEEDS if (rd / f"oodgate_ins_s{s}.pkl").exists()]
    out["missing_runs"] += [f"ins_s{s}" for s in data.SPLIT_SEEDS if s not in seeds]
    if seeds:
        ys_in = {t: data.load(t, data.ENCODERS[0], 224)[1] for t in tasks}
        bo = {t: metrics.strat_boot_indices(ys_in[t], 2000, 17 + i) for i, t in enumerate(tasks)}
        runs = {s: pickle.loads((rd / f"oodgate_ins_s{s}.pkl").read_bytes()) for s in seeds}

        def assemble(get):
            P = {t: [] for t in tasks}
            for s in seeds:
                for t in tasks:
                    K = int(ys_in[t].max()) + 1
                    full = np.full((len(ys_in[t]), K), np.nan, np.float32)
                    for f in range(data.N_OUTER):
                        full[runs[s][f]["_idx"][t]] = get(s, f, t)
                    assert not np.isnan(full).any()
                    P[t].append(full)
            return P
        preds = {k: assemble(lambda s, f, t, k=k: runs[s][f][k]["test"][t]) for k in variants}
        preds["healthrouter"] = assemble(lambda s, f, t: main[("healthrouter", s, f)]["test"][t][1])
        preds["distance_fallback"] = assemble(lambda s, f, t: main[("healthrouter_fallback", s, f)]["test"][t][1])
        rep = {k: {t: np.array([metrics.seed_mean_metric(t, ys_in[t], preds[k][t], b) for b in bo[t]]) for t in tasks}
               for k in preds}
        val = {k: {t: metrics.seed_mean_metric(t, ys_in[t], preds[k][t]) for t in tasks} for k in preds}
        ins = {"seeds": seeds, "scores": {}, "diff_minus_healthrouter": {}, "flag_rate": {}}
        for k in preds:
            ins["scores"][k] = {**{t: r(val[k][t]) for t in tasks}, "macro": r(np.mean(list(val[k].values()))),
                                "macro_ci": [r(x) for x in metrics.ci(np.mean([rep[k][t] for t in tasks], 0))]}
            if k != "healthrouter":
                md = np.mean([rep[k][t] for t in tasks], 0) - np.mean([rep["healthrouter"][t] for t in tasks], 0)
                ins["diff_minus_healthrouter"][k] = {"value": r(np.mean([val[k][t] - val["healthrouter"][t] for t in tasks])),
                                                     "ci": [r(x) for x in metrics.ci(md)]}
        for k in variants:
            fl = {t: np.concatenate([runs[s][f][k]["flags"][t] for s in seeds for f in range(data.N_OUTER)]) for t in tasks}
            ins["flag_rate"][k] = {**{t: r(fl[t].mean()) for t in tasks}, "all": r(np.concatenate(list(fl.values())).mean())}
        fl = {t: np.concatenate([main[("healthrouter_fallback", s, f)]["test"][t][2].all(1) for s in seeds
                                 for f in range(data.N_OUTER)]) for t in tasks}
        ins["flag_rate"]["distance_fallback"] = {**{t: r(fl[t].mean()) for t in tasks}, "all": r(np.concatenate(list(fl.values())).mean())}
        out["in_source"] = ins
    out["status"] = "complete" if not out["missing_runs"] else "partial"
    (ROOT / "results" / "router_review_oodgate.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote results/router_review_oodgate.json;", out["status"], out["missing_runs"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="ext_<condition> or ins_s<seed>")
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    if a.run:
        run_one(a.run)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()

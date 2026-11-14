#!/usr/bin/env python3
"""Plan amendment 5: byte-identical images inside the evaluation pools (descriptive).

Reads the released HealthRouter-Bench files (rows keyed by sha256 of the source image bytes)
and the aggregate results. For each pool it counts duplicated hashes and label conflicts; for
the cross-fitted in-source pools it counts (seed, fold) cases where copies of one image sit in
both the training part (train_oof or validation) and the test fold; it then recomputes every
method's per-task score with all duplicated rows removed. Writes results/router_duplicates.json.

Usage: python scripts/router_duplicates.py --bench <path to hf_dataset>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from healthrouter.collection import metrics

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]


def r(x, d=4):
    return round(float(x), d)


def groups(ids, y):
    c = collections.defaultdict(list)
    for i, h in enumerate(ids):
        c[h].append(i)
    dup = {h: v for h, v in c.items() if len(v) > 1}
    conflict = sum(1 for v in dup.values() if len(set(y[v].tolist())) > 1)
    rows = sorted(i for v in dup.values() for i in v)
    return dup, conflict, np.asarray(rows, int)


def insource(B, tag, task, ref):
    C = np.load(B / "data" / "predictions" / f"{tag}_crossfit.npz")
    R = np.load(B / "data" / "router_outputs" / f"{tag}_test_outputs.npz")
    ids, y = C["ids"].astype(str), C["labels"].astype(int)
    dup, conflict, rows = groups(ids, y)
    straddle, test_rows = 0, 0
    for s in range(3):
        for f in range(5):
            te = set(C[f"s{s}f{f}_test_index"].tolist())
            tr = set(C[f"s{s}f{f}_train_oof_index"].tolist()) | set(C[f"s{s}f{f}_validation_index"].tolist())
            for v in dup.values():
                a, b = te & set(v), tr & set(v)
                if a and b:
                    straddle += 1
                    test_rows += len(a)
    keep = np.setdiff1d(np.arange(len(y)), rows)
    out = {"rows": int(len(y)), "duplicated_hashes": len(dup), "rows_in_duplicate_groups": int(len(rows)),
           "conflicting_label_groups": conflict, "straddling_cases_seed_fold": straddle,
           "test_rows_with_copy_in_training": test_rows, "methods": {}}
    deltas = []
    for m in R["methods"]:
        P = R[f"{m}__probs"]
        full = float(np.mean([metrics.primary(task, y, P[s]) for s in range(P.shape[0])]))
        dedup = float(np.mean([metrics.primary(task, y[keep], P[s][keep]) for s in range(P.shape[0])]))
        out["methods"][str(m)] = {"full": r(full), "without_duplicates": r(dedup), "difference": r(dedup - full)}
        if not str(m).startswith(("abl_", "cost_", "harness_theta", "oracle")):
            deltas.append(abs(dedup - full))
    out["max_abs_difference_main_methods"] = r(max(deltas))
    hr = out["methods"]["healthrouter"]
    out["healthrouter_difference"] = hr["difference"]
    order_full = sorted([m for m in out["methods"] if not m.startswith(("abl_", "cost_", "harness_theta", "oracle", "hr_", "healthrouter_fallback", "harness_fallback"))],
                        key=lambda m: -out["methods"][m]["full"])
    order_dedup = sorted(order_full, key=lambda m: -out["methods"][m]["without_duplicates"])
    out["healthrouter_rank_full"] = order_full.index("healthrouter") + 1
    out["healthrouter_rank_without_duplicates"] = order_dedup.index("healthrouter") + 1
    return out


def external(B, name, task):
    E = np.load(B / "data" / "external" / f"{name}.npz")
    ids, y = E["ids"].astype(str), E["labels"].astype(int)
    dup, conflict, rows = groups(ids, y)
    keep = np.setdiff1d(np.arange(len(y)), rows)
    out = {"rows": int(len(y)), "duplicated_hashes": len(dup), "rows_in_duplicate_groups": int(len(rows)),
           "conflicting_label_groups": conflict, "conditions": {}}
    mx = 0.0
    for cond in CONDITIONS:
        res = {}
        for k in E.files:
            if k.startswith(f"{cond}__") and k.endswith("__probs"):
                m = k[len(cond) + 2:-len("__probs")]
                full = metrics.primary(task, y, E[k])
                dedup = metrics.primary(task, y[keep], E[k][keep])
                res[m] = {"full": r(full), "without_duplicates": r(dedup), "difference": r(dedup - full)}
                if "fallback" not in m and m != "oracle_single":
                    mx = max(mx, abs(dedup - full))
        out["conditions"][cond] = res
    out["max_abs_difference_methods"] = r(mx)
    out["healthrouter_difference_by_condition"] = {c: out["conditions"][c]["healthrouter"]["difference"] for c in CONDITIONS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    a = ap.parse_args()
    B = Path(a.bench)
    res = {"amendment": "router_plan.json amendment 5 (2026-11-14T19:13:38+04:00), descriptive",
           "provenance": "scripts/router_duplicates.py on the released HealthRouter-Bench files",
           "pathology_pcam": insource(B, "pcam", "pathology", "router_main"),
           "mammography_cbis_ddsm": insource(B, "cbis_ddsm", "mammography", "router_main"),
           "external_fundus_idrid": external(B, "external_idrid", "fundus")}
    (ROOT / "results" / "router_duplicates.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {kk: vv for kk, vv in v.items() if kk not in ("methods", "conditions")})


if __name__ == "__main__":
    main()

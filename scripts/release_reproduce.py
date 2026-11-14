#!/usr/bin/env python3
"""Reproduce the paper's numbers that depend only on the released HealthRouter-Bench files.

Reads <bench>/data/ (no images, no cache) and recomputes, with the paper's metric code:
  * Tab. 1 pathology and mammography columns for all 20 routers and ensembles (+ ablations),
  * the EuroSAT column of the six-task table,
  * the per-cohort external scores (Camelyon17-WILDS, IDRiD, CMMD) under all five conditions,
  * the single-encoder scores of Tab. 2 for pathology and mammography,
and compares each value with the aggregate result files shipped in <bench>/protocol/.
Columns that need Shenzhen, Montgomery, APTOS or DTD (and therefore every macro score)
require rebuilding those sources from your own download (see the code README).

Usage: python scripts/release_reproduce.py --bench <path to hf_dataset> [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from healthrouter.collection import metrics

CONDITIONS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]
ENC = ["imagenet_convnext", "dinov2_base", "dinov2_large", "siglip2_base", "biomedclip", "rad_dino",
       "phikon", "phikon_v2", "retfound", "mammo_clip"]


def seed_mean(task, y, P):
    return float(np.mean([metrics.primary(task, y, P[s]) for s in range(P.shape[0])]))


def compare(rows, name, got, ref, tol=6e-5):
    ok = ref is not None and abs(got - ref) <= tol
    rows.append({"item": name, "reproduced": round(got, 4), "paper_file": ref, "match": bool(ok)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    B = Path(a.bench)
    prot = {n: json.loads((B / "protocol" / f"{n}.json").read_text())
            for n in ("router_main", "router_general", "router_external", "router_descriptive")}
    rows = []
    for tag, task, ref in (("pcam", "pathology", "router_main"), ("cbis_ddsm", "mammography", "router_main"),
                           ("eurosat", "eurosat", "router_general")):
        R = np.load(B / "data" / "router_outputs" / f"{tag}_test_outputs.npz")
        y = np.load(B / "data" / "predictions" / f"{tag}_crossfit.npz")["labels"].astype(int)
        for m in R["methods"]:
            got = seed_mean(task, y, R[f"{m}__probs"])
            refv = prot[ref]["methods"].get(str(m), {}).get("per_task", {}).get(task, {}).get("value")
            compare(rows, f"{ref}:{m}:{task}", got, refv)
        if task in ("pathology", "mammography"):
            C = np.load(B / "data" / "predictions" / f"{tag}_crossfit.npz")
            seeds = sorted({int(k[1]) for k in C.files if k.startswith("s") and k.endswith("_test_index")})
            for c, enc in enumerate(ENC):
                P = np.zeros((len(seeds), len(y), C["s0f0_test_probs"].shape[2]), np.float32)
                for si, s in enumerate(seeds):
                    for f in range(5):
                        P[si, C[f"s{s}f{f}_test_index"]] = C[f"s{s}f{f}_test_probs"][:, c]
                got = seed_mean(task, y, P)
                refv = prot["router_descriptive"]["single_candidates"]["candidates"][enc]["per_task"][task]["value"]
                compare(rows, f"router_descriptive:single_{enc}:{task}", got, refv)
    for name, task in (("external_camelyon17_ood", "pathology"), ("external_idrid", "fundus"),
                       ("external_cmmd", "mammography")):
        E = np.load(B / "data" / "external" / f"{name}.npz")
        y = E["labels"].astype(int)
        for cond in CONDITIONS:
            for k in E.files:
                if k.startswith(f"{cond}__") and k.endswith("__probs"):
                    m = k[len(cond) + 2:-len("__probs")]
                    got = metrics.primary(task, y, E[k])
                    refv = prot["router_external"]["results"][cond].get(m, {}).get("per_task", {}).get(task, {}).get("value")
                    compare(rows, f"router_external:{cond}:{m}:{task}", got, refv)
    n_ok = sum(r["match"] for r in rows)
    print(f"reproduced {n_ok}/{len(rows)} values from the released files")
    for r in rows:
        if not r["match"]:
            print("  MISMATCH", r)
    if a.out:
        Path(a.out).write_text(json.dumps({"matched": n_ok, "total": len(rows), "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()

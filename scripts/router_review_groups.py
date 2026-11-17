#!/usr/bin/env python3
"""Post-plan analysis R3b, step 1 (plan amendment 9): patient / slide groups for the in-source
mammography (CBIS-DDSM) and pathology (PatchCamelyon) pools, keyed by pool row.

CBIS-DDSM: rows are keyed by sha256 of the source PNG bytes (released *_labels.csv); local file
names are <DICOM series UID>_1-1.png; series UIDs are mapped to patient IDs through the public
TCIA case-description CSVs (mass/calc x train/test). PatchCamelyon: rows are the seed-2026
balanced 2,000-patch subsample of the official test split (healthrouter/grid.py::_pcam_pool),
verified by sha256 of the patch bytes against the released row ids; the slide is the `wsi`
column of camelyonpatch_level_2_split_test_meta.csv.

Run with `python -I` (downloaded CSVs are untrusted data):
  python -I scripts/router_review_groups.py --data <benchmark root containing data/>
      --labels <dir with cbis_ddsm_labels.csv, pcam_labels.csv>
      --tcia <dir with the four TCIA CSVs> --pcam-meta <camelyonpatch_level_2_split_test_meta.csv>
Writes results/router_review_groups.json (coverage only, no identifiers) and the row->group
arrays to $HEALTHROUTER_CACHE/router/review/groups.npz.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

TCIA = ["mass_case_description_train_set.csv", "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv", "calc_case_description_test_set.csv"]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cbis(data_root, labels, tcia):
    want = [r["image_sha256"] for r in rows(Path(labels) / "cbis_ddsm_labels.csv")]
    files = glob.glob(str(Path(data_root) / "data" / "cbis_ddsm_1024" / "**" / "*.png"), recursive=True)
    by_hash = {}
    for p in files:
        by_hash.setdefault(hashlib.sha256(open(p, "rb").read()).hexdigest(), []).append(p)
    series_to_patient = {}
    for name in TCIA:
        for r in rows(Path(tcia) / name):
            for col in ("image file path", "cropped image file path", "ROI mask file path"):
                parts = r.get(col, "").strip().strip('"').split("/")
                if len(parts) >= 3:
                    series_to_patient.setdefault(parts[2].strip(), set()).add(r["patient_id"].strip())
                    series_to_patient.setdefault(parts[1].strip(), set()).add(r["patient_id"].strip())
    group, how = [], {"matched_file": 0, "series_in_tcia": 0, "unique_patient": 0}
    ambiguous_files = 0
    for h in want:
        ps = by_hash.get(h, [])
        if not ps:
            group.append(None)
            continue
        how["matched_file"] += 1
        uids = {Path(p).stem.rsplit("_", 1)[0] for p in ps}
        pats = set().union(*[series_to_patient.get(u, set()) for u in uids])
        ambiguous_files += len(ps) > 1
        if pats:
            how["series_in_tcia"] += 1
        if len(pats) == 1:
            how["unique_patient"] += 1
            group.append(next(iter(pats)))
        else:
            group.append(None if not pats else "|".join(sorted(pats)))
    return want, group, how, ambiguous_files


def pcam(data_root, labels, meta):
    import h5py
    want = [r["image_sha256"] for r in rows(Path(labels) / "pcam_labels.csv")]
    base = Path(data_root) / "data"
    with h5py.File(base / "camelyonpatch_level_2_split_test_y.h5") as f:
        y = f["y"][:].reshape(-1).astype(np.float32)
    rng = np.random.RandomState(2026)
    chosen = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], 1000, replace=False) for c in (0, 1)]))
    with h5py.File(base / "camelyonpatch_level_2_split_test_x.h5") as f:
        x = f["x"][chosen]
    got = [hashlib.sha256(x[i].tobytes()).hexdigest() for i in range(len(chosen))]
    m = rows(meta)
    assert len(m) == len(y), (len(m), len(y))
    wsi = [m[i]["wsi"].strip() for i in chosen]
    return want, got, wsi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--tcia", required=True)
    p.add_argument("--pcam-meta", required=True)
    a = p.parse_args()
    out = {"amendment": "router_plan.json amendment 9, R3b step 1", "provenance": "scripts/router_review_groups.py"}
    want, group, how, amb = cbis(a.data, a.labels, a.tcia)
    ok = all(g is not None and "|" not in g for g in group)
    pats = sorted({g for g in group if g})
    out["mammography_cbis_ddsm"] = {"rows": len(want), **how, "rows_with_several_local_copies": amb,
                                    "patients": len(pats), "mapping_complete": ok,
                                    "max_rows_per_patient": max(group.count(g) for g in pats) if pats else 0}
    pw, pg, wsi = pcam(a.data, a.labels, a.pcam_meta)
    pok = pw == pg
    out["pathology_pcam"] = {"rows": len(pw), "sha256_rows_verified": int(sum(x == y for x, y in zip(pw, pg))),
                             "mapping_complete": pok, "slides": len(set(wsi)),
                             "max_rows_per_slide": max(wsi.count(s) for s in set(wsi))}
    cache = Path(os.environ.get("HEALTHROUTER_CACHE", "~/.cache/healthrouter")).expanduser() / "router" / "review"
    cache.mkdir(parents=True, exist_ok=True)
    enc = lambda g: np.unique(np.asarray(g), return_inverse=True)[1]  # noqa: E731
    np.savez(cache / "groups.npz",
             mammography=enc([g if g else f"row{i}" for i, g in enumerate(group)]) if ok else np.zeros(0, int),
             pathology=enc(wsi) if pok else np.zeros(0, int))
    (ROOT / "results" / "router_review_groups.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

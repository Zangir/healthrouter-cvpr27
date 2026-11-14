#!/usr/bin/env python3
"""Build the HealthRouter-Bench data release (derived artefacts only; no images).

Only sources whose terms allow redistributing derived outputs are exported:
  in-source   pathology   PatchCamelyon official test, balanced 2,000 patches   CC0 1.0
              mammography CBIS-DDSM test mirror, exact-deduplicated (614)       CC BY 3.0 (TCIA)
  external    pathology   Camelyon17-WILDS OOD test hospital (1,000)            CC0 1.0
              fundus      IDRiD disease grading (516)                           CC BY 4.0
              mammography CMMD, MLO views via TOMPEI-CMMD (396)                 CC BY 4.0
  non-medical eurosat     EuroSAT RGB, 200 per class (2,000)                    MIT
Shenzhen and Montgomery (no licence stated by the host), APTOS 2019 (Kaggle competition
rules) and DTD ("for research purposes", no licence) are NOT exported; the code rebuilds
their artefacts from the user's own download.

Every item is keyed by its row (array position) and by sha256 of its source image bytes (PatchCamelyon and in-memory
patches: sha256 of the raw uint8 array bytes as stored in the official HDF5 file). Source
file names, patient identifiers and mirror metadata are never written. The item order is
checked against the cached evaluation pools before anything is exported.

Usage:
  HEALTHROUTER_DATA_ROOT=... HEALTHROUTER_EXTERNAL_ROOT=... HEALTHROUTER_CACHE=... \
  python scripts/release_build.py --out <dir>/hf_dataset
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from healthrouter import external as ext_src
from healthrouter.collection import data as cdata
from healthrouter.data import cbis_split, nonoverlap_mask
from healthrouter.grid import _pcam_pool

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]
CANDS = cdata.ENCODERS
NONMED_ROOT = Path(__import__("os").environ.get("HEALTHROUTER_NONMED_ROOT", str(ROOT.parents[1] / "external_data" / "nonmedical")))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(path: str) -> str:
    return sha(Path(path).read_bytes())


def square(img: Image.Image, size=224) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), np.uint8)


def check_order(name, decoded, pool, n_check=None):
    """Exact pixel check of the re-derived order against the cached pool."""
    idx = range(len(decoded)) if n_check is None else np.linspace(0, len(decoded) - 1, n_check).astype(int)
    bad = [int(i) for i in idx if not np.array_equal(decoded[i], pool[i])]
    print(f"order check {name}: {len(list(idx))} items compared, {len(bad)} mismatches", flush=True)
    assert not bad, (name, bad[:5])
    return len(list(idx))


# ----------------------------------------------------------------------------- id maps
def ids_pathology(root):
    images, y = _pcam_pool(root)
    pool = np.load(glob.glob(str(cdata.cache_dir() / "pool_pathology_224_*.npy"))[0], mmap_mode="r")
    n = check_order("pathology", [square(Image.fromarray(np.asarray(im))) for im in images], pool, 200)
    return [sha(np.ascontiguousarray(im).tobytes()) for im in images], np.asarray(y), n


def ids_mammography(root):
    train, _, test, y = cbis_split(root)
    keep = nonoverlap_mask(train, test)
    files = [f for f, k in zip(test, keep) if k]
    pool = np.load(glob.glob(str(cdata.cache_dir() / "pool_mammography_224_*.npy"))[0], mmap_mode="r")
    n = check_order("mammography", [square(Image.open(f)) for f in files], pool, None)
    return [file_sha(f) for f in files], y[keep].astype(int), n


def ids_eurosat():
    rng = np.random.RandomState(0)
    root = NONMED_ROOT / "eurosat" / "2750"
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    files, ys = [], []
    for k, c in enumerate(classes):
        fs = sorted(glob.glob(str(root / c / "*.jpg")))
        pick = rng.choice(len(fs), 200, replace=False)
        for i in sorted(pick):
            files.append(fs[i])
            ys.append(k)
    pool = np.load(cdata.cache_dir() / "pool_eurosat_224.npz")["x"]
    n = check_order("eurosat", [square(Image.open(f)) for f in files], pool, None)
    return [file_sha(f) for f in files], np.asarray(ys), classes, n


def ids_external():
    """External cohorts in the order of the gate image arrays (dispatch_items lane order)."""
    items = np.load(cdata.cache_dir() / "dispatch_items.npz")
    out = {}
    src = {"pathology": ext_src.pathology_camelyon17(), "fundus": ext_src.fundus_idrid(),
           "mammography": ext_src.mammography_cmmd()}
    for task, (blobs, y, _groups) in src.items():
        assert np.array_equal(np.asarray(y).astype(int), items[f"y_{task}"].astype(int)), task
        out[task] = ([sha(b) for b in blobs], np.asarray(y).astype(int))
        print(f"order check external {task}: labels identical for {len(y)} items", flush=True)
    return out


# ----------------------------------------------------------------------------- exports
def assemble_outputs(store, task, seeds, n, methods=None):
    """{method: (probs (n_seeds, n, K), invoked (n_seeds, n, C))} from the stored test outputs."""
    names = sorted({k[0] for k in store if len(k) == 3})
    if methods:
        names = [m for m in names if m in methods]
    out = {}
    for m in names:
        P = M = None
        for si, s in enumerate(seeds):
            for f in range(cdata.N_OUTER):
                idx, Pt, Mt = store[(m, s, f)]["test"][task]
                if P is None:
                    P = np.full((len(seeds), n, Pt.shape[1]), np.nan, np.float32)
                    M = np.zeros((len(seeds), n, Mt.shape[1]), bool)
                P[si, idx], M[si, idx] = Pt, Mt
        assert not np.isnan(P).any(), (m, task)
        out[m] = (P, M)
    return out


def export_insource(out_dir, task, ids, y, tag, pkl):
    seeds = list(cdata.SPLIT_SEEDS)
    n = len(ids)
    arr = {"ids": np.asarray(ids), "labels": y.astype(np.int16)}
    rows = []
    Z = None
    for s in seeds:
        for f in range(cdata.N_OUTER):
            c = np.load(cdata.cache_dir() / "router" / f"cand_{task}_s{s}_f{f}_224.npz")
            assert np.array_equal(c["y_te"], y[c["idx_te"]]), (task, s, f)
            for sp, role in (("tr", "train_oof"), ("va", "validation"), ("te", "test")):
                arr[f"s{s}f{f}_{role}_index"] = c[f"idx_{sp}"].astype(np.int32)
                arr[f"s{s}f{f}_{role}_probs"] = c[f"P_{sp}"].astype(np.float32)
                rows += [(int(i), ids[i], s, f, role) for i in c[f"idx_{sp}"]]
            arr[f"s{s}f{f}_C"] = c["C"]
            if Z is None:
                Z = np.zeros((n, c["Z_tr"].shape[1]), np.float32)
            for sp in ("tr", "va", "te"):
                Z[c[f"idx_{sp}"]] = c[f"Z_{sp}"]
    preds = out_dir / "predictions" / f"{tag}_crossfit.npz"
    preds.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preds, **arr)
    feats = out_dir / "features" / f"{tag}_router_features_convnext_t.npz"
    feats.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feats, ids=np.asarray(ids), z=Z)
    split = out_dir / "splits" / f"{tag}_splits.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    with split.open("w") as fh:
        fh.write("row,image_sha256,seed,outer_fold,role\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
    lab = out_dir / "splits" / f"{tag}_labels.csv"
    with lab.open("w") as fh:
        fh.write("row,image_sha256,label\n")
        for r, (i, l) in enumerate(zip(ids, y)):
            fh.write(f"{r},{i},{int(l)}\n")
    outs = assemble_outputs(pkl, task, seeds, n)
    ro = {"ids": np.asarray(ids), "methods": np.asarray(sorted(outs))}
    for m, (P, M) in outs.items():
        ro[f"{m}__probs"], ro[f"{m}__invoked"] = P, M
    rpath = out_dir / "router_outputs" / f"{tag}_test_outputs.npz"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rpath, **ro)
    return [preds, feats, split, lab, rpath]


def export_external(out_dir, ext_ids, ext_pkl):
    files = []
    for task, (ids, y) in ext_ids.items():
        e = np.load(cdata.cache_dir() / "router" / f"ext_{task}_s0.npz")
        assert np.array_equal(e["y_ext"], y), task
        arr = {"ids": np.asarray(ids), "labels": y.astype(np.int16)}
        for cond in CONDITIONS:
            arr[f"{cond}_candidate_probs"] = e[f"P_ext_{cond}"].astype(np.float32)
            arr[f"{cond}_router_features"] = e[f"Z_ext_{cond}"].astype(np.float32)
            methods = sorted(k[0] for k in ext_pkl if k[1] == cond)
            for m in methods:
                P, M = ext_pkl[(m, cond)]["test"][task]
                arr[f"{cond}__{m}__probs"] = np.asarray(P, np.float32)
                arr[f"{cond}__{m}__invoked"] = np.asarray(M, bool)
        name = {"pathology": "external_camelyon17_ood", "fundus": "external_idrid",
                "mammography": "external_cmmd"}[task]
        p = out_dir / "external" / f"{name}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **arr)
        lab = out_dir / "external" / f"{name}_labels.csv"
        with lab.open("w") as fh:
            fh.write("row,image_sha256,label\n")
            for r, (i, l) in enumerate(zip(ids, y)):
                fh.write(f"{r},{i},{int(l)}\n")
        files += [p, lab]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import os
    root = os.environ["HEALTHROUTER_DATA_ROOT"]
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    d = out / "data"
    log = {"order_checks": {}}
    main_pkl = pickle.loads((cdata.cache_dir() / "router" / "final_main.pkl").read_bytes())
    gen_pkl = pickle.loads((cdata.cache_dir() / "router" / "final_general.pkl").read_bytes())
    ext_pkl = pickle.loads((cdata.cache_dir() / "router" / "final_external.pkl").read_bytes())
    files = []
    ids, y, n = ids_pathology(root)
    log["order_checks"]["pathology"] = n
    files += export_insource(d, "pathology", ids, y, "pcam", main_pkl)
    ids, y, n = ids_mammography(root)
    log["order_checks"]["mammography"] = n
    files += export_insource(d, "mammography", ids, y, "cbis_ddsm", main_pkl)
    ids, y, classes, n = ids_eurosat()
    log["order_checks"]["eurosat"] = n
    files += export_insource(d, "eurosat", ids, y, "eurosat", gen_pkl)
    (d / "splits" / "eurosat_classes.json").write_text(json.dumps(classes) + "\n")
    ext_ids = ids_external()
    log["order_checks"]["external"] = {t: len(v[0]) for t, v in ext_ids.items()}
    files += export_external(d, ext_ids, ext_pkl)
    # protocol, aggregate results and candidate pool description
    prot = out / "protocol"
    prot.mkdir(exist_ok=True)
    for name in ("router_plan.json", "router_main.json", "router_main_harness.json", "router_external.json",
                 "router_general.json", "router_ood_flag.json", "router_descriptive.json", "router_paper_numbers.json"):
        shutil.copy2(ROOT / "results" / name, prot / name)
    prov = json.loads((ROOT / "results" / "external_provenance.json").read_text())
    pool = {"candidates": CANDS, "cost_ms_per_image": [round(float(c), 2) for c in cdata.costs_ms()],
            "router_encoder": cdata.ROUTER_ENCODER, "input": "square bicubic resize to 224 px, RGB",
            "head": "standardised logistic regression per (encoder, task); C chosen by inner 5-fold OOF log-loss",
            "model_sources": {k: {kk: v[kk] for kk in ("hub", "revision", "card_license") if kk in v}
                              for k, v in prov["models"].items()},
            "other_sources": {"imagenet_convnext": "timm convnext_tiny ImageNet-1k weights"}}
    (prot / "candidate_pool.json").write_text(json.dumps(pool, indent=1) + "\n")
    (out / "build_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("files:", len(files))


if __name__ == "__main__":
    main()

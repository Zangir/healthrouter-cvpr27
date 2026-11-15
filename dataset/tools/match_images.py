#!/usr/bin/env python3
"""Join HealthRouter-Bench rows to images in YOUR OWN download of a source dataset.

Every released row is keyed by the sha256 of the source image bytes (for PatchCamelyon,
the raw uint8 bytes of one 96x96x3 patch as stored in the official HDF5 test file).
This script hashes your local copy and writes a CSV mapping image_sha256 -> your local
reference, for the rows present in a released labels file. Nothing leaves your machine.

Examples:
  python tools/match_images.py --labels data/splits/cbis_ddsm_labels.csv --folder /path/to/cbis_images
  python tools/match_images.py --labels data/splits/eurosat_labels.csv --folder /path/to/EuroSAT/2750
  python tools/match_images.py --labels data/splits/pcam_labels.csv \
      --pcam-h5 /path/to/camelyonpatch_level_2_split_test_x.h5
  python tools/match_images.py --labels data/external/external_idrid_labels.csv \
      --parquet "/path/to/idrid-disease-grading/data/*.parquet"
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folder")
    ap.add_argument("--pcam-h5")
    ap.add_argument("--parquet")
    ap.add_argument("--out", default="matched.csv")
    a = ap.parse_args()
    want = {r["image_sha256"] for r in csv.DictReader(open(a.labels))}
    found = {}
    if a.folder:
        for dirpath, _, names in os.walk(a.folder):
            for n in names:
                p = os.path.join(dirpath, n)
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()
                if h in want:
                    found[h] = p
    if a.pcam_h5:
        import h5py
        with h5py.File(a.pcam_h5) as f:
            x = f["x"]
            for i in range(len(x)):
                h = hashlib.sha256(x[i].tobytes()).hexdigest()
                if h in want:
                    found[h] = f"pcam_test_index={i}"
    if a.parquet:
        import pyarrow.parquet as pq
        for path in sorted(glob.glob(a.parquet)):
            col = pq.read_table(path, columns=["image"]).column("image").to_pylist()
            for i, r in enumerate(col):
                h = hashlib.sha256(r["bytes"]).hexdigest()
                if h in want:
                    found[h] = f"{os.path.basename(path)}#row={i}"
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_sha256", "local_reference"])
        for h, ref in sorted(found.items()):
            w.writerow([h, ref])
    print(f"matched {len(found)}/{len(want)} released rows -> {a.out}")


if __name__ == "__main__":
    main()

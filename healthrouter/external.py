"""External (held-out institution) sources for all five lanes.

Chest (Montgomery) and endoscopy (CVC-ClinicDB) come from the main data root. Fundus
(IDRiD), mammography (CMMD, MLO views joined by TOMPEI-CMMD) and pathology (the
Camelyon17-WILDS out-of-distribution test hospital) are read in place from public
Hugging Face mirrors under ``HEALTHROUTER_EXTERNAL_ROOT``; nothing is copied into the repo.
"""
from __future__ import annotations

import glob
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .data import cvc_images, montgomery

LANES = ["chest_xray", "endoscopy", "pathology", "fundus", "mammography"]
CAMELYON_PER_CLASS = 500


def external_root() -> Path:
    return Path(os.environ["HEALTHROUTER_EXTERNAL_ROOT"]).expanduser()


def _parquet(pattern: str, columns: list[str]) -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    for path in sorted(glob.glob(str(external_root() / pattern))):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def _decode(item) -> Image.Image:
    if isinstance(item, (bytes, bytearray)):
        return Image.open(io.BytesIO(item)).convert("RGB")
    return Image.open(item).convert("RGB")


def fundus_idrid():
    rows = _parquet("amin-nejad__idrid-disease-grading/data/*.parquet", ["image", "label"])
    rows.sort(key=lambda r: r["image"]["path"])
    return ([r["image"]["bytes"] for r in rows], np.array([r["label"] for r in rows]),
            np.arange(len(rows)))


def mammography_cmmd():
    rows = _parquet("MedOtter__TOMPEI-CMMD/data/*.parquet",
                    ["image", "patient_id", "classification", "is_exclusion"])
    rows = [r for r in rows if not r["is_exclusion"]]
    patients = {p: i for i, p in enumerate(sorted({r["patient_id"] for r in rows}))}
    return ([r["image"]["bytes"] for r in rows],
            np.array([int(r["classification"] == "Malignant") for r in rows]),
            np.array([patients[r["patient_id"]] for r in rows]))


def pathology_camelyon17(seed: int = 2026):
    import pyarrow.parquet as pq
    path = sorted(glob.glob(str(external_root() / "jxie__camelyon17/data/ood_test-*.parquet")))[0]
    labels = np.array(pq.read_table(path, columns=["label"]).column("label").to_pylist())
    rng = np.random.RandomState(seed)
    chosen = np.sort(np.concatenate([rng.choice(np.where(labels == c)[0], CAMELYON_PER_CLASS,
                                                replace=False) for c in (0, 1)]))
    images = pq.read_table(path, columns=["image"]).column("image").take(chosen).to_pylist()
    # Patch ids are sequential within a slide region; blocks of 200 approximate grouping.
    return ([r["bytes"] for r in images], labels[chosen], chosen // 200)


def sources(root: str) -> dict:
    """{lane: {items, y, groups}} for every lane's external cohort."""
    chest_files, chest_y = montgomery(root)
    cvc = cvc_images(root)
    fundus = fundus_idrid()
    mammo = mammography_cmmd()
    patho = pathology_camelyon17()
    return {
        "chest_xray": {"items": chest_files, "y": chest_y, "groups": np.arange(len(chest_y)),
                       "cohort": "Montgomery County (USA)"},
        "endoscopy": {"items": cvc, "y": np.zeros(len(cvc), int),
                      "groups": np.array([int(Path(f).stem) // 21 for f in cvc]), "cohort": "CVC-ClinicDB (Spain)"},
        "pathology": {"items": patho[0], "y": patho[1], "groups": patho[2],
                      "cohort": "Camelyon17-WILDS OOD test hospital"},
        "fundus": {"items": fundus[0], "y": fundus[1], "groups": fundus[2],
                   "cohort": "IDRiD (India)"},
        "mammography": {"items": mammo[0], "y": mammo[1], "groups": mammo[2],
                        "cohort": "CMMD (China), MLO views"},
    }


def decode(item) -> Image.Image:
    return _decode(item)

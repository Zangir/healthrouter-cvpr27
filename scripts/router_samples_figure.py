#!/usr/bin/env python3
"""Benchmark sample figure (user request 2026-11-16): one image per public source whose licence permits
reproduction, and one external fundus image under the four synthetic acquisition shifts.

Layout mirrors SemLA (CVPR 2025) Fig. 6, a strip of example images with column labels, and the dataset
overview role of VL-RouterBench (CVPR 2026) Fig. 3 (source name and sample count per group).

Every tile is the exact 224 px array the encoders received, read from the local feature cache
(HEALTHROUTER_CACHE): pool_<task>_224_*.npy for the in-source pools, gate_external5_<condition>_*.npy
for the external cohorts (row order given by dispatch_items.npz), pool_eurosat_224.npz for EuroSAT.
Raw images are not in the release, so this script runs only where the cache exists.

Licences (projects/CVPR27_release/healthrouter/MANIFEST.md): PatchCamelyon CC0 1.0, Camelyon17-WILDS
CC0, CBIS-DDSM CC BY 3.0 (TCIA), CMMD CC BY 4.0 (TCIA), IDRiD CC BY 4.0, EuroSAT MIT. Shenzhen,
Montgomery, APTOS and DTD state no licence that permits reproduction and are not shown.

Tiles are chosen by row index within each source. The indices below were fixed after a visual check that
the image carries no burned-in text (view markers, labels) and no identifier; no file name is read or
printed. Counts come from results/router_main.json, router_external.json and router_general.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import style  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("HEALTHROUTER_CACHE", "~/.cache/healthrouter")).expanduser()
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parents[1] / "paper"
ORANGE, GREEN, VERM, GRAY = "#E69F00", "#009E73", "#D55E00", "#8a8a8a"
LANE = {"pathology": 2, "fundus": 3, "mammography": 4}  # dispatch_items.npz lane ids
CONDS = [("standard", "Clean", "as acquired"), ("grayscale", "Grayscale", "luminance"),
         ("blur", "Blur", "radius 2"), ("low_resolution", "Low-res.", "64 px"),
         ("jpeg", "JPEG", "quality 10")]
# (key, name, task, colour, role, class label, row index); indices fixed after the visual check above
PICK = [("pcam", "PCam", "Pathology", ORANGE, "in-source", "metastasis", 5),
        ("camelyon17", "Camelyon17", "Pathology", ORANGE, "external", "tumour", 503),
        ("cbis", "CBIS-DDSM", "Mammography", VERM, "in-source", "malignant", 372),
        ("cmmd", "CMMD", "Mammography", VERM, "external", "malignant", 40),
        ("idrid", "IDRiD", "Fundus", GREEN, "external", "DR grade 4", 19),
        ("eurosat", "EuroSAT", "Satellite", GRAY, "in-source", "river", None)]
FUNDUS_SHIFT_ROW = 19  # the IDRiD tile of panel (a), shown under every condition in panel (b)


def load_json(name):
    return json.loads((ROOT / "results" / name).read_text())


def first(pattern):
    hits = sorted(glob.glob(str(CACHE / pattern)))
    assert hits, pattern
    return hits[0]


def tiles():
    items = np.load(CACHE / "dispatch_items.npz")
    lane = items["lane_ids"]
    ext = {c: np.load(first(f"gate_external5_{c}_*.npy"), mmap_mode="r") for c, _, _ in CONDS}
    def ext_rows(task):
        return np.where(lane == LANE[task])[0]
    eu = np.load(CACHE / "pool_eurosat_224.npz")
    river = int(np.where(eu["y"] == list(eu["classes"]).index("River"))[0][0])
    y = {"pcam": np.load(CACHE / "features" / "pathology_imagenet_convnext_224.npz")["y"],
         "cbis": np.load(CACHE / "features" / "mammography_imagenet_convnext_224.npz")["y"],
         "camelyon17": items["y_pathology"], "cmmd": items["y_mammography"], "idrid": items["y_fundus"]}
    src = {"pcam": lambda i: np.load(first("pool_pathology_224_*.npy"), mmap_mode="r")[i],
           "cbis": lambda i: np.load(first("pool_mammography_224_*.npy"), mmap_mode="r")[i],
           "camelyon17": lambda i: ext["standard"][ext_rows("pathology")[i]],
           "cmmd": lambda i: ext["standard"][ext_rows("mammography")[i]],
           "idrid": lambda i: ext["standard"][ext_rows("fundus")[i]],
           "eurosat": lambda i: eu["x"][river]}
    expect = {"pcam": 1, "cbis": 1, "camelyon17": 1, "cmmd": 1, "idrid": 4}  # class shown in the label
    out = []
    for key, *_rest, idx in PICK:
        if key in expect:
            assert int(y[key][idx]) == expect[key], (key, idx)
        out.append(np.asarray(src[key](idx)))
    shifts = [np.asarray(ext[c][ext_rows("fundus")[FUNDUS_SHIFT_ROW]]) for c, _, _ in CONDS]
    return out, shifts


def counts():
    main, ext, gen = load_json("router_main.json"), load_json("router_external.json"), load_json("router_general.json")
    n_in = {t: v["n"] for t, v in main["methods"]["healthrouter"]["per_task"].items()}
    n_gen = {t: v["n"] for t, v in gen["methods"]["healthrouter"]["per_task"].items()}
    return {"pcam": n_in["pathology"], "cbis": n_in["mammography"], "camelyon17": ext["n"]["pathology"],
            "cmmd": ext["n"]["mammography"], "idrid": ext["n"]["fundus"], "eurosat": n_gen["eurosat"]}


def check_layout(fig):
    """Fail if two labels overlap or a label leaves the figure (measured on the rendered figure)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in fig.texts if t.get_text()]
    for txt, b in boxes:
        assert b.x0 >= fb.x0 - 0.5 and b.x1 <= fb.x1 + 0.5 and b.y0 >= fb.y0 - 0.5 and b.y1 <= fb.y1 + 0.5, ("clipped", txt)
    for i, (t1, b1) in enumerate(boxes):
        for t2, b2 in boxes[i + 1:]:
            ov = min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 0.5 and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 0.5
            assert not ov, ("overlap", t1, t2)


def main():
    style()
    plt.rcParams["font.family"] = "Arial"
    imgs, shifts = tiles()
    n = counts()
    FS = 7.0
    W, T, G, GG = 6.85, 0.58, 0.03, 0.18            # width, tile, gap, gap between panels (inches)
    H = 1.42
    fig = plt.figure(figsize=(W, H))
    y0 = 0.47                                        # bottom of the tiles
    band_h = 0.13
    def ax_at(x, img):
        a = fig.add_axes([x / W, y0 / H, T / W, T / H])
        a.imshow(img, interpolation="lanczos")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(True); s.set_linewidth(0.4); s.set_color("#555555")
        return a
    def band(x0, x1, text, colour):
        fig.patches.append(Rectangle((x0 / W, (y0 + T + 0.03) / H), (x1 - x0) / W, band_h / H,
                                     transform=fig.transFigure, color=colour, alpha=0.9, lw=0))
        fig.text((x0 + x1) / 2 / W, (y0 + T + 0.03 + band_h / 2) / H, text, ha="center", va="center",
                 fontsize=FS, color="white", fontweight="bold")
    x = 0.0
    xs = []
    for (key, name, task, col, role, lab, _), img in zip(PICK, imgs):
        ax_at(x, img)
        fig.text((x + T / 2) / W, (y0 - 0.05) / H, name, ha="center", va="top", fontsize=FS, fontweight="bold")
        fig.text((x + T / 2) / W, (y0 - 0.155) / H, role, ha="center", va="top", fontsize=FS)
        fig.text((x + T / 2) / W, (y0 - 0.26) / H, f"n = {n[key]:,}", ha="center", va="top", fontsize=FS)
        fig.text((x + T / 2) / W, (y0 - 0.365) / H, lab, ha="center", va="top", fontsize=FS, style="italic", color="#333333")
        xs.append(x)
        x += T + G
    # task bands over panel (a): consecutive tiles of one task share a band
    i = 0
    while i < len(PICK):
        j = i
        while j + 1 < len(PICK) and PICK[j + 1][2] == PICK[i][2]:
            j += 1
        band(xs[i], xs[j] + T, PICK[i][2], PICK[i][3])
        i = j + 1
    xa_end = xs[-1] + T
    x = xa_end + GG
    xb0 = x
    for (c, name, desc), img in zip(CONDS, shifts):
        ax_at(x, img)
        fig.text((x + T / 2) / W, (y0 - 0.05) / H, name, ha="center", va="top", fontsize=FS, fontweight="bold")
        fig.text((x + T / 2) / W, (y0 - 0.155) / H, desc, ha="center", va="top", fontsize=FS)
        x += T + G
    band(xb0, x - G, "Fundus, external (IDRiD), same image", GREEN)
    ty = (y0 + T + 0.03 + band_h + 0.035) / H
    fig.text(0, ty, "(a) One image per public source whose licence permits reproduction", ha="left", va="bottom", fontsize=FS + 0.3, fontweight="bold")
    fig.text(xb0 / W, ty, "(b) The four synthetic acquisition shifts", ha="left", va="bottom", fontsize=FS + 0.3, fontweight="bold")
    fig.add_artist(plt.Line2D([(xa_end + GG / 2) / W] * 2, [0.02, ty + 0.1], transform=fig.transFigure,
                              color="#999999", lw=0.6, ls=(0, (3, 2))))
    check_layout(fig)
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "router_samples.pdf", bbox_inches=None, dpi=300)
    fig.savefig(out / "router_samples.png", bbox_inches=None, dpi=300)
    plt.close(fig)
    (PAPER / "figures").mkdir(exist_ok=True)
    import shutil
    shutil.copy2(out / "router_samples.pdf", PAPER / "figures" / "router_samples.pdf")
    print("wrote", out / "router_samples.pdf")


if __name__ == "__main__":
    main()

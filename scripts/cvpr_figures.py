#!/usr/bin/env python3
"""CVPR 2027 figures, rendered from committed results JSON only (no images, no models)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import style  # noqa: E402

DOMAINS = ["chest_xray", "pathology", "fundus", "mammography"]
DOMAIN_LABEL = {"chest_xray": "Chest\n(AUROC)", "pathology": "Pathology\n(AUROC)",
                "fundus": "Fundus\n(QWK)", "mammography": "Mammo.\n(AUROC)"}
ORDER = ["imagenet_convnext", "dinov2_base", "dinov2_large", "siglip2_base", "biomedclip",
         "rad_dino", "phikon", "phikon_v2", "retfound", "mammo_clip"]
LABEL = {"imagenet_convnext": "ConvNeXt-T", "dinov2_base": "DINOv2-B", "dinov2_large": "DINOv2-L",
         "siglip2_base": "SigLIP2-B", "biomedclip": "BiomedCLIP", "rad_dino": "RAD-DINO",
         "phikon": "Phikon", "phikon_v2": "Phikon-v2", "retfound": "RETFound",
         "mammo_clip": "Mammo-CLIP"}
IN_DOMAIN = {"chest_xray": ["rad_dino"], "pathology": ["phikon", "phikon_v2"],
             "fundus": ["retfound"], "mammography": ["mammo_clip"]}
LANES = ["chest_xray", "endoscopy", "pathology", "fundus", "mammography"]
LANE_LABEL = {"chest_xray": "Chest", "endoscopy": "Endo.", "pathology": "Path.",
              "fundus": "Fundus", "mammography": "Mammo."}
GATES = {"supervised": "Supervised gate", "probe": "Frozen probe",
         "probe_grayaug": "Probe + gray aug.", "zero_shot_biomedclip": "Zero-shot BiomedCLIP"}
GATE_COLOR = {"supervised": "#0072B2", "probe": "#E69F00", "probe_grayaug": "#009E73",
              "zero_shot_biomedclip": "#CC79A7"}


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def panel(ax, text):
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, fontweight="bold", fontsize=11,
            va="bottom", ha="right")


def grid_heatmap(ax, grid: dict, size: str, fontsize=7.0, cbar=True, decimals=3, short=False):
    """Gap to the best package in each domain; star = winner, box = in-domain model."""
    doms = grid["sizes"][size]["domains"]
    gap = np.full((len(DOMAINS), len(ORDER)), np.nan)
    for i, d in enumerate(DOMAINS):
        scores = {p: v["oof_score"] for p, v in doms[d]["packages"].items()}
        best = max(scores.values())
        for j, p in enumerate(ORDER):
            if p in scores:
                gap[i, j] = scores[p] - best
    im = ax.imshow(gap, cmap="Blues_r", vmin=-0.25, vmax=0, aspect="auto")
    ax.grid(False)
    for i, d in enumerate(DOMAINS):
        for j, p in enumerate(ORDER):
            if np.isnan(gap[i, j]):
                continue
            score = doms[d]["packages"][p]["oof_score"]
            star = p == doms[d]["best_package"]
            label = f"{score:.{decimals}f}"
            if short:
                label = label.replace("0.", ".", 1)
            ax.text(j, i, label + ("" if short or not star else "★"), ha="center", va="center",
                    fontsize=fontsize, color="#1b2a38" if gap[i, j] > -0.13 else "white",
                    fontweight="bold" if star else "normal")
            if p in IN_DOMAIN[d]:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                           edgecolor="#D55E00", linewidth=1.6))
    ax.set_xticks(range(len(ORDER)), [LABEL[p] for p in ORDER], rotation=40, ha="right",
                  fontsize=fontsize + 0.6)
    ax.set_yticks(range(len(DOMAINS)), [DOMAIN_LABEL[d].replace("\n", " ") for d in DOMAINS],
                  fontsize=fontsize + 0.6)
    for x in (3.5, 4.5):
        ax.axvline(x, color="white", linewidth=2.5)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        cb.set_label("gap to best", fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
    return im


def retained_bars(ax, dispatch: dict, gate="supervised", conds=("standard", "grayscale", "jpeg")):
    colors = {"standard": "#0072B2", "grayscale": "#D55E00", "jpeg": "#E69F00",
              "blur": "#56B4E9", "low_resolution": "#999999"}
    width = 0.8 / len(conds)
    x = np.arange(len(LANES))
    for k, c in enumerate(conds):
        block = dispatch["gates"][gate][f"external:{c}"]
        vals = [block[l]["retained_skill"] for l in LANES]
        ax.bar(x + (k - (len(conds) - 1) / 2) * width, vals, width, color=colors[c],
               label=f"{c.replace('_', ' ')} (mean {block['macro_retained_skill']:.2f})")
    ax.set_xticks(x, [LANE_LABEL[l] for l in LANES])
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="#1b2a38", linewidth=0.6)
    ax.set_ylabel("retained skill $R_m$")
    ax.legend(fontsize=7.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0,
              columnspacing=0.8, labelspacing=0.25)


def gate_bars(ax, dispatch: dict):
    """Macro retained skill for every gate under every acquisition condition (dot plot)."""
    conds = [("standard", "clean"), ("grayscale", "gray"), ("blur", "blur"),
             ("low_resolution", "low-res"), ("jpeg", "JPEG")]
    markers = {"supervised": "o", "probe": "s", "probe_grayaug": "^", "zero_shot_biomedclip": "D"}
    x = np.arange(len(conds))
    for k, (g, name) in enumerate(GATES.items()):
        vals = [dispatch["gates"][g][f"external:{c}"]["macro_retained_skill"] for c, _ in conds]
        ax.plot(x + (k - 1.5) * 0.06, vals, marker=markers[g], color=GATE_COLOR[g], label=name,
                linewidth=1.0, markersize=4.5)
    ax.set_xticks(x, [n for _, n in conds], fontsize=8.6, rotation=30)
    ax.set_ylim(0.6, 1.03)
    ax.tick_params(labelsize=9.5)
    ax.set_ylabel("macro retained skill $\\bar R$", fontsize=9.5)
    ax.legend(fontsize=8.6, loc="lower left", ncol=1, handlelength=1.2)


def abstention_bars(ax, abstention: dict, gate="supervised"):
    scen = ["clean_to_clean", "clean_to_grayscale", "clean_to_jpeg", "grayscale_to_grayscale"]
    labels = ["C\u2192C", "C\u2192G", "C\u2192J", "G\u2192G"]
    block = abstention["gates"][gate]
    risk = [block[s]["test_selective_risk"] for s in scen]
    cov = [block[s]["test_coverage"] for s in scen]
    ok = [block[s]["risk_met"] for s in scen]
    bars = ax.bar(range(len(scen)), risk, color=["#009E73" if o else "#D55E00" for o in ok], width=0.6)
    ax.axhline(abstention["alpha"], color="#1b2a38", linestyle="--", linewidth=1)
    for b, c in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), abstention["alpha"]) + 0.004,
                f"{c:.2f}", ha="center", va="bottom", fontsize=8.2)
    ax.set_xticks(range(len(scen)), labels, fontsize=9)
    ax.tick_params(labelsize=9)
    ax.set_ylabel("test misroute rate", fontsize=9.5)
    ax.set_ylim(0, max(max(risk) * 1.2, abstention["alpha"] * 2))


def emit(fig, stem, meta, caption):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)
    meta.append({"file": f"figures/{stem}.pdf", "caption_draft": caption})


def main():
    style()
    grid, dispatch, abstention = load("matched_grid"), load("dispatch"), load("abstention")
    meta: list[dict] = []
    fig = plt.figure(figsize=(9.2, 3.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1.0, 0.9])
    ax = fig.add_subplot(gs[0])
    grid_heatmap(ax, grid, "224", fontsize=8.4, cbar=False, decimals=2, short=True)
    ax.set_title("(a) Ten frozen encoders, one pixel budget", fontsize=10, loc="left")
    ax = fig.add_subplot(gs[1])
    gate_bars(ax, dispatch)
    ax.set_title("(b) What routing keeps under shift", fontsize=10, loc="left")
    ax = fig.add_subplot(gs[2])
    abstention_bars(ax, abstention)
    ax.set_title("(c) Calibrated abstention", fontsize=10, loc="left")
    emit(fig, "figure_1_teaser", meta, "teaser")
    fig, ax = plt.subplots(figsize=(3.3, 2.5), constrained_layout=True)
    retained_bars(ax, dispatch, conds=("standard", "grayscale", "low_resolution", "jpeg"))
    ax.tick_params(labelsize=8.0)
    ax.yaxis.label.set_size(8.0)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    emit(fig, "lane_skill_supervised", meta, "per-lane retained skill, supervised gate")
    for size in grid["sizes"]:
        fig, ax = plt.subplots(figsize=(7.0, 2.6), constrained_layout=True)
        grid_heatmap(ax, grid, size, fontsize=8.2)  # >= 7 pt printed at 0.9 textwidth
        emit(fig, f"matched_grid_{size}", meta, f"matched grid at {size}px")
    (FIGURES / "cvpr_figures.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

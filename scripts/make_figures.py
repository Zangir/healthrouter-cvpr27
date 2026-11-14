#!/usr/bin/env python3
"""Render the HealthRouter paper figures from committed results JSON only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_style import METHOD_COLORS, style  # noqa: E402
from release_figures import scorecard_figure  # noqa: E402
from routing_figures import routing_figure  # noqa: E402
from schematic import pipeline_panel  # noqa: E402
from story_figures import forest_figure, teaser_figure  # noqa: E402


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def panel_label(ax, text: str) -> None:
    ax.text(-0.12, 1.04, text, transform=ax.transAxes, fontweight="bold",
            fontsize=12, va="bottom")


def emit(fig, stem: str, pattern: str, caption: str, shows: str,
         metadata: list[dict]) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)
    metadata.append({"file": f"figures/{stem}.pdf", "png": f"figures/{stem}.png",
                     "pattern": pattern, "caption_draft": caption, "shows": shows})
    print(f"wrote figures/{stem}.pdf and figures/{stem}.png")


def chest_folds(probes: dict, router: dict, reported: dict, colors: dict,
                metadata: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.75), constrained_layout=True)
    ax = axes[0]
    generic = np.asarray(probes["imagenet_convnext_probe"]["fold_aurocs"])
    radiology = np.asarray(probes["rad_dino_probe"]["fold_aurocs"])
    delta = probes["paired_oof_auroc_difference"]
    folds = np.arange(1, len(generic) + 1)
    for x, lo, hi in zip(folds, generic, radiology):
        ax.plot([x, x], [lo, hi], color="#CBD2D9", linewidth=1.4, zorder=1)
    ax.plot(folds, generic, "o-", color=colors["generic"], linewidth=1.8,
            markersize=5.5, label="ImageNet ConvNeXt probe", zorder=2)
    ax.plot(folds, radiology, "o-", color=colors["radiology"], linewidth=2.2,
            markersize=6, label="RAD-DINO probe", zorder=3)
    ark = reported["ark_plus_shenzhen"]
    ax.axhline(ark["value"], color=colors["reported"], linestyle=(0, (4, 3)),
               linewidth=1.5, label="Ark+ (reported; different protocol)")
    ax.set_xticks(folds)
    ax.set_xlabel("Paired cross-validation fold")
    ax.set_ylabel("Shenzhen TB AUROC")
    ax.set_ylim(max(0, min(generic.min(), radiology.min()) - 0.025), 1.005)
    ax.legend(loc="lower left", fontsize=8.6)
    ax.text(0.97, 0.055,
            f"Paired OOF gain  +{delta['value']:.4f}\n"
            f"95% CI  [{delta['ci_low']:.4f}, {delta['ci_high']:.4f}]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            color="#243B53",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                  "edgecolor": "#CBD2D9", "alpha": 0.94})
    panel_label(ax, "(a)")

    ax = axes[1]
    scale = router["data_scale_ablation"]
    sizes = np.asarray(sorted(int(k) for k in scale))
    means = np.asarray([scale[str(x)]["mean"] for x in sizes])
    stds = np.asarray([scale[str(x)]["std"] for x in sizes])
    for size in sizes:
        reps = np.asarray(scale[str(size)]["replicates"])
        offsets = np.linspace(-0.045, 0.045, len(reps))
        ax.scatter(size * np.exp(offsets), reps, s=20, color=colors["frozen_router"],
                   alpha=0.24, edgecolors="none", zorder=1)
    ax.fill_between(sizes, means - stds, means + stds,
                    color=colors["frozen_router"], alpha=0.17, linewidth=0)
    ax.plot(sizes, means, "o-", color=colors["frozen_router"], linewidth=2.2,
            markersize=6, label="Frozen ImageNet router")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(x) for x in sizes])
    ax.set_xlabel("Per-modality training cap")
    ax.set_ylabel("External two-modality route accuracy")
    observed = np.concatenate([np.asarray(scale[str(x)]["replicates"]) for x in sizes])
    ax.set_ylim(max(0, observed.min() - 0.04), min(1, observed.max() + 0.06))
    ax.legend(loc="lower right")
    panel_label(ax, "(b)")

    caption = (
        "HealthRouter dynamics. (a) Identical folds and logistic heads compare frozen representation packages: "
        f"the RAD-DINO probe reaches {probes['rad_dino_probe']['fold_mean']:.4f} mean AUROC "
        f"versus {probes['imagenet_convnext_probe']['fold_mean']:.4f} with ImageNet ConvNeXt; "
        f"the paired out-of-fold gain is {delta['value']:.4f} (95\\% item-bootstrap CI "
        f"{delta['ci_low']:.4f}--{delta['ci_high']:.4f}). The dashed Ark+ line is its "
        f"reported {ark['value']:.4f} Shenzhen result \\citep{{d2025fully}} under a different protocol and is context, "
        "not a matched comparison. (b) Five repeated logistic heads on frozen ImageNet features "
        "show external routing as the per-modality training cap varies; points are repeats and the "
        "band is one standard deviation."
    )
    emit(fig, "chest_folds", "DeepSeek-style paired method dynamics",
         caption, "Radiology-aligned frozen features improve every TB fold, while frozen-router "
         "external performance is variable and does not improve monotonically with more examples.", metadata)


def main() -> None:
    style()
    probes, router = load("tb_probes"), load("router")
    specialists, reported = load("specialists"), load("reported_context")
    grid, cascade = load("encoder_grid"), load("cascade")
    colors = {
        "radiology": METHOD_COLORS.assign("RAD-DINO probe"),
        "generic": METHOD_COLORS.assign("ImageNet ConvNeXt probe"),
        "reported": METHOD_COLORS.assign("Ark+ reported"),
        "frozen_router": METHOD_COLORS.assign("Frozen ImageNet router"),
        "trained_router": METHOD_COLORS.assign("Fine-tuned ConvNeXt router"),
        "simple": METHOD_COLORS.assign("RGB-statistics baseline"),
    }
    metadata: list[dict] = []
    for stem, pattern, builder in [
        ("earlier_two_lane_teaser", "Earlier-protocol two-lane teaser",
         lambda: teaser_figure(grid, cascade)),
        ("pipeline_schematic", "Compact color-typed executed pipeline",
         lambda: pipeline_panel(cascade)),
        ("encoder_attribution", "Forest plot of paired encoder contrasts",
         lambda: forest_figure(grid)),
        ("routing_dispatch", "Gate variants, acquisition stress, risk-coverage",
         lambda: routing_figure(router, cascade)),
        ("diagnostic_scorecard", "Domain small multiples with metric-separated axes",
         lambda: scorecard_figure(specialists, router)),
    ]:
        fig, caption, shows = builder()
        emit(fig, stem, pattern, caption, shows, metadata)
    chest_folds(probes, router, reported, colors, metadata)
    (FIGURES / "figures.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {FIGURES / 'figures.json'}")


if __name__ == "__main__":
    main()

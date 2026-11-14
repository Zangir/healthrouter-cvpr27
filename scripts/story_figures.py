"""Story figures: encoder grid, executed dispatch, and selective routing."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

DOMAIN_LABELS = {"chest_xray": "Chest\n(AUROC)", "pathology": "Pathology\n(AUROC)",
                 "fundus": "Fundus\n(QWK)", "mammography": "Mammo.\n(AUROC)"}
PACKAGE_LABELS = {"dinov2_base": "DINOv2-B (87M, generic)",
                  "dinov2_large": "DINOv2-L (304M, generic)",
                  "rad_dino": "RAD-DINO (87M, radiology)"}
PACKAGE_COLORS = {"imagenet_convnext": "#8A8A8A", "dinov2_base": "#E69F00",
                  "dinov2_large": "#009E73", "rad_dino": "#0072B2"}
ORDER = ["chest_xray", "pathology", "fundus", "mammography"]
GRID_TEXT = "#243B53"


def _label(ax, text: str, dx: float = -0.11) -> None:
    ax.text(dx, 1.045, text, transform=ax.transAxes, fontweight="bold", fontsize=11,
            va="bottom")


def grid_delta_panel(ax, grid: dict) -> None:
    """Change in each domain's own metric relative to the ImageNet reference."""
    packages = ["dinov2_base", "dinov2_large", "rad_dino"]
    width = 0.26
    xs = np.arange(len(ORDER))
    for i, package in enumerate(packages):
        offset = (i - 1) * width
        values = [grid["domains"][d]["packages"][package]["oof_score"]
                  - grid["domains"][d]["packages"]["imagenet_convnext"]["oof_score"]
                  for d in ORDER]
        ax.bar(xs + offset, values, width * 0.92, color=PACKAGE_COLORS[package],
               label=PACKAGE_LABELS[package], zorder=3)
        for x, value in zip(xs + offset, values):
            va, pad = ("bottom", 3) if value >= 0 else ("top", -3)
            ax.annotate(f"{value:+.3f}".replace("-", "\u2212"), (x, value),
                        xytext=(0, pad), rotation=90, textcoords="offset points",
                        ha="center", va=va, fontsize=6.6, color=GRID_TEXT)
    ax.axhline(0, color="#52616B", linewidth=1.0, zorder=2)
    ax.set_xticks(xs, [DOMAIN_LABELS[d] for d in ORDER], fontsize=9)
    ax.set_ylabel("Change vs ImageNet\nreference (metric points)", fontsize=10)
    every = [grid["domains"][d]["packages"][p]["oof_score"]
             - grid["domains"][d]["packages"]["imagenet_convnext"]["oof_score"]
             for d in ORDER for p in packages]
    span = max(every) - min(every)
    ax.set_ylim(min(every) - 0.20 * span, max(every) + 0.36 * span)
    ax.legend(loc="upper right", fontsize=7.6, ncol=1)
    ax.grid(axis="x", visible=False)
    _label(ax, "(a)")


def dispatch_cost_panel(ax, cascade: dict) -> None:
    """Executed end-to-end segmentation quality under reference and shifted routing."""
    oracle = cascade["oracle_routing"]["endoscopy_mean_dice"]
    clean = cascade["conditions"]["standard"]["selective_dispatch"][0]
    shifted = cascade["conditions"]["grayscale"]["selective_dispatch"][0]
    blocks = [("Reference\nrouting", oracle, "#8A8A8A"),
              ("Executed\ndispatch", clean["endoscopy_served_mean_dice"], "#0072B2"),
              ("Executed under\ngrayscale shift", shifted["endoscopy_served_mean_dice"],
               "#D55E00")]
    xs = np.arange(len(blocks))
    for x, (_, block, color) in zip(xs, blocks):
        value = block["value"]
        ax.bar(x, value, 0.56, color=color, zorder=3)
        ax.errorbar(x, value, yerr=[[value - block["ci_low"]], [block["ci_high"] - value]],
                    fmt="none", ecolor=GRID_TEXT, capsize=3, linewidth=1.1, zorder=4)
        ax.annotate(f"{value:.3f}", (x, block["ci_high"]), xytext=(0, 4), ha="center",
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color=GRID_TEXT)
    drop = oracle["value"] - shifted["endoscopy_served_mean_dice"]["value"]
    ax.annotate("", xy=(2.46, shifted["endoscopy_served_mean_dice"]["value"]),
                xytext=(2.46, oracle["value"]),
                arrowprops={"arrowstyle": "<->", "color": "#D55E00", "linewidth": 1.3})
    ax.text(2.56, (oracle["value"] + shifted["endoscopy_served_mean_dice"]["value"]) / 2,
            f"\u2212{drop:.3f}\nDice", ha="left", va="center", fontsize=8.6,
            fontweight="bold", color="#D55E00")
    ax.set_xticks(xs, [name for name, _, _ in blocks], fontsize=8.6)
    ax.set_xlim(-0.55, 3.35)
    ax.set_ylim(0, 0.99)
    ax.set_ylabel("End-to-end mean Dice,\n612 CVC frames", fontsize=10)
    ax.grid(axis="x", visible=False)
    _label(ax, "(b)")


def selective_panel(ax, cascade: dict) -> None:
    """Route accuracy among covered items as the abstention threshold rises."""
    for variant, color, marker in [("standard", "#0072B2", "o"),
                                   ("grayscale", "#D55E00", "s")]:
        sweep = cascade["conditions"][variant]["selective_dispatch"]
        coverage = [s["coverage"] for s in sweep]
        accuracy = [s["selective_route_accuracy"] for s in sweep]
        ax.plot(coverage, accuracy, marker=marker, color=color, linewidth=1.9,
                markersize=6, label=f"{variant.capitalize()} inputs", zorder=3)
    sweep = cascade["conditions"]["grayscale"]["selective_dispatch"]
    for entry, dx, dy in [(sweep[0], -8, 9), (sweep[3], 9, -13)]:
        ax.annotate(f"\u03c4={entry['threshold']:g}\n{entry['selective_route_accuracy']:.3f}",
                    (entry["coverage"], entry["selective_route_accuracy"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    ha="left" if dx > 0 else "right", color="#D55E00")
    ax.set_xlabel("Coverage (fraction dispatched)", fontsize=10)
    ax.set_ylabel("Route accuracy among\ndispatched items", fontsize=10)
    ax.set_xlim(0.02, 1.05)
    ax.set_ylim(0.865, 1.012)
    ax.legend(loc="lower left", fontsize=8.4)
    _label(ax, "(c)")


def teaser_figure(grid: dict, cascade: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.05), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.30, 0.90, 1.0]})
    grid_delta_panel(axes[0], grid)
    dispatch_cost_panel(axes[1], cascade)
    selective_panel(axes[2], cascade)
    chest = grid["domains"]["chest_xray"]
    winners = grid["best_package_by_domain"]
    oracle = cascade["oracle_routing"]["endoscopy_mean_dice"]["value"]
    shifted = cascade["conditions"]["grayscale"]["selective_dispatch"][0]
    gray = cascade["conditions"]["grayscale"]
    caption = (
        "The case for routing, its cost, and its mitigation. (a) Under one shared five-fold "
        "logistic protocol the best frozen encoder changes by modality: RAD-DINO gains "
        f"{chest['packages']['rad_dino']['oof_score'] - chest['packages']['imagenet_convnext']['oof_score']:+.3f} "
        "AUROC on chest radiographs yet loses on all three non-radiology domains, and a 304M-parameter "
        "generic encoder does not close the gap. (b) Executing the dispatch: reference routing gives "
        f"{oracle:.3f} mean Dice on 612 unseen-institution frames, the gate's own decisions cost "
        f"{oracle - cascade['conditions']['standard']['selective_dispatch'][0]['endoscopy_served_mean_dice']['value']:.3f}, "
        f"and under grayscale shift {gray['misrouted_items']} frames reach the mammography lane and "
        f"end-to-end Dice falls {oracle - shifted['endoscopy_served_mean_dice']['value']:.3f} with the "
        "segmentation checkpoint unchanged. (c) Abstaining below $\\tau$ raises route accuracy among "
        f"dispatched items from {gray['selective_dispatch'][0]['selective_route_accuracy']:.3f} to "
        f"{gray['selective_dispatch'][3]['selective_route_accuracy']:.3f} under shift."
    )
    shows = ("No frozen encoder wins every modality, a routing error propagates into the task "
             "metric, and confidence-based abstention recovers most of that loss.")
    return fig, caption, shows


def forest_figure(grid: dict):
    """Paired contrasts that separate pretraining data from architecture and scale."""
    contrasts = [("rad_dino_minus_imagenet_convnext",
                  "RAD-DINO \u2212 ImageNet ConvNeXt\n(data + arch. + objective)", "#0072B2"),
                 ("rad_dino_minus_dinov2_base",
                  "RAD-DINO \u2212 DINOv2-B\n(pretraining data only)", "#CC79A7"),
                 ("dinov2_large_minus_dinov2_base",
                  "DINOv2-L \u2212 DINOv2-B\n(encoder scale only)", "#009E73")]
    fig, ax = plt.subplots(figsize=(6.7, 4.1), constrained_layout=True)
    rows, labels = [], []
    for domain in ORDER:
        for key, _, color in contrasts:
            rows.append((grid["domains"][domain]["paired_contrasts"][key], color))
            labels.append("")
        labels[-2] = DOMAIN_LABELS[domain].replace("\n", " ")
        rows.append((None, None))
        labels.append("")
    ys = np.arange(len(rows))[::-1]
    for y, (block, color) in zip(ys, rows):
        if block is None:
            continue
        value = block["value"]
        ax.errorbar(value, y, xerr=[[value - block["ci_low"]], [block["ci_high"] - value]],
                    fmt="o", color=color, markersize=5.6, capsize=2.6, linewidth=1.4)
    ax.axvline(0, color="#52616B", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_xlabel("Paired difference in the domain's own metric (95% item bootstrap)",
                  fontsize=9.5)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([0], [0], marker="o", color=color, linestyle="none",
                          markersize=6, label=label) for _, label, color in contrasts]
    ax.legend(handles=handles, loc="upper left", fontsize=7.2, labelspacing=0.85,
              bbox_to_anchor=(-0.01, 1.015), borderpad=0.4)
    chest = grid["domains"]["chest_xray"]["paired_contrasts"]
    caption = (
        "Attributing the chest gain. Each row is a paired item-bootstrap difference on one domain's "
        "fixed pool, holding folds and head constant. Against an architecture- and objective-matched "
        "generic encoder, RAD-DINO still gains "
        f"{chest['rad_dino_minus_dinov2_base']['value']:.4f} AUROC on chest "
        f"(95\\% CI {chest['rad_dino_minus_dinov2_base']['ci_low']:.4f}--"
        f"{chest['rad_dino_minus_dinov2_base']['ci_high']:.4f}), so the effect is attributable to the "
        "pretraining corpus rather than backbone, objective, or capacity; tripling generic encoder size "
        f"moves chest AUROC by {chest['dinov2_large_minus_dinov2_base']['value']:+.4f}. The same contrast "
        "is negative or interval-crossing on every non-radiology domain."
    )
    shows = ("The chest advantage comes from radiology pretraining data, not architecture, "
             "objective, or scale, and it does not transfer to other modalities.")
    return fig, caption, shows

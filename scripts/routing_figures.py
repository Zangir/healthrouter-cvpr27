"""Gate variants, acquisition stress, and the risk-coverage trade-off."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from story_figures import GRID_TEXT, _label


def routing_figure(router: dict, cascade: dict):
    """Gate variants, acquisition stress, and the risk-coverage trade-off."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 2.95), constrained_layout=True)
    ax = axes[0]
    variants = [("HealthRouter gate", "in_source", "unseen_institution_two_modalities",
                 "#0072B2", "o"),
                ("Frozen ImageNet probe", "frozen_imagenet_router_in_source",
                 "frozen_imagenet_router_external", "#E69F00", "^"),
                ("RGB-moment control", "simple_feature_baseline_in_source",
                 "simple_feature_baseline_external", "#CC79A7", "s")]
    xs = np.arange(2)
    for i, (name, inside, outside, color, marker) in enumerate(variants):
        blocks = [router[inside], router[outside]]
        values = np.asarray([b["value"] for b in blocks])
        lower = values - np.asarray([b["ci_low"] for b in blocks])
        upper = np.asarray([b["ci_high"] for b in blocks]) - values
        ax.errorbar(xs + (i - 1) * 0.11, values, yerr=np.vstack([lower, upper]),
                    fmt=marker, color=color, markersize=7, capsize=3, linewidth=1.5,
                    label=name, zorder=3)
        ax.plot(xs + (i - 1) * 0.11, values, color=color, linewidth=1.1, alpha=0.45,
                zorder=2)
    ax.set_xticks(xs, ["In-source\n(5-way, diagnostic)", "Held-out sources\n(2-way)"],
                  fontsize=8.8)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Route accuracy", fontsize=10)
    ax.set_ylim(0.53, 1.03)
    ax.legend(loc="lower left", fontsize=7.8)
    ax.grid(axis="x", visible=False)
    _label(axes[0], "(a)")

    ax = axes[1]
    conditions = [("Standard", router["unseen_institution_two_modalities"])] + [
        (name.replace("_", " ").capitalize(), router["corruption_ablations"][name])
        for name in ("blur", "low_resolution", "grayscale")]
    values = np.asarray([b["value"] for _, b in conditions])
    lower = values - np.asarray([b["ci_low"] for _, b in conditions])
    upper = np.asarray([b["ci_high"] for _, b in conditions]) - values
    x = np.arange(len(conditions))
    ax.plot(x, values, color="#0072B2", linewidth=1.7, alpha=0.85, zorder=2)
    ax.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="o", markersize=7,
                capsize=3, color="#0072B2", linewidth=1.4, zorder=3)
    for xi, (value, hi) in enumerate(zip(values, values + upper)):
        ax.annotate(f"{value:.3f}", (xi, hi), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8.5, color=GRID_TEXT,
                    fontweight="bold" if xi == 3 else "normal")
    ax.set_xticks(x, [name for name, _ in conditions], rotation=14, ha="right",
                  fontsize=8.8)
    ax.set_ylabel("Held-out route accuracy", fontsize=10)
    ax.set_ylim(0.845, 1.017)
    ax.grid(axis="x", visible=False)
    _label(axes[1], "(b)")

    ax = axes[2]
    for variant, color, marker in [("standard", "#0072B2", "o"),
                                   ("grayscale", "#D55E00", "s")]:
        sweep = [s for s in cascade["conditions"][variant]["selective_dispatch"]
                 if s["endoscopy_covered_mean_dice"]]
        ax.plot([s["coverage"] for s in sweep],
                [s["endoscopy_covered_mean_dice"]["value"] for s in sweep],
                marker=marker, color=color, linewidth=1.9, markersize=6,
                label=f"{variant.capitalize()} inputs")
    ax.axhline(cascade["oracle_routing"]["endoscopy_mean_dice"]["value"], color="#8A8A8A",
               linestyle=(0, (4, 3)), linewidth=1.3, label="Reference routing")
    ax.set_xlabel("Coverage (fraction dispatched)", fontsize=10)
    ax.set_ylabel("Mean Dice among\ndispatched frames", fontsize=10)
    ax.set_xlim(0.02, 1.05)
    ax.legend(loc="lower left", fontsize=8.0)
    _label(axes[2], "(c)")

    gray = router["corruption_ablations"]["grayscale"]
    external = router["unseen_institution_two_modalities"]
    simple = router["simple_feature_baseline_external"]
    caption = (
        "Routing behaviour and the abstention trade-off. (a) The five-way in-source split is a "
        "diagnostic, not evidence: an RGB-moment control also solves it "
        f"({router['simple_feature_baseline_in_source']['value']:.3f}) yet collapses to "
        f"{simple['value']:.3f} on held-out sources, where the learned gate keeps "
        f"{external['value']:.4f} ({external['correct']}/{external['n']}). (b) Blur and low-resolution "
        f"resampling barely move held-out routing; grayscale lowers it to {gray['value']:.4f} "
        f"(95\\% CI {gray['ci_low']:.4f}--{gray['ci_high']:.4f}). (c) Restricting dispatch to confident "
        "items trades coverage for end-to-end quality; above the reference line the gain is a "
        "subset-selection effect, not repair."
    )
    shows = ("A pooled-source split rewards dataset recognition, grayscale conversion breaks the "
             "gate, and abstention buys back end-to-end quality.")
    return fig, caption, shows

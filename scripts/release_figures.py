"""Complex schematic and scorecard layouts used by ``make_figures.py``."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


def pipeline_figure(release: dict, router: dict):
    fig, ax = plt.subplots(figsize=(12.4, 4.7))
    ax.set(xlim=(0, 12.6), ylim=(0, 5.1))
    ax.axis("off")
    positions = {
        "input": (0.15, 2.05, 1.55, 0.9), "index": (2.05, 2.05, 1.75, 0.9),
        "router": (4.15, 3.20, 1.7, 0.9), "route_trace": (6.05, 3.20, 2.0, 0.9),
        "classification": (4.15, 1.55, 2.05, 0.95),
        "segmentation": (4.15, 0.20, 2.05, 0.95),
        "prediction": (6.65, 1.00, 1.85, 0.9), "audit": (9.15, 2.05, 2.05, 0.9),
    }
    fills = {"data": "#D9EEF8", "model": "#EADCF0", "algorithm": "#DCE8F5",
             "processing": "#E8ECEF"}
    edges = {"data": "#56B4E9", "model": "#CC79A7", "algorithm": "#0072B2",
             "processing": "#7B8794"}
    nodes = {node["id"]: node for node in release["pipeline"]["nodes"]}

    def center(node_id: str) -> tuple[float, float]:
        x, y, width, height = positions[node_id]
        return x + width / 2, y + height / 2

    for source, target in release["pipeline"]["edges"]:
        ax.add_patch(FancyArrowPatch(center(source), center(target), arrowstyle="-|>",
                     mutation_scale=13, linewidth=1.2, color="#52616B", zorder=1))
    for node_id, node in nodes.items():
        x, y, width, height = positions[node_id]
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle="round,pad=0.03,rounding_size=0.12",
                     facecolor=fills[node["type"]], edgecolor=edges[node["type"]],
                     linewidth=1.4, zorder=2))
        ax.text(x + width / 2, y + height / 2, node["label"], ha="center", va="center",
                fontsize=8.3, color="#243B53", fontweight="semibold", zorder=3)
    classes = "  ·  ".join(name.replace("_", " ") for name in router["classes"])
    ax.text(5.0, 4.33, classes, ha="center", fontsize=8.1, color="#52616B")
    handles = [Patch(facecolor=fills[key], edgecolor=edges[key],
                     label=release["pipeline"]["type_labels"][key])
               for key in ("data", "model", "algorithm", "processing")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, frameon=False, columnspacing=1.6)
    caption = (
        "HealthRouter evaluation pipeline. Restricted images remain in place and observed local split "
        "counts are recorded. Exact duplicates are excluded in the in-source router, APTOS, CBIS-DDSM, and "
        "PatchCamelyon audits. The gate emits a modality label and trace. Classification and "
        "segmentation checkpoints are evaluated on separate paths with task-appropriate metrics and "
        "item-bootstrap intervals. The shared result schema does not imply live dispatch from a gate "
        "prediction to a specialist. Node color denotes artifact type."
    )
    shows = ("Auditable gate and specialist evaluations use a common result schema while "
             "remaining experimentally separate.")
    return fig, caption, shows


def _score_panel(ax, label: str, metric: str, rows: list[tuple[str, dict, str]],
                 colors: dict) -> None:
    ys = np.arange(len(rows))[::-1]
    lows, highs = [], []
    for y, (name, block, kind) in zip(ys, rows):
        value, low, high = block["value"], block["ci_low"], block["ci_high"]
        color = colors[kind]
        marker = "D" if kind == "external" else ("s" if kind == "raw" else "o")
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt=marker,
                    color=color, markersize=4.5, capsize=2, linewidth=1.1)
        ax.annotate(f"{value:.3f}", (value, y), xytext=(0, 4.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.6, color="#243B53")
        lows.append(low)
        highs.append(high)
    margin = max(0.008, (max(highs) - min(lows)) * 0.22)
    ax.set_xlim(max(0, min(lows) - margin), min(1.001, max(highs) + 3.2 * margin))
    ax.set_yticks(ys, [name for name, _, _ in rows])
    ax.set_ylim(-0.45, ys.max() + 0.8)  # room for the value label above the top row
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=2))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax.set_xlabel(metric)
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8.2)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)


def scorecard_figure(specialists: dict, router: dict):
    # Sized for the printed text width so that no label prints below 7 pt (review 2026-11-14).
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8})
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.4))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.13, top=0.91, wspace=0.95, hspace=0.85)
    colors = {"audited": "#0072B2", "raw": "#8A8A8A", "external": "#D55E00"}
    _score_panel(axes[0, 0], "(a) Chest, external", "AUROC",
                 [("Montgomery", specialists["tb_montgomery_auroc"], "external")], colors)
    pcam = specialists["pcam_official_test_auroc"]
    _score_panel(axes[0, 1], "(b) Pathology, official test", "AUROC",
                 [("Raw", pcam["raw"], "raw"), ("Exact-dedup", pcam["clean"], "audited")], colors)
    aptos = specialists["aptos_random_holdout"]
    _score_panel(axes[0, 2], "(c) Fundus, random holdout", "Quadratic weighted kappa",
                 [("Raw", aptos["raw_qwk"], "raw"),
                  ("Exact-dedup", aptos["clean_qwk"], "audited")], colors)
    cbis = specialists["cbis_local_mirror_test_auroc"]
    _score_panel(axes[1, 0], "(d) Mammo., local mirror", "AUROC",
                 [("Raw", cbis["raw"], "raw"), ("Exact-dedup", cbis["clean"], "audited")], colors)
    seg = specialists["segmentation"]
    _score_panel(axes[1, 1], "(e) Endoscopy, masks", "Mean Dice",
                 [("Kvasir val.", seg["kvasir_validation_mean_dice"], "audited"),
                  ("CVC external", seg["cvc_external_mean_dice"], "external")], colors)
    ax = axes[1, 2]
    ax.axis("off")
    legend = [Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["raw"],
                     markeredgecolor=colors["raw"], label="Raw protocol"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["audited"],
                     markeredgecolor=colors["audited"], label="Exact-deduplicated / internal"),
              Line2D([0], [0], marker="D", color="none", markerfacecolor=colors["external"],
                     markeredgecolor=colors["external"], label="External dataset")]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=7.8)
    ax.text(0, 0.12, "Each panel has its own metric axis;\n"
            "panels are not ranked. Intervals resample\n"
            "fixed items, not training seeds.", transform=ax.transAxes,
            fontsize=7.8, color="#52616B", linespacing=1.35, va="top")
    caption = (
        "Five-domain diagnostic scorecard. Every panel uses its own task metric and fixed-set "
        "uncertainty axis; values should be read within panels rather than ranked across domains. "
        "Exact cross-split duplicate exclusion leaves the APTOS, CBIS-DDSM, and PatchCamelyon "
        "conclusions nearly unchanged. Diamonds denote evaluation on a distinct dataset: the "
        f"Shenzhen-trained chest specialist reaches {specialists['tb_montgomery_auroc']['value']:.4f} "
        f"AUROC on Montgomery, and the Kvasir-trained segmenter reaches "
        f"{seg['cvc_external_mean_dice']['value']:.4f} mean Dice on CVC-ClinicDB. Evaluation sizes "
        f"are {router['unseen_source_counts']['chest_xray']} Montgomery images, {pcam['n_clean']:,} "
        f"deduplicated PCam patches, {aptos['n_clean']} deduplicated APTOS images, "
        f"{cbis['n_clean']} deduplicated CBIS-DDSM images, {seg['kvasir_n']} Kvasir validation "
        f"images, and {seg['cvc_n']} CVC images. These are public benchmark stress tests, not "
        "prospective clinical validation."
    )
    shows = ("All five image domains have checkpoint re-evaluations, duplicate removal is not "
             "driving the classification results, and external transfer remains imperfect.")
    return fig, caption, shows

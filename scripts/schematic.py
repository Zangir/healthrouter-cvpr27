"""Compact single-column schematic of the executed dispatch path."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from story_figures import GRID_TEXT, _label


def _box(ax, x, y, w, h, text, fill, edge, fontsize=8.0):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.09",
                 facecolor=fill, edgecolor=edge, linewidth=1.15, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=GRID_TEXT, fontweight="semibold", zorder=3)
    return (x + w / 2, y + h / 2), (x + w, y + h / 2), (x, y + h / 2)


def _arrow(ax, start, end, color="#52616B", style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                 linewidth=1.05, color=color, zorder=1,
                 shrinkA=1.5, shrinkB=2.5))


def pipeline_panel(cascade: dict):
    """Compact single-column schematic of the executed dispatch path."""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(3.32, 2.95), constrained_layout=True)
    ax.set(xlim=(0, 10), ylim=(0, 9.4))
    ax.axis("off")
    ax.grid(False)
    data, model, algo, proc = "#D9EEF8", "#EADCF0", "#DCE8F5", "#E8ECEF"
    edata, emodel, ealgo, eproc = "#56B4E9", "#CC79A7", "#0072B2", "#7B8794"
    _, img_r, _ = _box(ax, 0.2, 7.5, 3.4, 1.15, "Decoded pixels\n(no DICOM tags)", data, edata)
    gate_c, gate_r, gate_l = _box(ax, 4.5, 7.5, 5.3, 1.15,
                                  "Five-way modality gate\n+ selected-route score", model, emodel)
    _arrow(ax, img_r, gate_l)
    tau_c, tau_r, tau_l = _box(ax, 2.4, 5.7, 5.2, 1.0, "$s(x)$ \u2265 $\\tau$? (calibrated)", algo, ealgo)
    _arrow(ax, (gate_c[0], 7.5), (tau_c[0] + 1.4, 6.7))
    ab_c, _, ab_l = _box(ax, 8.1, 5.75, 1.8, 0.9, "abstain", proc, eproc, 7.6)
    _arrow(ax, tau_r, ab_l)
    reg_c, _, _ = _box(ax, 0.9, 3.95, 8.2, 1.15,
                       "Registry lookup: preprocess,\ncheckpoint, output type, metric",
                       algo, ealgo)
    _arrow(ax, (tau_c[0], 5.7), (reg_c[0], 5.1))
    lanes = [("3 classifiers\nscore", 0.2), ("DR grader\nordinal grade", 3.55),
             ("Segmenter\nmask", 6.9)]
    for text, x in lanes:
        _, _, lane_l = _box(ax, x, 2.15, 2.9, 1.15, text, model, emodel, 7.6)
        _arrow(ax, (reg_c[0], 3.95), (x + 1.45, 3.3))
    chk_c, _, _ = _box(ax, 0.9, 0.35, 8.2, 1.15,
                       "Score with the lane metric;\nwrong route = null output\n(0.5, grade 0, no mask)", proc, eproc, 7.6)
    for _, x in lanes:
        _arrow(ax, (x + 1.45, 2.15), (chk_c[0], 1.5))
    handles = [Patch(facecolor=data, edgecolor=edata, label="data"),
               Patch(facecolor=model, edgecolor=emodel, label="checkpoint"),
               Patch(facecolor=algo, edgecolor=ealgo, label="decision"),
               Patch(facecolor=proc, edgecolor=eproc, label="scoring")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4,
              frameon=False, fontsize=7.6, columnspacing=0.9, handlelength=1.1,
              handletextpad=0.4)
    standard = cascade["conditions"]["standard"]["selective_dispatch"][0]
    caption = (
        "The executed path. The gate reads decoded pixels, abstains below $\\tau$, and otherwise "
        "resolves the predicted route to one independently trained checkpoint with its own "
        "preprocessing, output type $\\mathcal{Y}$, and metric $\\mu$. Because lanes emit "
        "incompatible types, a wrong route yields no valid output rather than a degraded one: on "
        f"held-out sources {standard['valid_output_rate']:.4f} of items are served, and the "
        "shortfall is charged to the task metric."
    )
    shows = ("Routing, dispatch, and scoring are one executed path, so a route error reaches the "
             "task metric instead of stopping at the gate.")
    return fig, caption, shows

#!/usr/bin/env python3
"""Method figure for HealthRouter (pure drawing; the only numbers are structural constants).

Layout: three dashed panels, (1) build the collection library offline, (2) retrieve per image,
(3) run the collection or the harness (VL-RouterBench Fig. 2 layout), with the library drawn
as a tinted box of collection chips (SemLA Figs. 1-2 icon language).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import style  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE, ORANGE, GRAY, GREEN, VERM, SKY = "#0072B2", "#E69F00", "#8a8a8a", "#009E73", "#D55E00", "#56B4E9"
ENC_COL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999", "#000000", "#7f3b08"]


def box(ax, x, y, w, h, text, fc="white", ec=GRAY, fs=7.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.0, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color=GRAY, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color, ls=ls))


def chip(ax, x, y, members, s=0.13):
    for k, m in enumerate(members):
        ax.add_patch(Rectangle((x + k * s, y), s * 0.9, s * 0.9, fc=ENC_COL[m], ec="white", lw=0.4))


def panel(ax, x, w, title):
    ax.add_patch(FancyBboxPatch((x, 0.08), w, 3.25, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec="#555555", lw=1.0, ls=(0, (4, 3))))
    ax.text(x + w / 2, 3.12, title, ha="center", va="center", fontsize=7, fontweight="bold")


def main():
    style()
    fig, ax = plt.subplots(figsize=(7.0, 2.25))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    # (1) offline library
    panel(ax, 0.1, 4.6, "(1) Build the library (offline)")
    for k in range(10):
        ax.add_patch(Rectangle((0.35 + k * 0.17, 2.45), 0.15, 0.3, fc=ENC_COL[k], ec="white", lw=0.4))
    ax.text(1.2, 2.25, "10 encoders + heads", ha="center", fontsize=5.6)
    box(ax, 2.35, 2.35, 1.0, 0.5, "5-fold\ncross-fit", fs=5.8)
    box(ax, 3.55, 2.35, 1.05, 0.5, "out-of-fold\n$p_c^{(-i)}(y_i|x_i)$", fs=5.8)
    arrow(ax, (2.1, 2.6), (2.35, 2.6))
    arrow(ax, (3.35, 2.6), (3.55, 2.6))
    ax.add_patch(FancyBboxPatch((0.35, 0.3), 4.15, 1.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#e8f6fb", ec=SKY, lw=1.0))
    ax.text(0.5, 1.72, "library $\\mathcal{L}$: all collections, $|S|\\leq 3$", fontsize=5.8, color=BLUE)
    combos = [(5,), (6, 7), (1, 8), (0, 3, 9), (4, 6), (2,), (1, 2, 8), (6,), (3, 4), (0, 5, 7), (9,), (1,)]
    for k, c in enumerate(combos):
        chip(ax, 0.55 + (k % 6) * 0.65, 1.2 - (k // 6) * 0.38, c)
    ax.text(2.42, 0.45, "utility $u_S(x_i)$ per training image",
            ha="center", fontsize=5.6)
    arrow(ax, (4.05, 2.35), (3.6, 1.92), color=BLUE)
    # (2) retrieval
    panel(ax, 4.9, 4.55, "(2) Retrieve per image, no label")
    box(ax, 5.1, 2.3, 0.85, 0.6, "image $x$", fs=6)
    box(ax, 6.2, 2.3, 1.2, 0.6, "ConvNeXt-T\n$z(x)$", fs=5.8)
    arrow(ax, (5.95, 2.6), (6.2, 2.6))
    ax.add_patch(FancyBboxPatch((5.1, 0.3), 2.55, 1.75, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#f7f7f7", ec="#bbbbbb", lw=0.8))
    import numpy as np
    rng = np.random.RandomState(3)
    cents = [(5.6, 1.55), (6.9, 1.6), (5.75, 0.75), (7.1, 0.8)]
    for (cx, cy), col in zip(cents, [BLUE, ORANGE, GREEN, VERM]):
        pts = rng.normal(0, 0.14, size=(14, 2))
        ax.scatter(cx + pts[:, 0], cy + pts[:, 1] * 0.8, s=3, color=col, alpha=0.6, lw=0)
    ax.add_patch(Circle((6.9, 1.6), 0.28, fc="none", ec="black", lw=0.8, ls="--"))
    ax.scatter([6.9], [1.6], marker="*", s=40, color="black", zorder=5)
    ax.text(6.38, 0.36, "router space: $K$ neighbours", ha="center", fontsize=5.3)
    arrow(ax, (6.8, 2.3), (6.9, 1.92))
    box(ax, 7.8, 1.35, 1.5, 1.0, "average\nneighbour\nutilities $\\to S^\\star$", fs=5.6, ec=BLUE)
    box(ax, 7.8, 0.35, 1.5, 0.75, "task vote of\nneighbours\n(picks heads)", fs=5.6, ec=GRAY)
    arrow(ax, (7.2, 1.6), (7.8, 1.85), color=BLUE)
    arrow(ax, (7.2, 1.45), (7.8, 0.75), color=GRAY)
    arrow(ax, (4.5, 1.1), (5.1, 1.15), color=BLUE, ls="--")
    # (3) run
    panel(ax, 9.65, 4.25, "(3) Run collection or harness")
    box(ax, 9.85, 2.15, 3.85, 0.65, "HealthRouter: run and average $S^\\star(x)$", fs=5.6, fc="#eaf2fa", ec=BLUE, bold=True)
    box(ax, 9.85, 0.3, 3.85, 1.6, "", fc="#eaf7f2", ec=GREEN)
    ax.text(11.78, 1.7, "Harness: stop once confident ($\\geq\\theta$)", ha="center", fontsize=5.6,
            fontweight="bold")
    for row, (lab, conf) in enumerate([("easy image", [0.93]), ("hard image", [0.58, 0.66, 0.81])]):
        y0 = 1.1 - row * 0.62
        ax.text(10.0, y0 + 0.12, lab, fontsize=5.8, va="center")
        for k, c in enumerate(conf):
            ax.add_patch(Rectangle((11.5 + k * 0.62, y0), 0.5, 0.26 * c / 0.93 + 0.02, fc=GREEN if c >= 0.8 else "#9ecae1",
                                   ec="white"))
        ax.plot([11.4, 13.4], [y0 + 0.26 * 0.8 / 0.93 + 0.02] * 2, ls=":", color="black", lw=0.7)
    ax.text(13.45, 0.8, "$\\theta$", fontsize=6.5)
    arrow(ax, (9.3, 1.9), (9.85, 2.4), color=BLUE)
    arrow(ax, (9.3, 1.7), (9.85, 1.2), color=GREEN)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "router_method.pdf", bbox_inches="tight")
    fig.savefig(OUT / "router_method.png", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "router_method.pdf")


if __name__ == "__main__":
    main()

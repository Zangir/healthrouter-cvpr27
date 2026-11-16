#!/usr/bin/env python3
"""Method figure for HealthRouter (pure drawing; the only numbers are structural constants).

Layout: three dashed panels, (1) build the collection library offline, (2) retrieve per image,
(3) run the collection or the harness (VL-RouterBench Fig. 2 layout), with the library drawn
as a tinted box of collection chips (SemLA Figs. 1-2 icon language).

2026-11-15 (surgical revision): every arrow is axis-aligned and runs from one box edge to the
next with a fixed clear gap (GAP) at both ends; arrows between stages run inside the white gap
between two dashed panels, so no arrow crosses or touches text, a box border or a panel border.
`check_layout()` measures this on the rendered figure and fails the build on any violation.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import style  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE, ORANGE, GRAY, GREEN, VERM, SKY = "#0072B2", "#E69F00", "#8a8a8a", "#009E73", "#D55E00", "#56B4E9"
ENC_COL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999", "#000000", "#7f3b08"]
PAD = 0.02   # FancyBboxPatch pad: drawn edge = nominal edge +/- PAD
GAP = 0.08   # clear space between an arrow end and the box or panel edge it points from or to
FS = 6.0
BOXES, ARROWS, TEXTS = [], [], []   # registered for check_layout()


def box(ax, x, y, w, h, text, fc="white", ec=GRAY, fs=FS, bold=False, ls="-", lw=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    BOXES.append(p)
    if text:
        TEXTS.append(ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                             fontweight="bold" if bold else "normal"))
    return (x - PAD, y - PAD, x + w + PAD, y + h + PAD)   # drawn extent


def label(ax, *a, **k):
    t = ax.text(*a, **k)
    TEXTS.append(t)
    return t


def harrow(ax, x_from, x_to, y, color=GRAY, ls="-"):
    """Horizontal arrow between two drawn edges, GAP clear at both ends."""
    s = 1 if x_to > x_from else -1
    p = FancyArrowPatch((x_from + s * GAP, y), (x_to - s * GAP, y), arrowstyle="-|>", mutation_scale=9,
                        lw=1.1, color=color, ls=ls, shrinkA=0, shrinkB=0)
    ax.add_patch(p)
    ARROWS.append(p)


def varrow(ax, x, y_from, y_to, color=GRAY):
    s = 1 if y_to > y_from else -1
    p = FancyArrowPatch((x, y_from + s * GAP), (x, y_to - s * GAP), arrowstyle="-|>", mutation_scale=9,
                        lw=1.1, color=color, shrinkA=0, shrinkB=0)
    ax.add_patch(p)
    ARROWS.append(p)


def chip(ax, x, y, members, s=0.13):
    for k, m in enumerate(members):
        ax.add_patch(Rectangle((x + k * s, y), s * 0.9, s * 0.9, fc=ENC_COL[m], ec="white", lw=0.4))


def panel(ax, x, w, title):
    p = FancyBboxPatch((x, 0.08), w, 3.25, boxstyle=f"round,pad={PAD},rounding_size=0.12",
                       fc="none", ec="#555555", lw=1.0, ls=(0, (4, 3)))
    ax.add_patch(p)
    BOXES.append(p)
    label(ax, x + w / 2, 3.12, title, ha="center", va="center", fontsize=6.6, fontweight="bold")
    return x - PAD, x + w + PAD


def check_layout(fig, min_gap_pt=1.5):
    """Every arrow must keep >= min_gap_pt printed points from every text, and must not touch any box
    or panel border (it lies wholly outside every box, or wholly inside one with that clearance)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pt = 72.0 / fig.dpi
    bad = []
    for a in ARROWS:
        ab = a.get_window_extent(r)
        for t in TEXTS:
            tb = t.get_window_extent(r)
            if ab.expanded(1, 1).padded(min_gap_pt / pt).overlaps(tb):
                bad.append(f"arrow {a.get_path().vertices[0]} too close to text {t.get_text()!r}")
        for b in BOXES:
            bb = b.get_window_extent(r)
            inside = (ab.x0 > bb.x0 + min_gap_pt / pt and ab.x1 < bb.x1 - min_gap_pt / pt
                      and ab.y0 > bb.y0 + min_gap_pt / pt and ab.y1 < bb.y1 - min_gap_pt / pt)
            if ab.padded(min_gap_pt / pt).overlaps(bb) and not inside:
                bad.append(f"arrow at {ab.bounds} touches box {bb.bounds}")
    for t in TEXTS:   # text must sit inside its box with clearance (no text touching a border)
        tb = t.get_window_extent(r)
        for b in BOXES:
            bb = b.get_window_extent(r)
            if tb.overlaps(bb):
                inside = (tb.x0 >= bb.x0 + min_gap_pt / pt and tb.x1 <= bb.x1 - min_gap_pt / pt
                          and tb.y0 >= bb.y0 + min_gap_pt / pt and tb.y1 <= bb.y1 - min_gap_pt / pt)
                if not inside:
                    bad.append(f"text {t.get_text()!r} crosses a box border {bb.bounds}")
    ext = [(t, t.get_window_extent(r)) for t in TEXTS]   # labels and threshold lines must not touch
    for i, (t1, b1) in enumerate(ext):
        for t2, b2 in ext[i + 1:]:
            if b1.padded(1.0 / pt).overlaps(b2):
                bad.append(f"{getattr(t1, 'get_text', lambda: 'line')()!r} touches "
                           f"{getattr(t2, 'get_text', lambda: 'line')()!r}")
    return bad


def main():
    style()
    fig, ax = plt.subplots(figsize=(7.2, 2.25))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    # (1) offline library ---------------------------------------------------------------------
    p1l, p1r = panel(ax, 0.04, 4.55, "(1) Build the library (offline)")
    for k in range(10):
        ax.add_patch(Rectangle((0.26 + k * 0.095, 2.47), 0.085, 0.3, fc=ENC_COL[k], ec="white", lw=0.4))
    label(ax, 0.73, 2.37, "10\nencoders", ha="center", va="top", fontsize=FS, linespacing=1.0)
    b1 = box(ax, 1.66, 2.34, 0.98, 0.52, "5-fold\ncross-fit")
    b2 = box(ax, 3.14, 2.34, 1.36, 0.52, "out-of-fold\npredictions")
    harrow(ax, 1.2, b1[0], 2.6)
    harrow(ax, b1[2], b2[0], 2.6)
    lib = box(ax, 0.24, 0.3, 4.14, 1.52, "", fc="#e8f6fb", ec=SKY)
    label(ax, 0.4, 1.62, "library $\\mathcal{L}$: all collections, $|S|\\leq 3$", fontsize=FS,
          color=BLUE, va="center")
    combos = [(5,), (6, 7), (1, 8), (0, 3, 9), (4, 6), (2,), (1, 2, 8), (6,), (3, 4), (0, 5, 7), (9,), (1,)]
    for k, c in enumerate(combos):
        chip(ax, 0.5 + (k % 6) * 0.65, 1.16 - (k // 6) * 0.34, c)
    label(ax, 2.31, 0.38, "utility of each collection\non each training image", ha="center", fontsize=FS)
    varrow(ax, 3.82, b2[1], lib[3], color=BLUE)

    # (2) retrieval ----------------------------------------------------------------------------
    p2l, p2r = panel(ax, 5.14, 4.5, "(2) Retrieve per image, no label")
    harrow(ax, p1r, p2l, 1.1, color=BLUE, ls=(0, (2, 1.2)))          # library -> retrieval, in the gap
    bi = box(ax, 5.34, 2.44, 0.84, 0.5, "image\n$x$")
    be = box(ax, 6.62, 2.44, 1.44, 0.5, "ConvNeXt-T\n$z(x)$")
    harrow(ax, bi[2], be[0], 2.69)
    rs = box(ax, 5.34, 0.3, 2.1, 1.66, "", fc="#f7f7f7", ec="#bbbbbb", lw=0.8)
    rng = np.random.RandomState(3)
    cents = [(5.78, 1.5), (6.88, 1.5), (5.92, 0.82), (6.98, 0.85)]
    for (cx, cy), col in zip(cents, [BLUE, ORANGE, GREEN, VERM]):
        pts = rng.normal(0, 0.13, size=(14, 2))
        ax.scatter(cx + pts[:, 0], cy + pts[:, 1] * 0.75, s=3, color=col, alpha=0.6, lw=0)
    ax.add_patch(Circle((6.88, 1.5), 0.27, fc="none", ec="black", lw=0.8, ls="--"))
    ax.scatter([6.88], [1.5], marker="*", s=40, color="black", zorder=5)
    label(ax, 6.39, 0.4, "router space", ha="center", fontsize=FS)
    varrow(ax, 7.2, be[1], rs[3])
    bp = box(ax, 7.9, 1.28, 1.6, 1.04, "average\nneighbour\nutilities,\npick the best", ec=BLUE)
    bt = box(ax, 7.9, 0.3, 1.6, 0.72, "task vote of\nneighbours\n(picks heads)", ec=GRAY)
    harrow(ax, rs[2], bp[0], 1.75, color=BLUE)
    harrow(ax, rs[2], bt[0], 0.66, color=GRAY)

    # (3) run ----------------------------------------------------------------------------------
    p3l, p3r = panel(ax, 10.19, 4.15, "(3) Run collection or harness")
    bh = box(ax, 10.34, 2.35, 3.85, 0.55, "HealthRouter: run and\naverage the collection",
             fc="#eaf2fa", ec=BLUE, bold=True)
    bg = box(ax, 10.34, 0.3, 3.85, 1.85, "", fc="#eaf7f2", ec=GREEN)
    label(ax, 12.265, 2.03, "Harness: stop when confident", ha="center", va="center", fontsize=FS,
          fontweight="bold")
    # Two mini charts (user round 4): bars = fused confidence after calls 1-3; dashed line = threshold theta.
    # Schematic heights, not data. The easy image crosses theta at call 1; the hard image needs 3 calls.
    base, scale, theta = 0.68, 0.78, 0.8
    label(ax, 10.5, 1.02, "confidence", rotation=90, ha="center", va="center", fontsize=FS - 0.4)
    for x0, name, conf in [(10.66, "easy image", [0.93]), (12.5, "hard image", [0.58, 0.66, 0.88])]:
        label(ax, x0 + 0.72, 1.8, name, ha="center", va="center", fontsize=FS)
        for k in range(3):
            bx = x0 + 0.1 + k * 0.5
            label(ax, bx + 0.16, 0.53, str(k + 1), ha="center", va="center", fontsize=FS - 0.4)
            if k < len(conf):
                ax.add_patch(Rectangle((bx, base), 0.32, conf[k] * scale,
                                       fc=GREEN if conf[k] >= theta else "#9ecae1", ec="white", lw=0.4))
            else:   # call not made
                ax.add_patch(Rectangle((bx, base), 0.32, 0.2, fc="none", ec="#9a9a9a", lw=0.5, ls=(0, (1.5, 1.5))))
        ax.plot([x0, x0 + 1.5], [base, base], color="#555555", lw=0.6)
        yt = base + theta * scale
        TEXTS.append(ax.plot([x0, x0 + 1.42], [yt, yt], ls=(0, (3, 1.6)), color="black", lw=0.8)[0])
        label(ax, x0 + 1.47, yt, "$\\theta$", fontsize=FS + 0.4, va="center", ha="left")
        k_stop = len(conf) - 1
        label(ax, x0 + 0.1 + k_stop * 0.5 + 0.16, 1.58, "stop", ha="center", va="center", fontsize=FS - 0.4,
              color=GREEN, fontweight="bold")
        if k_stop > 0:
            label(ax, x0 + 0.1 + 0.41, 1.58, "continue", ha="center", va="center", fontsize=FS - 0.4,
                  color="#3d7fb5")
    label(ax, 12.33, 0.53, "call", ha="center", va="center", fontsize=FS - 0.4, color="#555555")
    # retrieval -> run, both inside the white gap between panels (2) and (3)
    harrow(ax, p2r, p3l, 2.62, color=BLUE)     # pick the best -> HealthRouter
    harrow(ax, p2r, p3l, 1.5, color=GREEN)     # pick the best -> harness

    bad = check_layout(fig)
    if bad:
        print("\n".join(bad))
        raise SystemExit("router_method: layout check failed")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "router_method.pdf", bbox_inches="tight")
    fig.savefig(OUT / "router_method.png", dpi=400, bbox_inches="tight")
    print("wrote", OUT / "router_method.pdf", "(layout check passed)")


if __name__ == "__main__":
    main()

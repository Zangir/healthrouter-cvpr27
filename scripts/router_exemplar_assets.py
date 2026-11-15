#!/usr/bin/env python3
"""Figures and tables of the exemplar-driven layout (paper/EXEMPLAR_PLAN.md, approved 2026-11-14).

Called from router_paper_assets.main(). Reads results/router_main.json, router_external.json,
router_general.json and router_descriptive.json (plan amendment 4). Every plotted or tabulated
number is read from these files; derived numbers are written to router_paper_numbers.json with
their formula through router_paper_assets.put().

  F1  figures/router_teaser.pdf      accuracy against encoder time (VL-RouterBench Fig. 1 layout)
  F3  figures/router_selection.pdf   (a) selection heatmap + popularity bar (SemLA Fig. 3),
                                     (b) harness calls vs difficulty, (c) gain calibration (SemLA Fig. 5)
  S-F1 figures/router_gain_tasks.pdf  gain calibration per task (supplement)
  T1  tables/router_main_body.tex     leaderboard (VL-RouterBench Tab. 2 + SemLA Tab. 1 delta row)
  T2  tables/router_pool_body.tex     candidate pool (VL-RouterBench Tab. 1)
  T3  tables/router_ablation_body.tex ablations incl. label-free inference
  T4  tables/router_external_body.tex external transfer incl. label-free row and rank footer
  S-T* supplement tables (candidates, sweeps, label-free, difficulty, six-task)
"""
from __future__ import annotations

import numpy as np

import router_paper_assets as A
from paper_style import band, style

plt = A.plt
TASKS = A.TASKS
F3 = A.f3
BLUE, GREEN, ORANGE, GRAY = "#0072B2", "#009E73", "#E69F00", "#8a8a8a"
VERM, PURPLE, SKY, YELLOW = "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"
TASK_COL = {"chest_xray": BLUE, "pathology": ORANGE, "fundus": GREEN, "mammography": VERM}
TNAME = {"chest_xray": "Chest X-ray", "pathology": "Pathology", "fundus": "Fundus", "mammography": "Mammography",
         "eurosat": "EuroSAT", "dtd": "DTD"}
ENC = ["imagenet_convnext", "dinov2_base", "dinov2_large", "siglip2_base", "biomedclip", "rad_dino", "phikon",
       "phikon_v2", "retfound", "mammo_clip"]
PRETTY = {"imagenet_convnext": "ConvNeXt-T", "dinov2_base": "DINOv2-B", "dinov2_large": "DINOv2-L",
          "siglip2_base": "SigLIP~2-B", "biomedclip": "BiomedCLIP", "rad_dino": "RAD-DINO", "phikon": "Phikon",
          "phikon_v2": "Phikon-v2", "retfound": "RETFound", "mammo_clip": "Mammo-CLIP"}
PLAIN = {k: v.replace("~", " ") for k, v in PRETTY.items()}
SHORT = {"chest_xray": "Chest", "pathology": "Path.", "fundus": "Fundus", "mammography": "Mammo."}
KIND = {"imagenet_convnext": "G", "dinov2_base": "G", "dinov2_large": "G", "siglip2_base": "G",
        "biomedclip": "B", "rad_dino": "D", "phikon": "D", "phikon_v2": "D", "retfound": "D", "mammo_clip": "D"}
FAMILY = [  # (label, colour, marker, members) -- one colour per family in every figure
    ("Single models", GRAY, "o", ["best_single_overall", "best_single_per_task", "best_single_per_task_oof"]),
    ("Task-level ensembles", ORANGE, "s", ["uniform_ensemble", "stacking", "greedy_ensemble_selection", "logme_selection"]),
    ("Modality routers", YELLOW, "v", ["modality_router_supervised", "modality_router_supervised_oof",
                                       "modality_router_zeroshot_biomedclip"]),
    ("MoE gates", VERM, "^", ["soft_moe", "sparse_topk_moe"]),
    ("Dynamic selection", PURPLE, "P", ["knora_u", "knora_e", "meta_des"]),
    ("Per-sample routers", SKY, "D", ["knn_router", "routerbench_mlp", "routellm_mf"]),
]
GROUPS_T1 = [
    ("Single models", ["best_single_overall", "best_single_per_task", "best_single_per_task_oof"]),
    ("Task-level ensembles", ["uniform_ensemble", "stacking", "greedy_ensemble_selection", "logme_selection"]),
    ("Modality routers", ["modality_router_supervised", "modality_router_supervised_oof", "modality_router_zeroshot_biomedclip"]),
    ("Mixture-of-experts gates", ["soft_moe", "sparse_topk_moe"]),
    ("Dynamic ensemble selection", ["knora_u", "knora_e", "meta_des"]),
    ("Non-parametric per-sample router", ["knn_router"]),
    ("Learned utility routers", ["routerbench_mlp", "routellm_mf"]),
    ("Ours", ["healthrouter", "healthrouter_harness"]),
]


def save(fig, stem):
    A.FIG.mkdir(exist_ok=True)
    fig.savefig(A.FIG / f"{stem}.pdf")
    fig.savefig(A.FIG / f"{stem}.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- F1 teaser
def teaser(main, desc):
    # Drawn at the printed column width (3.28 in) so that no label prints below 7 pt (review 2026-11-14).
    style()
    FS = 7.4
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS})
    M = main["methods"]
    S = desc["single_candidates"]["candidates"]
    cost = dict(zip(ENC, main["candidate_cost_ms"]))
    fig, ax = plt.subplots(figsize=(2.98, 2.25))
    lo_y, hi_y = 0.789, 0.898
    # single candidates (gray) and their Pareto frontier
    pts = sorted((cost[e], S[e]["macro"]["value"], e) for e in ENC)
    for x, y, e in pts:
        if y >= lo_y:
            ax.scatter(x, y, s=16, color="#c4c4c4", zorder=2, edgecolor="none")
    front, best = [], -1
    for x, y, e in pts:
        if y > best:
            front.append((x, y))
            best = y
    ax.plot(*zip(*front), ls="--", color="#b0b0b0", lw=1.0, zorder=1)
    off = [e for _, y, e in pts if y < lo_y]
    for e in off:
        ax.annotate(f"{PLAIN[e]} alone: {F3(S[e]['macro']['value'])}↓", (cost[e] * 0.93, lo_y + 0.001),
                    fontsize=FS, color="#8a8a8a", ha="left")
    ch = min(ENC, key=lambda e: cost[e])
    ax.annotate("Cheapest", (cost[ch], S[ch]["macro"]["value"]), textcoords="offset points", xytext=(-4, -9),
                fontsize=FS, color="#666666", ha="left")
    so = M["best_single_overall"]
    ax.annotate("Strongest", (so["latency_ms_mean"], so["macro"]["value"]), textcoords="offset points",
                xytext=(-14, -10), fontsize=FS, color="#666666")
    handles = []
    for lab, col, mk, ns in FAMILY:
        for n in ns:
            h = ax.scatter(M[n]["latency_ms_mean"], M[n]["macro"]["value"], s=22, marker=mk, color=col,
                           edgecolor="#333333" if col == YELLOW else "none", linewidth=0.5, zorder=3)
        short = {"Single models": "Single models", "Task-level ensembles": "Task ensembles"}.get(lab, lab)
        handles.append((h, short))
    # cost dial (lambda) and harness dial (theta)
    lam = [(M["healthrouter"]["latency_ms_mean"], M["healthrouter"]["macro"]["value"])]
    lam += [(M[k]["latency_ms_mean"], M[k]["macro"]["value"]) for k in M if k.startswith("cost_lam")]
    lam.sort()
    ax.plot(*zip(*lam), "-", color=BLUE, lw=1.0, alpha=0.8, zorder=4)
    th = sorted((M[k]["latency_ms_mean"], M[k]["macro"]["value"]) for k in M if k.startswith("harness_theta"))
    ax.plot(*zip(*th), "-o", color=GREEN, lw=1.0, ms=2.2, zorder=4)
    hm = M["healthrouter_harness"]
    h1 = ax.scatter(hm["latency_ms_mean"], hm["macro"]["value"], s=40, marker="D", color=GREEN, edgecolor="white", zorder=6)
    hr = M["healthrouter"]
    h2 = ax.scatter(hr["latency_ms_mean"], hr["macro"]["value"], s=110, marker="*", color=BLUE, edgecolor="white", zorder=7)
    ax.text(hr["latency_ms_mean"] * 1.40, hr["macro"]["value"] + 0.0010, "HealthRouter", fontsize=FS + 0.6,
            color=BLUE, fontweight="bold", ha="left", va="bottom")
    ax.text(th[0][0] * 0.92, th[0][1], "Harness", fontsize=FS + 0.6, color=GREEN, fontweight="bold",
            ha="right", va="center")
    for n, txt, dy in [("stacking", "Stacking (tied)", 1), ("soft_moe", "Soft MoE", -2.5),
                       ("uniform_ensemble", "Uniform", -10), ("knora_u", "KNORA-U", -2.5),
                       ("meta_des", "META-DES", -2.5), ("knora_e", "KNORA-E", -2.5)]:
        ax.annotate(txt, (M[n]["latency_ms_mean"], M[n]["macro"]["value"]), textcoords="offset points",
                    xytext=(5, dy), fontsize=FS, color="#333333")
    kn = M["knn_router"]
    ax.annotate("Neighbourhood router (tied)", (kn["latency_ms_mean"], kn["macro"]["value"]), xytext=(6.2, 0.8475),
                fontsize=FS, color="#333333", arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5,
                                                              shrinkA=0, shrinkB=3))
    o = M["oracle_single"]["macro"]["value"]
    ax.text(5.6, hi_y - 0.0008, f"Oracle {F3(o)} \u2191 (off scale)", fontsize=FS, color="black",
            ha="left", va="top")
    ax.annotate("", xy=(520, 0.833), xytext=(950, 0.818),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))
    ax.text(480, 0.808, "better", fontsize=FS, color="#555555")
    ax.set_xscale("log")
    ax.set_xticks([6, 10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(["6", "10", "20", "50", "100", "200", "500"])
    ax.set_xlim(5, 1100)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("encoder time per image (ms)")
    ax.set_ylabel("macro score, 4 medical tasks")
    sc = ax.scatter([], [], s=16, color="#c4c4c4")
    leg = [(sc, "Encoders alone"), *handles, (h2, "HealthRouter"), (h1, "Harness")]
    ax.legend([h for h, _ in leg], [l for _, l in leg], loc="lower center", ncol=3, fontsize=FS,
              bbox_to_anchor=(0.43, 1.01), frameon=True, columnspacing=0.5, handletextpad=0.15,
              borderpad=0.3, labelspacing=0.2, handlelength=1.0, borderaxespad=0.2)
    save(fig, "router_teaser")
    plt.rcdefaults()


# ----------------------------------------------------------------------------- F3 selection analysis
def selection_figure(main, desc):
    style()
    FS = 7.4  # no printed label below 7 pt at text width (review 2026-11-14)
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "axes.titlesize": FS + 0.3})
    M = main["methods"]
    n = {t: M["healthrouter"]["per_task"][t]["n"] for t in TASKS}
    F = np.array([M["healthrouter"]["selection_frequency"][t] for t in TASKS])
    pooled = (F * np.array([n[t] for t in TASKS])[:, None]).sum(0) / sum(n.values())
    fig = plt.figure(figsize=(7.0, 2.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.55, 1.05, 1.2], height_ratios=[0.55, 2.0], wspace=0.45, hspace=0.06)
    axb = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[1, 0])
    axb.bar(range(len(ENC)), 100 * pooled, color="#6baed6", width=0.75)
    axb.set_xlim(-0.5, len(ENC) - 0.5)
    axb.set_xticks([])
    axb.set_yticks([0, 50])
    axb.set_ylim(0, 65)
    axb.set_ylabel("pooled", fontsize=FS)
    axb.grid(False)
    axb.set_title("(a) share of images (%) that run each model", fontsize=FS + 0.3, loc="left")
    axh.pcolormesh(np.arange(F.shape[1] + 1) - 0.5, np.arange(F.shape[0] + 1) - 0.5, F, cmap="Blues",
                   vmin=0, vmax=1)  # vector cells, no embedded raster
    axh.set_xlim(-0.5, F.shape[1] - 0.5)
    axh.set_ylim(F.shape[0] - 0.5, -0.5)
    axh.set_xticks(range(len(ENC)))
    axh.set_xticklabels([PLAIN[e] for e in ENC], rotation=32, ha="right")
    axh.set_yticks(range(len(TASKS)))
    axh.set_yticklabels([TNAME[t] for t in TASKS])
    axh.grid(False)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if F[i, j] >= 0.05:
                axh.text(j, i, f"{100 * F[i, j]:.0f}", ha="center", va="center", fontsize=7.4,
                         color="white" if F[i, j] > 0.55 else "black")
    # (b) harness calls vs difficulty
    D = desc["harness_difficulty"]
    ax = fig.add_subplot(gs[:, 1])
    x = np.arange(len(TASKS))
    one = [D[t]["error_one_call"] for t in TASKS]
    mul = [D[t]["error_multi_call"] for t in TASKS]
    e1 = np.array([[D[t]["error_one_call"] - D[t]["error_one_call_ci"][0], D[t]["error_one_call_ci"][1] - D[t]["error_one_call"]] for t in TASKS]).T
    e2 = np.array([[D[t]["error_multi_call"] - D[t]["error_multi_call_ci"][0], D[t]["error_multi_call_ci"][1] - D[t]["error_multi_call"]] for t in TASKS]).T
    ax.bar(x - 0.19, one, 0.38, yerr=e1, color="#9ecae1", label="1 call", error_kw=dict(lw=0.6, capsize=1.5))
    ax.bar(x + 0.19, mul, 0.38, yerr=e2, color=GREEN, label="$\\geq$2 calls", error_kw=dict(lw=0.6, capsize=1.5))
    for i, t in enumerate(TASKS):
        ax.text(i, max(one[i], mul[i]) + 0.045, f"{100 * D[t]['share_one_call']:.0f}", ha="center", fontsize=FS, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels(["Chest", "Path.", "Fundus", "Mammo."], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("error of best single model")
    ax.set_ylim(0, 0.72)
    ax.legend(fontsize=FS, loc="upper left", ncol=1, handlelength=1.0, borderaxespad=0.2, labelspacing=0.2)
    ax.set_title("(b) calls vs difficulty", fontsize=FS + 0.3, loc="left")
    # (c) gain calibration
    G = desc["gain_calibration"]
    ax = fig.add_subplot(gs[:, 2])
    for t in TASKS:
        b = G[t]["bins"]
        xs = [v["x_mean"] for v in b]
        ys = [v["y_mean"] for v in b]
        lo = [v["y_ci"][0] for v in b]
        hi = [v["y_ci"][1] for v in b]
        ax.plot(xs, ys, "-o", color=TASK_COL[t], ms=2.2, lw=0.9, label=f"{SHORT[t]} $\\rho$={G[t]['spearman']:+.2f}")
        band(ax, xs, lo, hi, TASK_COL[t])
    ax.axhline(0, color="#777777", lw=0.6, ls="--")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(left=0)
    ax.set_xticks([0, 0.01, 0.1])
    ax.set_xticklabels(["0", "0.01", "0.1"])
    ax.set_xlabel("estimated advantage (nats)")
    ax.set_ylabel("realised advantage (nats)")
    ax.legend(fontsize=FS, loc="upper left", handlelength=1.0, borderaxespad=0.2, labelspacing=0.15)
    ax.set_title("(c) estimated vs realised gain", fontsize=FS + 0.3, loc="left")
    save(fig, "router_selection")
    plt.rcdefaults()


def gain_tasks_figure(desc):
    style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    G = desc["gain_calibration"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharey=False)
    for ax, t in zip(axes, TASKS):
        b = G[t]["bins"]
        xs = np.array([v["x_mean"] for v in b])
        ys = np.array([v["y_mean"] for v in b])
        err = np.array([[v["y_mean"] - v["y_ci"][0], v["y_ci"][1] - v["y_mean"]] for v in b]).T
        ax.errorbar(xs, ys, yerr=err, fmt="o", color=TASK_COL[t], ms=2.5, lw=0.7, capsize=1.5)
        xx = np.linspace(xs.min(), xs.max(), 20)
        ax.plot(xx, G[t]["slope"] * xx + G[t]["intercept"], "--", color="#555555", lw=0.8)
        ax.axhline(0, color="#999999", lw=0.5)
        lo, hi = G[t]["spearman_ci"]
        ax.set_title(f"{TNAME[t]}\n$\\rho$={G[t]['spearman']:+.3f} [{lo:+.3f}, {hi:+.3f}]", fontsize=7.5)
    axes[0].set_ylabel("realised advantage (nats)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.supxlabel("estimated advantage (nats)", fontsize=8, y=0.02)
    save(fig, "router_gain_tasks")
    plt.rcdefaults()


# ----------------------------------------------------------------------------- T1 leaderboard
def _best_second(vals):
    u = sorted({round(v, 3) for v in vals.values()}, reverse=True)
    b = u[0]
    s = u[1] if len(u) > 1 else None
    return ({k for k, v in vals.items() if round(v, 3) == b}, {k for k, v in vals.items() if s is not None and round(v, 3) == s})


def main_table(main):
    M, D = main["methods"], main["differences_vs_healthrouter"]
    rows = [n for _, ns in GROUPS_T1 for n in ns]
    cols = TASKS + ["macro"]
    hl = {}
    for c in cols:
        vals = {n: (M[n]["macro"]["value"] if c == "macro" else M[n]["per_task"][c]["value"]) for n in rows}
        hl[c] = _best_second(vals)
    def calls_ms(n):
        return {"calls": M[n]["invoked_mean"], "ms": M[n]["latency_ms_mean"]}
    low = {k: _best_second({n: -calls_ms(n)[k] for n in rows}) for k in ("calls", "ms")}
    o = M["oracle_single"]
    lines = ["\\textit{Upper bound (uses the label)} &&&&&&&&\\\\",
             f"{A.NAMES['oracle_single']} & " + " & ".join(f"\\textcolor{{gray}}{{{F3(o['per_task'][t]['value'])}}}" for t in TASKS) +
             f" & \\textcolor{{gray}}{{{F3(o['macro']['value'])}}} & \\textcolor{{gray}}{{1.00}} & \\textcolor{{gray}}{{{o['latency_ms_mean']:.0f}}} & --\\\\",
             "\\midrule"]
    for g, ns in GROUPS_T1:
        lines.append(f"\\multicolumn{{9}}{{c}}{{\\textit{{{g}}}}}\\\\")
        for n in ns:
            cells = []
            for c in cols:
                e = M[n]["macro"] if c == "macro" else M[n]["per_task"][c]
                v = F3(e["value"])
                if n in hl[c][0]:
                    v = f"\\textbf{{{v}}}"
                elif n in hl[c][1]:
                    v = f"\\underline{{{v}}}"
                cells.append(v)  # means only (user 2026-11-15); paired intervals are in the supplement
            for k, fmt in (("calls", "{:.2f}"), ("ms", "{:.0f}")):
                v = fmt.format(calls_ms(n)[k])
                v = f"\\textbf{{{v}}}" if n in low[k][0] else (f"\\underline{{{v}}}" if n in low[k][1] else v)
                cells.append(v)
            if n == "healthrouter":
                sig = "--"
            else:
                sig = "$\\blacktriangledown$" if D[n]["macro"]["ci"][0] > 0 else "$\\approx$"
            lines.append(f"{A.NAMES[n]} & " + " & ".join(cells) + f" & {sig}\\\\")
            if n == "healthrouter":
                du = D["uniform_ensemble"]
                cells = []
                for c in cols:
                    v = du[c]["value"]
                    col = "dgreen" if v > 0 else ("dred" if v < 0 else "black")
                    cells.append(f"\\textcolor{{{col}}}{{{v:+.3f}}}".replace("{-", "{$-$"))
                lines.append("\\quad{\\footnotesize $\\Delta$ vs.\\ uniform ensemble} & " + " & ".join(cells) + " &&&\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    (A.PAPER / "tables" / "router_main_body.tex").write_text("\n".join(lines) + "\n")
    for c in cols:
        A.put(f"d_uniform_{c}", D["uniform_ensemble"][c]["value"], f"{D['uniform_ensemble'][c]['value']:+.3f}",
              f"differences_vs_healthrouter.uniform_ensemble.{c}", D["uniform_ensemble"][c]["ci"])


# ----------------------------------------------------------------------------- T2 candidate pool
def pool_table(main, desc):
    M = main["methods"]
    S = desc["single_candidates"]["candidates"]
    cost = dict(zip(ENC, main["candidate_cost_ms"]))
    n = {t: M["healthrouter"]["per_task"][t]["n"] for t in TASKS}
    F = {t: dict(zip(ENC, M["healthrouter"]["selection_frequency"][t])) for t in TASKS}
    best = {t: max(ENC, key=lambda e: S[e]["per_task"][t]["value"]) for t in TASKS}
    lines = []
    for e in sorted(ENC, key=lambda e: cost[e]):
        sel = sum(F[t][e] * n[t] for t in TASKS) / sum(n.values())
        A.put(f"sel_{e}", 100 * sel, f"{100 * sel:.0f}\\%", "sum_t selection_frequency[t][enc]*n_t / sum_t n_t (router_main.json)")
        cells = []
        for t in TASKS:
            v = F3(S[e]["per_task"][t]["value"]).replace("-", "$-$")
            cells.append(f"\\textbf{{{v}}}" if best[t] == e else v)
        lines.append(f"{PRETTY[e]} & {KIND[e]} & {cost[e]:.1f} & " + " & ".join(cells) + f" & {100 * sel:.0f}\\\\")
    lines.append("\\bottomrule")
    (A.PAPER / "tables" / "router_pool_body.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- T3 ablations (adds label-free rows)
def ablation_table(main, desc):
    M, D = main["methods"], main["differences_vs_healthrouter"]
    LF = desc["label_free"]["in_distribution"]
    groups = [
        [("healthrouter", "HealthRouter (full)")],
        [("abl_size1", "Single models ($s{=}1$)"), ("abl_size2", "Up to two ($s{=}2$)"),
         ("abl_size3", "Up to three ($s{=}3$)"), ("abl_size4", "Up to four ($s{=}4$)")],
        [("abl_picks1", "Hard pick only"), ("abl_picks3", "Soft top-3 only"), ("abl_geometric_fusion", "Geometric fusion")],
        [("abl_feat_biomedclip", "Features: BiomedCLIP"), ("abl_feat_dinov2_large", "Features: DINOv2-L"),
         ("abl_feat_task_id", "Task identity (oracle)")],
        [("healthrouter_labelfree", "Task inferred from neighbours$^\\ddagger$")],
        [("hr_collection_mlp", "MLP gate, collections"), ("hr_candidate_mlp", "MLP gate, single models")],
        [("healthrouter_fallback", "Distance fallback$^\\dagger$")],
    ]
    rows = {n: M[n]["macro"]["value"] for g in groups for n, _ in g if n in M}
    b, s = _best_second(rows)
    lines = []
    for g in groups:
        for n, label in g:
            if n == "healthrouter_labelfree":
                m = LF["methods"][n]
                dd = LF["differences_vs_healthrouter"][n]["macro"]
                calls, ms = M["healthrouter"]["invoked_mean"], M["healthrouter"]["latency_ms_mean"]
                v = F3(m["macro"]["value"])
            elif n in M:
                m = M[n]
                dd = None if n == "healthrouter" else D[n]["macro"]
                calls, ms = m["invoked_mean"], m["latency_ms_mean"]
                v = F3(m["macro"]["value"])
                v = f"\\textbf{{{v}}}" if n in b else (f"\\underline{{{v}}}" if n in s else v)
            else:
                continue
            if dd is None:
                d = "--"
            else:  # variant minus HealthRouter, with the Table 1 significance marks (user 2026-11-15)
                mark = "$\\blacktriangledown$" if dd['ci'][0] > 0 else ("$\\blacktriangle$" if dd['ci'][1] < 0 else "$\\approx$")
                d = f"{0.0 - dd['value']:+.3f}\\,{mark}"
            lines.append(f"{label} & {v} & {d} & {calls:.2f} & {ms:.0f}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    (A.PAPER / "tables" / "router_ablation_body.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- T4 external
def external_table(ext, desc):
    R = ext["results"]
    LFE = desc["label_free"]["external"]
    CONDS = A.CONDS
    rows = [n for n in A.EXT_ROWS if all(n in R[c] for c in CONDS)]
    vals = {c: {n: R[c][n]["macro"]["value"] for n in rows if "fallback" not in n} for c in CONDS}
    hl = {c: _best_second(vals[c]) for c in CONDS}
    lines = []
    for n in rows:
        cells = []
        for c in CONDS:
            v = R[c][n]["macro"]["value"]
            sv = F3(v)
            if n in hl[c][0]:
                sv = f"\\textbf{{{sv}}}"
            elif n in hl[c][1]:
                sv = f"\\underline{{{sv}}}"
            cells.append(sv)
        nm = A.EXT_SHORT.get(n, A.SHORT.get(n, n))
        lines.append(f"{nm} & " + " & ".join(cells) + f" & {np.mean([R[c][n]['macro']['value'] for c in CONDS]):.3f}\\\\")
        if n == "routellm_mf":
            lines.append("\\midrule")
        if n == "healthrouter_harness":
            lf = [LFE[c]["methods"]["healthrouter_labelfree"]["macro"]["value"] for c in CONDS]
            lines.append("HealthRouter, task inferred$^\\ddagger$ & " + " & ".join(F3(v) for v in lf) + f" & {np.mean(lf):.3f}\\\\")
            ranks = [int(A.NUM[f"ext_rank_hr_{c}"]["value"]) for c in CONDS]
            lines.append("\\quad{\\footnotesize rank of HealthRouter (of 20)} & " + " & ".join(str(r) for r in ranks) + " & \\\\")
            lines.append("\\midrule")
    (A.PAPER / "tables" / "router_external_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")


# ----------------------------------------------------------------------------- supplement tables
def supp_tables(main, gen, desc):
    M = main["methods"]
    S = desc["single_candidates"]
    cost = dict(zip(ENC, main["candidate_cost_ms"]))
    # S-T4: every candidate alone with CIs
    lines = []
    for e in sorted(ENC, key=lambda e: cost[e]):
        c = S["candidates"][e]
        cells = [f"{F3(c['per_task'][t]['value'])}{{\\tiny\\,[{F3(c['per_task'][t]['ci'][0])},\\,{F3(c['per_task'][t]['ci'][1])}]}}" for t in TASKS]
        cells.append(f"{F3(c['macro']['value'])}{{\\tiny\\,[{F3(c['macro']['ci'][0])},\\,{F3(c['macro']['ci'][1])}]}}")
        lines.append(f"{PRETTY[e]} & {cost[e]:.1f} & " + " & ".join(cells) + "\\\\")
    pb = S["per_image_other_candidate_better"]
    lines.append("\\midrule")
    lines.append("\\multicolumn{2}{@{}l}{Another candidate better on the image (\\%)} & " +
                 " & ".join(f"{100 * pb[t]:.1f}" for t in TASKS) + f" & {100 * pb['all']:.1f}\\\\")
    (A.PAPER / "tables" / "supp_candidates_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")
    # S-T5: cost and threshold sweeps
    lines = []
    lam = sorted([(0.0, "healthrouter")] + [(float(k[len("cost_lam"):]), k) for k in M if k.startswith("cost_lam")])
    for v, k in lam:
        m = M[k]
        lines.append(f"$\\lambda{{=}}{v:g}$ & {F3(m['macro']['value'])}{{\\tiny\\,[{F3(m['macro']['ci'][0])},\\,{F3(m['macro']['ci'][1])}]}} & {m['invoked_mean']:.2f} & {m['latency_ms_mean']:.0f}\\\\")
    lines.append("\\midrule")
    th = sorted((float(k[len("harness_theta"):]), k) for k in M if k.startswith("harness_theta"))
    for v, k in th:
        m = M[k]
        lines.append(f"$\\theta{{=}}{v:g}$ & {F3(m['macro']['value'])}{{\\tiny\\,[{F3(m['macro']['ci'][0])},\\,{F3(m['macro']['ci'][1])}]}} & {m['invoked_mean']:.2f} & {m['latency_ms_mean']:.0f}\\\\")
    (A.PAPER / "tables" / "supp_sweeps_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")
    # S-T6: label-free inference
    L = desc["label_free"]
    lines = []
    for key, lab, tasks in (("in_distribution", "In distribution (4 tasks)", TASKS),
                            ("six_task", "Six tasks (+ EuroSAT, DTD)", TASKS + ["eurosat", "dtd"])):
        v = L[key]
        acc = v["task_inference_accuracy"]["all"]
        mis = sum(v["misinferred_images"].values())
        ev = sum(v["image_evaluations"].values())
        for base, lf, nm in (("healthrouter", "healthrouter_labelfree", "HealthRouter"),
                             ("healthrouter_harness", "harness_labelfree", "Harness")):
            b_ = v["methods"][base]["macro"]
            l_ = v["methods"][lf]["macro"]
            if lf == "healthrouter_labelfree":
                dd = v["differences_vs_healthrouter"][lf]["macro"]
                dtxt = f"{0.0 - dd['value']:+.3f}{{\\tiny\\,[{0.0 - dd['ci'][1]:+.3f},\\,{0.0 - dd['ci'][0]:+.3f}]}}"
            else:
                dtxt = f"{l_['value'] - b_['value']:+.3f}"
            lines.append(f"{lab} & {nm} & {100 * acc:.2f} ({mis}/{ev}) & {F3(b_['value'])} & {F3(l_['value'])} & {dtxt}\\\\")
            A.put(f"lf_{key}_{lf}", l_["value"], F3(l_["value"]), f"router_descriptive.label_free.{key}.methods.{lf}.macro", l_["ci"])
        lines.append("\\midrule")
    cname = {"standard": "Clean", "grayscale": "Grayscale", "blur": "Blur", "low_resolution": "Low resolution", "jpeg": "JPEG"}
    for c in A.CONDS:
        v = L["external"][c]
        b_ = v["methods"]["healthrouter"]["macro"]["value"]
        l_ = v["methods"]["healthrouter_labelfree"]["macro"]
        dd = l_ if False else v["methods"]["healthrouter_labelfree"]["diff_base_minus_this"]
        lines.append(f"External, {cname[c]} & HealthRouter & {100 * v['task_inference_accuracy']['all']:.2f} & {F3(b_)} & "
                     f"{F3(l_['value'])} & {0.0 - dd['value']:+.3f}{{\\tiny\\,[{0.0 - dd['ci'][1]:+.3f},\\,{0.0 - dd['ci'][0]:+.3f}]}}\\\\")
    (A.PAPER / "tables" / "supp_labelfree_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")
    # S-T6b: external task-inference accuracy per cohort
    lines = []
    for c in A.CONDS:
        acc = L["external"][c]["task_inference_accuracy"]
        lines.append(f"{cname[c]} & " + " & ".join(f"{100 * acc[t]:.1f}" for t in TASKS) + f" & {100 * acc['all']:.1f}\\\\")
    (A.PAPER / "tables" / "supp_taskinf_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")
    # S-T7: harness difficulty
    Dd = desc["harness_difficulty"]
    lines = []
    for t in TASKS:
        d = Dd[t]
        lines.append(f"{TNAME[t]} & {100 * d['share_one_call']:.0f} & "
                     f"{F3(d['error_one_call'])}{{\\tiny\\,[{F3(d['error_one_call_ci'][0])},\\,{F3(d['error_one_call_ci'][1])}]}} & "
                     f"{F3(d['error_multi_call'])}{{\\tiny\\,[{F3(d['error_multi_call_ci'][0])},\\,{F3(d['error_multi_call_ci'][1])}]}} & "
                     f"{F3(d['logloss_one_call'])} & {F3(d['logloss_multi_call'])} & "
                     f"{d['spearman_calls_logloss']:.3f}{{\\tiny\\,[{d['spearman_calls_logloss_ci'][0]:.3f},\\,{d['spearman_calls_logloss_ci'][1]:.3f}]}}\\\\")
    (A.PAPER / "tables" / "supp_difficulty_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")
    # S-T8: gain calibration summary
    G = desc["gain_calibration"]
    lines = []
    for t in TASKS:
        g = G[t]
        lines.append(f"{TNAME[t]} & {g['n_pairs']:,} & ".replace(",", "{,}") +
                     f"{g['mean_realised_gain']:+.3f}{{\\tiny\\,[{g['mean_realised_gain_ci'][0]:+.3f},\\,{g['mean_realised_gain_ci'][1]:+.3f}]}} & "
                     f"{g['spearman']:+.3f}{{\\tiny\\,[{g['spearman_ci'][0]:+.3f},\\,{g['spearman_ci'][1]:+.3f}]}}\\\\")
    g = G["pooled"]
    lines.append("\\midrule")
    lines.append(f"Pooled & {g['n_pairs']:,} & -- & ".replace(",", "{,}") +
                 f"{g['spearman']:+.3f}{{\\tiny\\,[{g['spearman_ci'][0]:+.3f},\\,{g['spearman_ci'][1]:+.3f}]}}\\\\")
    (A.PAPER / "tables" / "supp_gain_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")
    # six-task table gets a label-free row
    A.general_table(gen)
    p = A.PAPER / "tables" / "router_general_body.tex"
    body = p.read_text().rstrip().split("\n")
    v = L["six_task"]["methods"]["healthrouter_labelfree"]
    row = ("HR, task voted$^\\ddagger$ & " +
           " & ".join(f"{F3(v['per_task'][t]['value'])}{{\\tiny\\,[{F3(v['per_task'][t]['ci'][0])},\\,{F3(v['per_task'][t]['ci'][1])}]}}" for t in ("eurosat", "dtd")) +
           f" & {F3(v['macro']['value'])}{{\\tiny\\,[{F3(v['macro']['ci'][0])},\\,{F3(v['macro']['ci'][1])}]}} & {gen['methods']['healthrouter']['invoked_mean']:.2f}\\\\")
    body.insert(len(body) - 1, row)
    p.write_text("\n".join(body) + "\n")


# ----------------------------------------------------------------------------- numbers quoted in the text
def numbers(main, desc):
    M = main["methods"]
    put = A.put
    S = desc["single_candidates"]
    n = {t: M["healthrouter"]["per_task"][t]["n"] for t in TASKS}
    put("n_insource", sum(n.values()), f"{sum(n.values()):,}".replace(",", "{,}"), "sum_t router_main.methods.healthrouter.per_task[t].n")
    put("n_collections_eleven", 231, "231", "sum_{k<=3} C(11,k) = 11+55+165")
    hr, bs, orc = (M[k]["macro"]["value"] for k in ("healthrouter", "best_single_per_task", "oracle_single"))
    share = 100 * (hr - bs) / (orc - bs)
    put("headroom_captured", share, f"{share:.0f}\\%", "100*(healthrouter.macro - best_single_per_task.macro)/(oracle_single.macro - best_single_per_task.macro)")
    for t in TASKS:
        put(f"oracle_{t}", M["oracle_single"]["per_task"][t]["value"], F3(M["oracle_single"]["per_task"][t]["value"]), f"methods.oracle_single.per_task.{t}")
        put(f"bs_{t}", M["best_single_per_task"]["per_task"][t]["value"], F3(M["best_single_per_task"]["per_task"][t]["value"]), f"methods.best_single_per_task.per_task.{t}")
    F = {t: dict(zip(ENC, M["healthrouter"]["selection_frequency"][t])) for t in TASKS}
    for t in TASKS:
        for e in ENC:
            put(f"sf_{t}_{e}", 100 * F[t][e], f"{100 * F[t][e]:.0f}\\%", f"100*router_main.methods.healthrouter.selection_frequency.{t}[{e}]")
    # N1
    for e in ENC:
        c = S["candidates"][e]
        put(f"single_{e}", c["macro"]["value"], F3(c["macro"]["value"]), f"router_descriptive.single_candidates.candidates.{e}.macro", c["macro"]["ci"])
        for t in TASKS:
            put(f"single_{e}_{t}", c["per_task"][t]["value"], F3(c["per_task"][t]["value"]), f"router_descriptive.single_candidates.candidates.{e}.per_task.{t}", c["per_task"][t]["ci"])
    put("n_distinct_best", S["n_distinct_best"], {4: "four"}.get(S["n_distinct_best"], str(S["n_distinct_best"])), "router_descriptive.single_candidates.n_distinct_best")
    for t, v in S["per_image_other_candidate_better"].items():
        put(f"other_better_{t}", 100 * v, f"{100 * v:.0f}\\%", f"100*router_descriptive.single_candidates.per_image_other_candidate_better.{t}")
    # N0a
    L = desc["label_free"]
    v = L["in_distribution"]
    v6 = L["six_task"]
    put("lf_mis_six", sum(v6["misinferred_images"].values()), str(sum(v6["misinferred_images"].values())), "sum router_descriptive.label_free.six_task.misinferred_images")
    ev6 = sum(v6["image_evaluations"].values())
    put("lf_eval_six", ev6, f"{ev6:,}".replace(",", "{,}"), "sum router_descriptive.label_free.six_task.image_evaluations")
    put("lf_mis_in", sum(v["misinferred_images"].values()), str(sum(v["misinferred_images"].values())), "sum router_descriptive.label_free.in_distribution.misinferred_images")
    ev = sum(v["image_evaluations"].values())
    put("lf_eval_in", ev, f"{ev:,}".replace(",", "{,}"), "sum router_descriptive.label_free.in_distribution.image_evaluations")
    for key in ("in_distribution", "six_task"):
        vv = L[key]
        put(f"lf_acc_{key}", 100 * vv["task_inference_accuracy"]["all"], f"{100 * vv['task_inference_accuracy']['all']:.1f}\\%",
            f"100*router_descriptive.label_free.{key}.task_inference_accuracy.all")
        for t, a in vv["task_inference_accuracy"].items():
            put(f"lf_acc_{key}_{t}", 100 * a, f"{100 * a:.1f}\\%", f"100*router_descriptive.label_free.{key}.task_inference_accuracy.{t}")
        for lf in ("healthrouter_labelfree", "harness_labelfree"):
            m = vv["methods"][lf]["macro"]
            put(f"lf_{key}_{lf}", m["value"], F3(m["value"]), f"router_descriptive.label_free.{key}.methods.{lf}.macro", m["ci"])
            d = vv["differences_vs_healthrouter"][lf]["macro"]
            put(f"lf_d_{key}_{lf}", d["value"], F3(d["value"]), f"router_descriptive.label_free.{key}.differences_vs_healthrouter.{lf}.macro", d["ci"])
    for c in A.CONDS:
        vv = L["external"][c]
        put(f"lf_acc_ext_{c}", 100 * vv["task_inference_accuracy"]["all"], f"{100 * vv['task_inference_accuracy']['all']:.1f}\\%",
            f"100*router_descriptive.label_free.external.{c}.task_inference_accuracy.all")
        put(f"lf_acc_ext_{c}_chest", 100 * vv["task_inference_accuracy"]["chest_xray"], f"{100 * vv['task_inference_accuracy']['chest_xray']:.0f}\\%",
            f"100*router_descriptive.label_free.external.{c}.task_inference_accuracy.chest_xray")
        m = vv["methods"]["healthrouter_labelfree"]["macro"]
        put(f"lf_ext_{c}", m["value"], F3(m["value"]), f"router_descriptive.label_free.external.{c}.methods.healthrouter_labelfree.macro", m["ci"])
    md = max(abs(L["external"][c]["methods"]["healthrouter_labelfree"]["diff_base_minus_this"]["value"]) for c in A.CONDS)
    put("lf_ext_maxdiff", md, F3(md), "max_c |router_descriptive.label_free.external[c].methods.healthrouter_labelfree.diff_base_minus_this|")
    lo = min(L["external"][c]["task_inference_accuracy"]["all"] for c in A.CONDS)
    put("lf_acc_ext_min", 100 * lo, f"{100 * lo:.1f}\\%", "100*min_c router_descriptive.label_free.external[c].task_inference_accuracy.all")
    # N2
    G = desc["gain_calibration"]
    for t in TASKS + ["pooled"]:
        rv = G[t]["spearman"]
        put(f"rho_{t}", rv, (f"$-${abs(rv):.3f}" if rv < 0 else f"{rv:.3f}"),
            f"router_descriptive.gain_calibration.{t}.spearman", G[t]["spearman_ci"])
        lo_, hi_ = G[t]["spearman_ci"]
        fm = lambda x: f"$-${abs(x):.3f}" if x < 0 else f"{x:.3f}"
        A.NUM[f"rho_{t}"]["ci_text"] = f"[{fm(lo_)}, {fm(hi_)}]"
        if t != "pooled":
            put(f"gain_{t}", G[t]["mean_realised_gain"], F3(G[t]["mean_realised_gain"]),
                f"router_descriptive.gain_calibration.{t}.mean_realised_gain", G[t]["mean_realised_gain_ci"])
    # N6
    Dd = desc["harness_difficulty"]
    for t in TASKS:
        for k in ("error_one_call", "error_multi_call"):
            put(f"{k}_{t}", 100 * Dd[t][k], f"{100 * Dd[t][k]:.1f}\\%", f"100*router_descriptive.harness_difficulty.{t}.{k}")
        put(f"rho_calls_{t}", Dd[t]["spearman_calls_logloss"], f"{Dd[t]['spearman_calls_logloss']:.2f}",
            f"router_descriptive.harness_difficulty.{t}.spearman_calls_logloss", Dd[t]["spearman_calls_logloss_ci"])


def close_tabular(names):
    """Main-paper tables with vertical rules: end the tabular inside the body file so that
    \\input does not leave a phantom row below \\bottomrule."""
    for nm in names:
        p = A.PAPER / "tables" / nm
        body = p.read_text().rstrip()
        if not body.endswith("\\end{tabular}"):
            p.write_text(body + "\n\\end{tabular}\n")


def run(main_res, ext, gen, desc):
    main_table(main_res)
    pool_table(main_res, desc)
    ablation_table(main_res, desc)
    if ext:
        external_table(ext, desc)
    supp_tables(main_res, gen, desc)
    teaser(main_res, desc)
    selection_figure(main_res, desc)
    gain_tasks_figure(desc)
    numbers(main_res, desc)

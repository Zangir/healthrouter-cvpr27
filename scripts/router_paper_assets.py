#!/usr/bin/env python3
"""Paper tables, derived numbers and figures for the HealthRouter method paper.

Reads only committed result files (results/router_*.json) and writes:
  results/router_paper_numbers.json  every derived number quoted in the text (with its formula)
  <paper>/numbers.tex                 LaTeX macros for those numbers
  <paper>/tables/router_*.tex         table bodies
  figures/router_*.pdf                teaser (accuracy-calls Pareto), selection analysis, robustness
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import style  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parents[1] / "paper"
TASKS = ["chest_xray", "pathology", "fundus", "mammography"]
BLUE, GREEN, ORANGE, GRAY, RED = "#0072B2", "#009E73", "#E69F00", "#8a8a8a", "#D55E00"

NAMES = {
    "best_single_overall": "Best single model (overall)",
    "best_single_per_task": "Best single per task (val.\\ ranked)",
    "best_single_per_task_oof": "Best single per task (OOF ranked)",
    "uniform_ensemble": "Uniform ensemble (all 10)",
    "stacking": "Stacking \\citep{wolpert1992stacked}",
    "greedy_ensemble_selection": "Greedy ens.\\ selection \\citep{caruana2004ensemble}",
    "logme_selection": "LogME top-$k$ \\citep{you2021logme}",
    "modality_router_supervised": "Modality router, supervised",
    "modality_router_supervised_oof": "Modality router, sup.\\ (OOF ranked)",
    "modality_router_zeroshot_biomedclip": "Modality router, zero-shot BiomedCLIP",
    "soft_moe": "Soft MoE gate \\citep{jacobs1991adaptive}",
    "sparse_topk_moe": "Sparse top-$k$ MoE \\citep{shazeer2017outrageously}",
    "knora_u": "KNORA-U \\citep{ko2008dynamic}",
    "knora_e": "KNORA-E \\citep{ko2008dynamic}",
    "meta_des": "META-DES \\citep{cruz2015meta}",
    "knn_router": "Neighbourhood router (single models)",
    "routerbench_mlp": "Predictive router \\citep{hu2024routerbench}",
    "routellm_mf": "Matrix-factorisation router \\citep{ong2024routellm}",
    "healthrouter": "\\textbf{HealthRouter (ours)}",
    "healthrouter_harness": "\\textbf{HealthRouter harness (ours)}",
    "oracle_single": "\\textcolor{gray}{Oracle best single per image}",
    "healthrouter_fallback": "HealthRouter + distance fallback$^\\dagger$",
    "harness_fallback": "Harness + distance fallback$^\\dagger$",
}
GROUPS = [
    ("Single models", ["best_single_overall", "best_single_per_task", "best_single_per_task_oof"]),
    ("Task-level ensembles and selection", ["uniform_ensemble", "stacking", "greedy_ensemble_selection", "logme_selection"]),
    ("Modality routers", ["modality_router_supervised", "modality_router_supervised_oof", "modality_router_zeroshot_biomedclip"]),
    ("Mixture-of-experts gates", ["soft_moe", "sparse_topk_moe"]),
    ("Dynamic ensemble selection", ["knora_u", "knora_e", "meta_des"]),
    ("Per-sample routers", ["knn_router", "routerbench_mlp", "routellm_mf"]),
    ("Ours", ["healthrouter", "healthrouter_harness"]),
]
SHORT = {"best_single_per_task": "Best single/task", "uniform_ensemble": "Uniform ens.", "stacking": "Stacking",
         "greedy_ensemble_selection": "Greedy ES", "soft_moe": "Soft MoE", "sparse_topk_moe": "Sparse MoE",
         "knora_u": "KNORA-U", "meta_des": "META-DES", "knora_e": "KNORA-E", "knn_router": "kNN router",
         "routerbench_mlp": "Predictive", "routellm_mf": "MF router", "knora_e": "KNORA-E", "logme_selection": "LogME",
         "modality_router_zeroshot_biomedclip": "Modality (zero-shot)", "modality_router_supervised": "Modality (sup.)",
         "best_single_overall": "Best single", "healthrouter": "HealthRouter", "healthrouter_harness": "Harness"}

NUM: dict[str, dict] = {}


def f3(x):
    return f"{x:.3f}"


def put(key, value, text, formula, ci=None):
    NUM[key] = {"value": value, "text": text, "formula": formula}
    if ci is not None:
        NUM[key]["ci"] = ci
        NUM[key]["ci_text"] = f"[{f3(ci[0])}, {f3(ci[1])}]"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def bold_best(values, exclude=()):
    keep = {k: v for k, v in values.items() if k not in exclude and v is not None}
    return max(keep, key=keep.get) if keep else None


def main_table(main):
    M = main["methods"]
    rows = [n for _, ns in GROUPS for n in ns]
    cols = TASKS + ["macro"]
    best = {}
    for c in cols:
        vals = {n: (M[n]["macro"]["value"] if c == "macro" else M[n]["per_task"][c]["value"]) for n in rows}
        top = max(vals.values())
        best[c] = {n for n, v in vals.items() if round(v, 3) == round(top, 3)}
    lines = []
    for g, ns in GROUPS:
        lines.append(f"\\textit{{{g}}} &&&&&&&\\\\")
        for n in ns:
            cells = []
            for c in cols:
                e = M[n]["macro"] if c == "macro" else M[n]["per_task"][c]
                v = f3(e["value"])
                v = f"\\textbf{{{v}}}" if n in best[c] else v
                cells.append(f"{v}{{\\tiny\\,[{f3(e['ci'][0])},\\,{f3(e['ci'][1])}]}}")
            cells.append(f"{M[n]['invoked_mean']:.2f}")
            cells.append(f"{M[n]['latency_ms_mean']:.0f}")
            lines.append(f"{NAMES[n]} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule" if g != "Ours" else "\\midrule")
    o = M["oracle_single"]
    cells = [f"\\textcolor{{gray}}{{{f3(o['per_task'][t]['value'])}}}" for t in TASKS] + \
            [f"\\textcolor{{gray}}{{{f3(o['macro']['value'])}}}", "\\textcolor{gray}{1.00}",
             f"\\textcolor{{gray}}{{{o['latency_ms_mean']:.0f}}}"]
    lines.append(f"{NAMES['oracle_single']} & " + " & ".join(cells) + "\\\\")
    lines.append("\\bottomrule")
    (PAPER / "tables" / "router_main_body.tex").write_text("\n".join(lines) + "\n")


def diff_table(main, harness):
    """Supplement: paired differences HealthRouter (and harness) minus every baseline."""
    D, H = main["differences_vs_healthrouter"], harness["differences_vs_healthrouter_harness"]
    lines = []
    for _, ns in GROUPS[:-1]:
        for n in ns:
            def cell(d):
                lo, hi = d["ci"]
                v = f"{d['value']:+.3f}{{\\tiny\\,[{lo:+.3f},\\,{hi:+.3f}]}}"
                return f"\\textbf{{{v}}}" if lo > 0 else v
            lines.append(f"{NAMES[n]} & " + " & ".join(cell(D[n][t]) for t in TASKS + ["macro"]) +
                         f" & {cell(H[n]['macro'])}\\\\")
    (PAPER / "tables" / "router_diff_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")


ABL = [("healthrouter", "HealthRouter (full)"),
       ("abl_size1", "Single models ($s{=}1$)"),
       ("abl_size2", "Up to two ($s{=}2$)"),
       ("abl_size3", "Up to three ($s{=}3$)"),
       ("abl_size4", "Up to four ($s{=}4$)"),
       ("abl_picks1", "Hard pick only"),
       ("abl_picks3", "Soft top-3 only"),
       ("abl_geometric_fusion", "Geometric fusion"),
       ("abl_feat_biomedclip", "Features: BiomedCLIP"),
       ("abl_feat_dinov2_large", "Features: DINOv2-L"),
       ("abl_feat_ijepa", "Features: I-JEPA"),
       ("abl_feat_task_id", "Task identity (oracle)"),
       ("hr_collection_mlp", "MLP gate, collections"),
       ("hr_candidate_mlp", "MLP gate, single models"),
       ("healthrouter_fallback", "Distance fallback$^\\dagger$"),
       ]


def ablation_table(main):
    M, D = main["methods"], main["differences_vs_healthrouter"]
    lines = []
    for n, label in ABL:
        if n not in M:
            continue
        m = M[n]
        if n == "healthrouter":
            d = "--"
        else:
            dd = D[n]["macro"]
            d = f"{-dd['value']:+.3f}{{\\tiny\\,[{-dd['ci'][1]:+.3f},\\,{-dd['ci'][0]:+.3f}]}}"
        lines.append(f"{label} & {f3(m['macro']['value'])} & {d} & {m['invoked_mean']:.2f} & {m['latency_ms_mean']:.0f}\\\\")
        if n in ("healthrouter", "abl_size4", "abl_geometric_fusion", "abl_feat_task_id", "hr_candidate_mlp"):
            lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines = lines[:-1]
    (PAPER / "tables" / "router_ablation_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")


EXT_ROWS = ["best_single_per_task", "uniform_ensemble", "stacking", "greedy_ensemble_selection",
            "modality_router_supervised", "modality_router_zeroshot_biomedclip", "soft_moe", "sparse_topk_moe",
            "knora_u", "meta_des", "knn_router", "routellm_mf", "healthrouter", "healthrouter_harness",
            "healthrouter_fallback", "harness_fallback"]
EXT_SHORT = {"healthrouter": "\\textbf{HealthRouter}", "healthrouter_harness": "\\textbf{Harness}",
             "healthrouter_fallback": "HealthRouter + fallback$^\\dagger$", "harness_fallback": "Harness + fallback$^\\dagger$",
             "modality_router_zeroshot_biomedclip": "Modality, zero-shot", "modality_router_supervised": "Modality, supervised",
             "best_single_per_task": "Best single per task", "uniform_ensemble": "Uniform ensemble",
             "greedy_ensemble_selection": "Greedy ens. selection", "knn_router": "Neighbourhood router",
             "routellm_mf": "MF router"}
GEN_SHORT = {"modality_router_supervised_oof": "Modality router", "logme_selection": "LogME top-$k$",
             "routerbench_mlp": "Predictive router", "sparse_topk_moe": "Sparse MoE", "soft_moe": "Soft MoE"}
CONDS = ["standard", "grayscale", "blur", "low_resolution", "jpeg"]


def external_table(ext):
    R = ext["results"]
    rows = [n for n in EXT_ROWS if all(n in R[c] for c in CONDS)]
    best = {c: max(R[c][n]["macro"]["value"] for n in rows if "fallback" not in n) for c in CONDS}
    lines = []
    for n in rows:
        cells = []
        for c in CONDS:
            v = R[c][n]["macro"]["value"]
            s = f3(v)
            cells.append(f"\\textbf{{{s}}}" if round(v, 3) == round(best[c], 3) else s)
        nm = EXT_SHORT.get(n, SHORT.get(n, n))
        lines.append(f"{nm} & " + " & ".join(cells) + f" & {np.mean([R[c][n]['macro']['value'] for c in CONDS]):.3f}\\\\")
        if n in ("routellm_mf", "healthrouter_harness"):
            lines.append("\\midrule")
    (PAPER / "tables" / "router_external_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")


def external_pertask_table(ext):
    R = ext["results"]
    rows = ["healthrouter", "healthrouter_harness", "healthrouter_fallback", "harness_fallback", "uniform_ensemble",
            "stacking", "greedy_ensemble_selection", "best_single_per_task", "soft_moe", "sparse_topk_moe"]
    cname = {"standard": "Clean", "grayscale": "Grayscale", "blur": "Blur", "low_resolution": "Low resolution", "jpeg": "JPEG"}
    lines = []
    for c in CONDS:
        lines.append(f"\\textit{{{cname[c]}}} &&&&&\\\\")
        for n in rows:
            if n not in R[c]:
                continue
            m = R[c][n]
            cells = [f"{f3(m['per_task'][t]['value'])}{{\\tiny\\,[{f3(m['per_task'][t]['ci'][0])},\\,{f3(m['per_task'][t]['ci'][1])}]}}" for t in TASKS]
            nm = EXT_SHORT.get(n, SHORT.get(n, n))
            lines.append(f"{nm} & " + " & ".join(cells) + f" & {f3(m['macro']['value'])}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    (PAPER / "tables" / "router_external_pertask_body.tex").write_text("\n".join(lines) + "\n")


def general_table(gen):
    M = gen["methods"]
    tasks = gen["tasks"]
    rows = [n for n in ["best_single_per_task", "uniform_ensemble", "stacking", "greedy_ensemble_selection",
                        "logme_selection", "modality_router_supervised_oof", "soft_moe", "sparse_topk_moe", "knora_u",
                        "meta_des", "knn_router", "routerbench_mlp", "routellm_mf", "healthrouter", "healthrouter_harness"] if n in M]
    cols = ["eurosat", "dtd", "macro"]
    def val(n, c):
        return M[n]["macro"]["value"] if c == "macro" else M[n]["per_task"][c]["value"]
    best = {c: max(val(n, c) for n in rows) for c in cols}
    lines = []
    for n in rows:
        cells = []
        for c in cols:
            e = M[n]["macro"] if c == "macro" else M[n]["per_task"][c]
            s = f3(e["value"])
            s = f"\\textbf{{{s}}}" if round(e["value"], 3) == round(best[c], 3) else s
            cells.append(f"{s}{{\\tiny\\,[{f3(e['ci'][0])},\\,{f3(e['ci'][1])}]}}")
        nm = GEN_SHORT.get(n, EXT_SHORT.get(n, SHORT.get(n, n)))
        lines.append(f"{nm} & " + " & ".join(cells) + f" & {M[n]['invoked_mean']:.2f}\\\\")
        if n == "routellm_mf":
            lines.append("\\midrule")
    (PAPER / "tables" / "router_general_body.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n")


def teaser(main):
    style()
    M = main["methods"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    base = [n for _, ns in GROUPS[:-1] for n in ns]
    for n in base:
        x, y = M[n]["invoked_mean"], M[n]["macro"]["value"]
        ax.scatter(x, y, s=28, color=GRAY, zorder=3)
    lab = {"stacking": (6, -3), "uniform_ensemble": (6, -9), "soft_moe": (-48, 4), "knora_u": (6, -2),
           "meta_des": (-46, 2), "best_single_per_task": (6, -6), "greedy_ensemble_selection": (5, 3),
           "sparse_topk_moe": (6, -9), "modality_router_supervised_oof": (6, -9), "knora_e": (6, -3),
           "best_single_overall": (6, -3), "knn_router": (-50, -10)}
    names = {"modality_router_supervised_oof": "Modality router", "best_single_overall": "Best single (overall)"}
    for n, off in lab.items():
        ax.annotate(names.get(n, SHORT.get(n, n)), (M[n]["invoked_mean"], M[n]["macro"]["value"]), textcoords="offset points",
                    xytext=off, fontsize=8, color="#555555")
    th = sorted([(M[k]["invoked_mean"], M[k]["macro"]["value"]) for k in M if k.startswith("harness_theta")])
    if th:
        ax.plot(*zip(*th), "-o", color=GREEN, ms=4, lw=1.6, zorder=4)
        ax.annotate("harness, sweeping $\\theta$", th[0], textcoords="offset points", xytext=(4, -14), fontsize=8.5, color=GREEN)
    hx, hy = M["healthrouter_harness"]["invoked_mean"], M["healthrouter_harness"]["macro"]["value"]
    ax.scatter(hx, hy, s=70, marker="D", color=GREEN, zorder=5, edgecolor="white")
    ax.annotate("Harness", (hx, hy), textcoords="offset points", xytext=(-14, 8), fontsize=9.5, color=GREEN, fontweight="bold")
    x, y = M["healthrouter"]["invoked_mean"], M["healthrouter"]["macro"]["value"]
    ax.scatter(x, y, s=150, marker="*", color=BLUE, zorder=6, edgecolor="white")
    ax.annotate("HealthRouter", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=10, color=BLUE, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xticklabels(["1", "2", "3", "5", "10"])
    ax.set_xlabel("models called per image (mean)")
    ax.set_ylabel("macro score, 4 medical tasks")
    lo = min(M[n]["macro"]["value"] for n in base)
    ax.set_ylim(lo - 0.006, y + 0.008)
    fig.savefig(FIG / "router_teaser.pdf")
    fig.savefig(FIG / "router_teaser.png", dpi=200)
    plt.close(fig)


def selection_figure(main):
    style()
    M = main["methods"]
    cands = [c.split("@")[0] for c in main["candidates"]]
    pretty = {"imagenet_convnext": "ConvNeXt-T", "dinov2_base": "DINOv2-B", "dinov2_large": "DINOv2-L",
              "siglip2_base": "SigLIP2-B", "biomedclip": "BiomedCLIP", "rad_dino": "RAD-DINO", "phikon": "Phikon",
              "phikon_v2": "Phikon-v2", "retfound": "RETFound", "mammo_clip": "Mammo-CLIP"}
    tname = {"chest_xray": "Chest X-ray", "pathology": "Pathology", "fundus": "Fundus", "mammography": "Mammography"}
    F = np.array([M["healthrouter"]["selection_frequency"][t] for t in TASKS])
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    im = ax.imshow(F, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels([pretty[c] for c in cands], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([tname[t] for t in TASKS])
    ax.grid(False)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if F[i, j] >= 0.05:
                ax.text(j, i, f"{F[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if F[i, j] > 0.55 else "black")
    ax.set_title("(a) share of test images on which HealthRouter runs each model", fontsize=10)
    ax = axes[1]
    H = M["healthrouter_harness"].get("invoked_hist", {t: [0] * 10 for t in TASKS})
    width = 0.2
    for i, t in enumerate(TASKS):
        h = np.array(H[t][:5])
        ax.bar(np.arange(1, 6) + (i - 1.5) * width, h, width, label=tname[t],
               color=[BLUE, ORANGE, GREEN, RED][i])
    ax.set_xlabel("models called by the harness")
    ax.set_ylabel("share of images")
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=7.5)
    ax.set_title("(b) harness calls per image", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "router_selection.pdf")
    fig.savefig(FIG / "router_selection.png", dpi=200)
    plt.close(fig)


def numbers(main, harness, ext, gen):
    M, D = main["methods"], main["differences_vs_healthrouter"]
    H = harness["differences_vs_healthrouter_harness"]
    hr = M["healthrouter"]
    put("hr_macro", hr["macro"]["value"], f3(hr["macro"]["value"]), "methods.healthrouter.macro.value", hr["macro"]["ci"])
    for k, n in [("d_soft_moe", "soft_moe"), ("d_sparse_moe", "sparse_topk_moe"), ("d_knora_u", "knora_u"),
                 ("d_meta_des", "meta_des"), ("d_knora_e", "knora_e"), ("d_stacking", "stacking"),
                 ("d_knn", "knn_router"), ("d_uniform", "uniform_ensemble"), ("d_greedy", "greedy_ensemble_selection"),
                 ("d_modality", "modality_router_supervised"), ("d_modality_oof", "modality_router_supervised_oof"),
                 ("d_routellm", "routellm_mf"), ("d_routerbench", "routerbench_mlp"), ("d_logme", "logme_selection"),
                 ("bs_task_gap", "best_single_per_task"), ("d_best_overall", "best_single_overall"),
                 ("d_harness", "healthrouter_harness")]:
        put(k, D[n]["macro"]["value"], f3(D[n]["macro"]["value"]), f"differences_vs_healthrouter.{n}.macro", D[n]["macro"]["ci"])
    for k, n in [("h_uniform", "uniform_ensemble"), ("h_stacking", "stacking"), ("h_soft_moe", "soft_moe"),
                 ("h_greedy", "greedy_ensemble_selection")]:
        put(k, H[n]["macro"]["value"], f3(H[n]["macro"]["value"]), f"differences_vs_healthrouter_harness.{n}.macro", H[n]["macro"]["ci"])
    hm = M["healthrouter_harness"]
    put("harness_macro", hm["macro"]["value"], f3(hm["macro"]["value"]), "methods.healthrouter_harness.macro", hm["macro"]["ci"])
    put("harness_calls", hm["invoked_mean"], f"{hm['invoked_mean']:.2f}", "methods.healthrouter_harness.invoked_mean")
    put("hr_calls", hr["invoked_mean"], f"{hr['invoked_mean']:.2f}", "methods.healthrouter.invoked_mean")
    fewer = 100 * (1 - hm["invoked_mean"] / M["uniform_ensemble"]["invoked_mean"])
    put("harness_fewer_calls", fewer, f"{fewer:.0f}\\%", "100*(1 - harness.invoked_mean / uniform_ensemble.invoked_mean)")
    less = 100 * (1 - hm["latency_ms_mean"] / M["uniform_ensemble"]["latency_ms_mean"])
    put("harness_less_time", less, f"{less:.1f}\\%", "100*(1 - harness.latency_ms_mean / uniform_ensemble.latency_ms_mean)")
    put("harness_ms", hm["latency_ms_mean"], f"{hm['latency_ms_mean']:.1f}", "methods.healthrouter_harness.latency_ms_mean")
    put("uniform_ms", M["uniform_ensemble"]["latency_ms_mean"], f"{M['uniform_ensemble']['latency_ms_mean']:.1f}", "methods.uniform_ensemble.latency_ms_mean")
    put("hr_ms", hr["latency_ms_mean"], f"{hr['latency_ms_mean']:.1f}", "methods.healthrouter.latency_ms_mean")
    put("router_ms", main["candidate_cost_ms"][0], f"{main['candidate_cost_ms'][0]:.1f}", "candidate_cost_ms[imagenet_convnext]")
    put("costmin", min(main["candidate_cost_ms"]), f"{min(main['candidate_cost_ms']):.1f}", "min(candidate_cost_ms)")
    put("costmax", max(main["candidate_cost_ms"]), f"{max(main['candidate_cost_ms']):.1f}", "max(candidate_cost_ms)")
    one = np.mean([hm.get("invoked_hist", {t: [0] for t in TASKS})[t][0] for t in TASKS]) * 100
    put("harness_one_model", one, f"{one:.0f}\\%", "100*mean_t harness.invoked_hist[t][k=1]")
    put("oracle_macro", M["oracle_single"]["macro"]["value"], f3(M["oracle_single"]["macro"]["value"]), "methods.oracle_single.macro")
    for n in M:
        if n.startswith(("abl_", "harness_theta", "cost_lam", "hr_c", "healthrouter_fallback", "harness_fallback")):
            put(f"m_{n}", M[n]["macro"]["value"], f3(M[n]["macro"]["value"]), f"methods.{n}.macro", M[n]["macro"]["ci"])
            put(f"calls_{n}", M[n]["invoked_mean"], f"{M[n]['invoked_mean']:.2f}", f"methods.{n}.invoked_mean")
            put(f"ms_{n}", M[n]["latency_ms_mean"], f"{M[n]['latency_ms_mean']:.0f}", f"methods.{n}.latency_ms_mean")
            if n in D:
                put(f"d_{n}", D[n]["macro"]["value"], f3(D[n]["macro"]["value"]), f"differences_vs_healthrouter.{n}.macro", D[n]["macro"]["ci"])
                put(f"a_d_{n}", abs(D[n]["macro"]["value"]), f3(abs(D[n]["macro"]["value"])), f"abs(differences_vs_healthrouter.{n}.macro)")
    for t in TASKS:
        put(f"harness_one_{t}", 100 * hm["invoked_hist"][t][0], f"{100 * hm['invoked_hist'][t][0]:.0f}\\%", f"100*harness.invoked_hist.{t}[k=1]")
    put("bs_task_macro", M["best_single_per_task"]["macro"]["value"], f3(M["best_single_per_task"]["macro"]["value"]), "methods.best_single_per_task.macro")
    put("n_collections", 175, "175", "sum_{k<=3} C(10,k) = 10+45+120")
    for t in TASKS:
        put(f"hr_{t}", hr["per_task"][t]["value"], f3(hr["per_task"][t]["value"]), f"healthrouter.per_task.{t}", hr["per_task"][t]["ci"])
        for n in ["greedy_ensemble_selection", "stacking", "uniform_ensemble", "soft_moe", "healthrouter_harness",
                  "modality_router_supervised_oof"]:
            put(f"d_{n}_{t}", D[n][t]["value"], f3(D[n][t]["value"]), f"differences_vs_healthrouter.{n}.{t}", D[n][t]["ci"])
            put(f"a_d_{n}_{t}", abs(D[n][t]["value"]), f3(abs(D[n][t]["value"])), f"abs(differences_vs_healthrouter.{n}.{t})")
    if ext:
        R = ext["results"]
        n_ext = sum(ext["n"].values())
        put("n_external", n_ext, f"{n_ext:,}".replace(",", "{,}"), "sum of router_external.n over the four cohorts")
        twenty = [n for _, ns in GROUPS for n in ns]
        for c in CONDS:
            order = sorted([n for n in twenty if n in R[c]], key=lambda n: -R[c][n]["macro"]["value"])
            put(f"ext_n_methods_{c}", len(order), str(len(order)), f"number of the 20 methods present in router_external.results.{c}")
            put(f"ext_rank_hr_{c}", order.index("healthrouter") + 1, str(order.index("healthrouter") + 1),
                f"rank of healthrouter among the {len(order)} methods in router_external.results.{c}")
            if "healthrouter_harness" in R[c]:
                put(f"ext_rank_harness_{c}", order.index("healthrouter_harness") + 1, str(order.index("healthrouter_harness") + 1),
                    f"rank of the harness among the {len(order)} methods in router_external.results.{c}")
            put(f"ext_hr_fundus_{c}", R[c]["healthrouter"]["per_task"]["fundus"]["value"],
                f3(R[c]["healthrouter"]["per_task"]["fundus"]["value"]), f"router_external.results.{c}.healthrouter.per_task.fundus")
            for n in ("healthrouter_harness", "healthrouter_fallback", "harness_fallback", "uniform_ensemble", "stacking", "soft_moe", "sparse_topk_moe", "best_single_per_task"):
                if n in R[c]:
                    put(f"ext_{n}_{c}", R[c][n]["macro"]["value"], f3(R[c][n]["macro"]["value"]), f"router_external.results.{c}.{n}.macro", R[c][n]["macro"]["ci"])
            hrv = R[c]["healthrouter"]["macro"]["value"]
            put(f"ext_hr_{c}", hrv, f3(hrv), f"router_external.results.{c}.healthrouter.macro")
            for n in ["soft_moe", "sparse_topk_moe", "stacking", "uniform_ensemble", "modality_router_zeroshot_biomedclip", "knora_u",
                      "healthrouter_harness", "modality_router_supervised", "best_single_per_task", "greedy_ensemble_selection",
                      "modality_router_supervised_oof", "healthrouter_fallback", "harness_fallback"]:
                if n in R[c]:
                    d = R[c][n].get("diff_healthrouter_minus_this")
                    if d:
                        put(f"ext_d_{n}_{c}", d["value"], f3(d["value"]), f"router_external.results.{c}.{n}.diff", d["ci"])
                        put(f"a_ext_d_{n}_{c}", abs(d["value"]), f3(abs(d["value"])), f"abs(router_external.results.{c}.{n}.diff)")
    flag = load("router_ood_flag.json")
    if flag:
        for c, rates in flag["flag_rate"].items():
            for t, v in rates.items():
                put(f"flag_{c}_{t}", 100 * v, f"{100 * v:.0f}\\%", f"100*router_ood_flag.flag_rate.{c}.{t}")
    if gen:
        G = gen["methods"]
        for n in G:
            put(f"gen_{n}", G[n]["macro"]["value"], f3(G[n]["macro"]["value"]), f"router_general.methods.{n}.macro", G[n]["macro"]["ci"])
        for n, d in gen.get("differences_vs_healthrouter", {}).items():
            put(f"gen_d_{n}", d["macro"]["value"], f3(d["macro"]["value"]), f"router_general.differences.{n}.macro", d["macro"]["ci"])
            put(f"a_gen_d_{n}", abs(d["macro"]["value"]), f3(abs(d["macro"]["value"])), f"abs(router_general.differences.{n}.macro)")
            for t in ("eurosat", "dtd"):
                put(f"gen_d_{n}_{t}", d[t]["value"], f3(d[t]["value"]), f"router_general.differences.{n}.{t}", d[t]["ci"])
                put(f"a_gen_d_{n}_{t}", abs(d[t]["value"]), f3(abs(d[t]["value"])), f"abs(router_general.differences.{n}.{t})")
        for n in G:
            for t in ("eurosat", "dtd"):
                put(f"gen_{n}_{t}", G[n]["per_task"][t]["value"], f3(G[n]["per_task"][t]["value"]), f"router_general.methods.{n}.per_task.{t}", G[n]["per_task"][t]["ci"])
        twenty = [n for _, ns in GROUPS for n in ns if n in G]
        order = sorted(twenty, key=lambda n: -G[n]["macro"]["value"])
        put("gen_rank_hr", order.index("healthrouter") + 1, str(order.index("healthrouter") + 1), "rank of healthrouter by 6-task macro in router_general")


def write_numbers():
    (RES / "router_paper_numbers.json").write_text(json.dumps(NUM, indent=1) + "\n")
    out = ["% Generated by experiments/repo/scripts/router_paper_assets.py from results/router_*.json. Do not edit."]
    for k, v in NUM.items():
        out.append(f"\\expandafter\\def\\csname hrnum@{k}\\endcsname{{{v['text']}}}")
        if "ci_text" in v:
            out.append(f"\\expandafter\\def\\csname hrci@{k}\\endcsname{{{v['ci_text']}}}")
    (PAPER / "numbers.tex").write_text("\n".join(out) + "\n")


def robustness_figure(ext):
    style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    R = ext["results"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    # family colours as in the main-paper figures (router_exemplar_assets.FAMILY)
    show = [("healthrouter", BLUE, "-o", "HealthRouter", 2.0), ("soft_moe", RED, "-^", "Soft MoE", 1.1),
            ("stacking", ORANGE, "-s", "Stacking", 1.1), ("uniform_ensemble", ORANGE, "--s", "Uniform ens.", 1.1),
            ("modality_router_zeroshot_biomedclip", "#C9B800", "-v", "Modality (zero-shot)", 1.1),
            ("knora_u", "#CC79A7", "-P", "KNORA-U", 1.1)]
    lab = ["clean", "gray", "blur", "low-res", "JPEG"]
    for n, c, st, l, lw in show:
        ax.plot(range(5), [R[k][n]["macro"]["value"] for k in CONDS], st, color=c, lw=lw, ms=3.5, label=l)
    ax.set_xticks(range(5))
    ax.set_xticklabels(lab)
    ax.set_ylabel("external macro score")
    ax.legend(fontsize=7.4, ncol=2, columnspacing=0.6, handlelength=1.6, labelspacing=0.25)
    fig.savefig(FIG / "router_robustness.pdf")
    fig.savefig(FIG / "router_robustness.png", dpi=300)
    plt.close(fig)
    plt.rcdefaults()


def main():
    import router_exemplar_assets as X
    X.A = sys.modules[__name__]  # share NUM/PAPER with the running module
    (PAPER / "tables").mkdir(exist_ok=True)
    main_res, harness = load("router_main.json"), load("router_main_harness.json")
    ext, gen = load("router_external.json"), load("router_general.json")
    desc = load("router_descriptive.json")
    diff_table(main_res, harness)
    if ext:
        external_pertask_table(ext)
        robustness_figure(ext)
    numbers(main_res, harness, ext, gen)
    X.run(main_res, ext, gen, desc)
    write_numbers()
    import shutil
    (PAPER / "figures").mkdir(exist_ok=True)
    for f in FIG.glob("router_*.pdf"):
        shutil.copy2(f, PAPER / "figures" / f.name)
    print("assets written:", len(NUM), "numbers")


if __name__ == "__main__":
    main()

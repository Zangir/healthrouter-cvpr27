"""Cost of committing to one frozen encoder for every modality (from matched_grid.json).

For each package and resolution: the gap to the per-domain best in each domain, its
worst-domain gap and its mean gap. A router that picks the per-domain best has zero gap
by construction, so the minimum worst-domain gap is what a universal encoder forgoes.
"""
from __future__ import annotations

import json
from pathlib import Path


def evaluate(results_dir: str) -> dict:
    grid = json.loads((Path(results_dir) / "matched_grid.json").read_text())
    out = {}
    for size, block in grid["sizes"].items():
        domains = block["domains"]
        best = {d: max(v["oof_score"] for v in b["packages"].values()) for d, b in domains.items()}
        rows = {}
        for p in next(iter(domains.values()))["packages"]:
            gaps = {d: round(best[d] - domains[d]["packages"][p]["oof_score"], 5) for d in domains}
            rows[p] = {"gaps": gaps, "worst_gap": max(gaps.values()),
                       "mean_gap": round(sum(gaps.values()) / len(gaps), 5)}
        universal = min(rows, key=lambda p: rows[p]["worst_gap"])
        out[size] = {"packages": rows, "best_single_package": universal,
                     "best_single_worst_gap": rows[universal]["worst_gap"],
                     "best_single_mean_gap": rows[universal]["mean_gap"]}
    return {"provenance": "derived from results/matched_grid.json", "sizes": out}


if __name__ == "__main__":
    print(json.dumps(evaluate("results"), indent=1)[:3000])

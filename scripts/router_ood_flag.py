#!/usr/bin/env python3
"""Out-of-scope flag diagnostic (plan amendment 2): share of external images whose router-feature
distance to their K-th nearest in-source meta-training neighbour exceeds the 95th percentile of
the same distance on the in-source validation split. Writes results/router_ood_flag.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_external import CONDITIONS  # noqa: E402

from healthrouter.collection import candidates, data, router  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = 50


def kth_distance(A, B, k):
    sims = B @ A.T
    return 1.0 - np.sort(sims, 1)[:, -k]


def main():
    tasks = data.MED_TASKS
    d = {t: candidates.external(t, CONDITIONS) for t in tasks}
    Ztr = np.concatenate([d[t]["Z_tr"] for t in tasks])
    std = router.Standardizer().fit(Ztr)
    norm = lambda z: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(std(z))
    A = norm(Ztr)
    val = np.concatenate([kth_distance(A, norm(d[t]["Z_va"]), K) for t in tasks])
    thr = float(np.percentile(val, 95))
    out = {"k": K, "threshold_quantile": 0.95, "threshold": round(thr, 5),
           "provenance": "scripts/router_ood_flag.py", "flag_rate": {}}
    for cond in CONDITIONS:
        rates = {t: round(float((kth_distance(A, norm(d[t][f"Z_ext_{cond}"]), K) > thr).mean()), 4) for t in tasks}
        rates["all"] = round(float(np.mean(np.concatenate(
            [kth_distance(A, norm(d[t][f"Z_ext_{cond}"]), K) > thr for t in tasks]))), 4)
        out["flag_rate"][cond] = rates
        print(cond, rates)
    (ROOT / "results" / "router_ood_flag.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

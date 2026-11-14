"""Validation-only quick comparison of task-level variants (seed 0)."""
import numpy as np
from router_experiment import load_run
from healthrouter.collection import data, methods, metrics
res = {}
for fold in range(5):
    run = load_run(0, fold, data.MED_TASKS)
    for name, fam in (("lane_oof", methods.modality_router_supervised(run, ["va"], source="tr")),
                      ("single_oof", methods.best_single_per_task(run, ["va"], source="tr")),
                      ("single_va", methods.best_single_per_task(run, ["va"])),
                      ("uniform", methods.uniform(run, ["va"])), ("stack", methods.stacking(run, ["va"])),
                      ("greedy", methods.greedy_ensemble(run, ["va"]))):
        for cfg, out in fam:
            res.setdefault((name, str(cfg)), []).append([metrics.primary(t, run[t]["y_va"], out["va"][t][0]) for t in run["tasks"]])
for k, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"{np.mean(v):.4f} {np.round(np.mean(v,0),3)} {k}")

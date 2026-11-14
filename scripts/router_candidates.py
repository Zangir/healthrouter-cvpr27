#!/usr/bin/env python3
"""Build (and cache) candidate predictions for the frozen cross-fitting splits."""
import argparse
import time

from healthrouter.collection import candidates, data

p = argparse.ArgumentParser()
p.add_argument("--tasks", nargs="+", default=data.MED_TASKS)
p.add_argument("--seeds", nargs="+", type=int, default=list(data.SPLIT_SEEDS))
p.add_argument("--sizes", nargs="+", type=int, default=[224])
a = p.parse_args()
for seed in a.seeds:
    for task in a.tasks:
        for fold in range(data.N_OUTER):
            t = time.time()
            out = candidates.outer_fold(task, seed, fold, tuple(a.sizes))
            print(task, seed, fold, out["P_tr"].shape, f"{time.time() - t:.1f}s", flush=True)

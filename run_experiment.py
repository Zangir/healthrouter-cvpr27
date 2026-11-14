#!/usr/bin/env python3
"""HealthRouter: one entry point for smoke tests and all paper experiments."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def write_result(out: str, name: str, result: dict) -> Path:
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{name}.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {target}")
    return target


def smoke(seed: int) -> dict:
    from healthrouter.metrics import bootstrap_accuracy, rounded_ci
    labels = np.arange(10) % 5
    predictions = labels.copy()
    return {"provenance": "procedural smoke only", "seed": seed, "n_samples": 10,
            "routing_accuracy": rounded_ci(bootstrap_accuracy(labels, predictions, 100), 5),
            "smoke_pass": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["router", "tb_probes", "encoder_grid", "cascade", "specialists",
                                 "grid_features", "matched_grid", "dispatch", "abstention",
                                 "finetune", "chest_resolution", "openset", "regret", "aggregate"],
                        default="router")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", default=os.environ.get("HEALTHROUTER_DATA_ROOT"))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    set_seed(args.seed)
    if args.smoke:
        write_result(args.out, "smoke", smoke(args.seed))
        return
    if args.method not in ("aggregate", "abstention", "regret") and not args.data_root:
        parser.error("--data-root is required outside smoke/aggregate mode")
    if args.method == "router":
        from healthrouter.routing import evaluate
        result = evaluate(args.data_root, args.seed)
    elif args.method == "tb_probes":
        from healthrouter.probes import evaluate
        result = evaluate(args.data_root)
    elif args.method == "encoder_grid":
        from healthrouter.grid import evaluate
        result = evaluate(args.data_root)
    elif args.method == "cascade":
        from healthrouter.cascade import evaluate
        result = evaluate(args.data_root)
    elif args.method == "grid_features":
        from healthrouter.grid2 import features
        features(args.data_root)
        return
    elif args.method == "matched_grid":
        from healthrouter.grid2_stats import evaluate
        result = evaluate(args.data_root)
    elif args.method == "dispatch":
        from healthrouter.dispatch import evaluate
        result = evaluate(args.data_root)
    elif args.method == "abstention":
        from healthrouter.abstain import evaluate
        result = evaluate()
    elif args.method == "chest_resolution":
        from healthrouter.chest_resolution import evaluate
        result = evaluate(args.data_root)
    elif args.method == "regret":
        from healthrouter.regret import evaluate
        result = evaluate(args.out)
        write_result(args.out, "universal_regret", result)
        return
    elif args.method == "openset":
        from healthrouter.openset import evaluate
        abstention = json.loads((Path(args.out) / "abstention.json").read_text())
        result = evaluate(args.data_root, abstention)
    elif args.method == "finetune":
        from healthrouter.finetune import evaluate
        result = evaluate(args.data_root)
    elif args.method == "specialists":
        from healthrouter.specialists import evaluate
        result = evaluate(args.data_root)
    else:
        from healthrouter.aggregate import ablations, aggregate
        result = aggregate(args.out)
        write_result(args.out, "ablations", ablations(args.out))
    write_result(args.out, args.method if args.method != "aggregate" else "main_results", result)


if __name__ == "__main__":
    main()

"""Five-lane executed dispatch on external cohorts, for four gates and five conditions.

A misrouted or abstained item receives a null output (score 0.5, grade 0, empty mask).
Retained skill = (U - U_null) / (U_ref - U_null) per lane, where U_ref is reference
routing and U_null serves nothing; it is unitless, so lanes can be macro-averaged.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import roc_auc_score

from . import external, gates, lanes, pools
from .data import router_sources
from .metrics import bootstrap_auc, bootstrap_qwk, mean_interval, qwk, rounded_ci

NULL = {"score": 0.5, "grade": 0, "mask": 0.0}


def lane_utility(lane: str, y, output, served) -> float:
    kind = lanes.SPECIALISTS[lane][3]
    if kind == "mask":
        return float(np.where(served, output, 0.0).mean())
    if kind == "grade":
        return float(qwk(y, np.where(served, output.argmax(1), NULL["grade"])))
    return float(roc_auc_score(y, np.where(served, output, NULL["score"])))


def utility_block(src: dict, outputs: dict, route: np.ndarray, conf: np.ndarray,
                  tau: float = 0.0, mask: np.ndarray | None = None) -> dict:
    """Per-lane utility and retained skill for one gate decision vector."""
    lane_ids = np.concatenate([[i] * len(src[l]["y"]) for i, l in enumerate(external.LANES)])
    keep = np.ones(len(lane_ids), bool) if mask is None else mask
    block, skills = {}, []
    for i, lane in enumerate(external.LANES):
        sel = (lane_ids == i) & keep
        local = sel[lane_ids == i]
        y, out = src[lane]["y"][local], outputs[lane][local]
        served = (route[sel] == i) & (conf[sel] >= tau)
        u = lane_utility(lane, y, out, served)
        ref = lane_utility(lane, y, out, np.ones(len(y), bool))
        null = lane_utility(lane, y, out, np.zeros(len(y), bool))
        skill = (u - null) / (ref - null) if ref != null else float("nan")
        skills.append(skill)
        block[lane] = {"served": round(float(served.mean()), 5), "utility": round(u, 5),
                       "reference": round(ref, 5), "retained_skill": round(float(skill), 5)}
    covered = conf[keep] >= tau
    correct = route[keep] == lane_ids[keep]
    block["macro_retained_skill"] = round(float(np.mean(skills)), 5)
    block["coverage"] = round(float(covered.mean()), 5)
    block["route_accuracy"] = round(float(correct.mean()), 5)
    block["selective_route_accuracy"] = (round(float(correct[covered].mean()), 5)
                                         if covered.any() else None)
    return block


def reference_metric(lane: str, y, output) -> dict:
    kind = lanes.SPECIALISTS[lane][3]
    if kind == "mask":
        return rounded_ci(mean_interval(output, 2_000), 5)
    if kind == "grade":
        return rounded_ci(bootstrap_qwk(y, output.argmax(1), 2_000), 5)
    return rounded_ci(bootstrap_auc(y, output, 2_000), 5)


def gate_scores(root: str, src: dict) -> dict:
    """{gate: {condition: (n, 5) probabilities}} for external and in-source items."""
    items = [x for lane in external.LANES for x in src[lane]["items"]]
    ext = gates.condition_arrays("external5", items)
    tr_files, tr_y, te_files, te_y, _ = router_sources(root)
    ins = gates.condition_arrays("insource_test", te_files)
    train = gates.condition_arrays("insource_train", tr_files, ["standard", "grayscale"])
    out = {"supervised": {}, "probe": {}, "probe_grayaug": {}}
    feats_train = {c: gates.imagenet_features(a) for c, a in train.items()}
    probe = gates.probe_gate(feats_train["standard"], tr_y)
    aug = gates.probe_gate(np.vstack([feats_train["standard"], feats_train["grayscale"]]),
                           np.concatenate([tr_y, tr_y]))
    for split, arrays in [("external", ext), ("insource", ins)]:
        for c, pixels in arrays.items():
            out["supervised"][f"{split}:{c}"] = gates.supervised_gate(root, pixels)
            f = gates.imagenet_features(pixels)
            out["probe"][f"{split}:{c}"] = probe.predict_proba(f)
            out["probe_grayaug"][f"{split}:{c}"] = aug.predict_proba(f)
        print(f"  gates done for {split}", flush=True)
    zs = gates.zero_shot_gate({f"external:{c}": a for c, a in ext.items()} |
                              {f"insource:{c}": a for c, a in ins.items()})
    out["zero_shot_biomedclip"] = zs
    return out, te_y


def evaluate(root: str) -> dict:
    start = time.time()
    src = external.sources(root)
    outputs = lanes.lane_outputs(root, src)
    scores, insource_y = gate_scores(root, src)
    lane_ids = np.concatenate([[i] * len(src[l]["y"]) for i, l in enumerate(external.LANES)])
    np.savez(pools.cache_dir() / "dispatch_items.npz", lane_ids=lane_ids, insource_y=insource_y,
             groups=np.concatenate([src[l]["groups"] for l in external.LANES]),
             **{f"y_{l}": src[l]["y"] for l in external.LANES},
             **{f"out_{l}": outputs[l] for l in external.LANES},
             **{f"{g}|{k}": v for g, d in scores.items() for k, v in d.items()})
    result = {"gates": {}}
    for gate, by_key in scores.items():
        result["gates"][gate] = {}
        for key, prob in by_key.items():
            split, cond = key.split(":")
            route, conf = prob.argmax(1), prob.max(1)
            truth = lane_ids if split == "external" else insource_y
            entry = {"route_accuracy": round(float((route == truth).mean()), 5)}
            if split == "external":
                entry = utility_block(src, outputs, route, conf)
                confusion = np.zeros((5, 5), int)
                np.add.at(confusion, (lane_ids, route), 1)
                entry["confusion"] = confusion.tolist()
                entry["mean_conf_correct"] = round(float(conf[route == truth].mean()), 5)
                wrong = route != truth
                entry["mean_conf_wrong"] = round(float(conf[wrong].mean()), 5) if wrong.any() else None
            result["gates"][gate][key] = entry
    return {"provenance": "run", "lanes": external.LANES,
            "cohorts": {l: {"name": src[l]["cohort"], "n": int(len(src[l]["y"])),
                            "positives_or_grades": np.bincount(src[l]["y"]).tolist(),
                            "reference_metric": reference_metric(l, src[l]["y"], outputs[l])}
                        for l in external.LANES},
            "null_outputs": NULL, "conditions": gates.CONDITIONS, **result,
            "runtime_seconds": round(time.time() - start, 2)}

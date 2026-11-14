"""Calibrated abstention: choose the gate threshold on data disjoint from the test items.

External items are split 50/50 into calibration and test halves by group (patient,
breast, video-frame block or patch block) within each lane. The threshold is chosen on
the calibration half by Learn-then-Test with a Bonferroni correction over the threshold
grid: every tau whose one-sided Clopper-Pearson bound at level delta/|grid| on the selective
misroute rate is at most alpha is certified, and the lowest certified tau is used. Test coverage, selective risk and retained skill are then measured on the
untouched test half. A post-hoc threshold picked on the test half is reported for contrast.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import beta

from . import external, pools
from .dispatch import utility_block

ALPHA, DELTA = 0.02, 0.05
GRID = np.unique(np.r_[0.0, np.linspace(0.3, 0.99, 70), 1 - np.logspace(-2, -6, 30)])


def ucb(errors: int, n: int, level: float) -> float:
    return 1.0 if n == 0 else float(beta.ppf(1 - level, errors + 1, n - errors))


def choose_tau(correct: np.ndarray, conf: np.ndarray) -> float:
    """Bonferroni LTT over GRID; returns 1.0 (abstain on everything) if no tau is certified.

    A fixed-sequence scan from the highest tau stops at once when the top thresholds cover
    too few items to certify anything, so the family-wise correction is used instead.
    """
    level = DELTA / len(GRID)
    certified = [float(t) for t in GRID
                 if ucb(int((~correct[conf >= t]).sum()), int((conf >= t).sum()), level) <= ALPHA]
    return min(certified) if certified else 1.0


def split_halves(lane_ids: np.ndarray, groups: np.ndarray, seed: int = 2026) -> np.ndarray:
    """Boolean mask: True = calibration half, grouped within each lane."""
    rng = np.random.RandomState(seed)
    calib = np.zeros(len(lane_ids), bool)
    for lane in np.unique(lane_ids):
        idx = np.where(lane_ids == lane)[0]
        uniq = np.unique(groups[idx])
        chosen = set(rng.permutation(uniq)[: len(uniq) // 2].tolist())
        calib[idx] = [g in chosen for g in groups[idx]]
    return calib


def _rebuild_sources(data) -> tuple[dict, dict]:
    src = {l: {"y": data[f"y_{l}"]} for l in external.LANES}
    outputs = {l: data[f"out_{l}"] for l in external.LANES}
    return src, outputs


def evaluate(gates=("supervised", "probe", "probe_grayaug", "zero_shot_biomedclip")) -> dict:
    start = time.time()
    data = np.load(pools.cache_dir() / "dispatch_items.npz")
    lane_ids, groups = data["lane_ids"], data["groups"]
    src, outputs = _rebuild_sources(data)
    calib = split_halves(lane_ids, groups)
    scenarios = {"clean_to_clean": ("standard", "standard"),
                 "clean_to_grayscale": ("standard", "grayscale"),
                 "clean_to_jpeg": ("standard", "jpeg"),
                 "grayscale_to_grayscale": ("grayscale", "grayscale")}
    out = {}
    for gate in gates:
        out[gate] = {}
        for name, (cal_c, test_c) in scenarios.items():
            cal_p, test_p = data[f"{gate}|external:{cal_c}"], data[f"{gate}|external:{test_c}"]
            cal_ok = cal_p.argmax(1)[calib] == lane_ids[calib]
            tau = choose_tau(cal_ok, cal_p.max(1)[calib])
            route, conf = test_p.argmax(1), test_p.max(1)
            test_block = utility_block(src, outputs, route, conf, tau, ~calib)
            ok = (route == lane_ids)[~calib]
            covered = conf[~calib] >= tau
            risk = float((~ok[covered]).mean()) if covered.any() else 0.0
            # post-hoc contrast: the same rule fitted on the test half itself
            tau_post = choose_tau(ok, conf[~calib])
            post = utility_block(src, outputs, route, conf, tau_post, ~calib)
            no_abstain = utility_block(src, outputs, route, conf, 0.0, ~calib)
            out[gate][name] = {
                "tau": round(tau, 6), "test_coverage": test_block["coverage"],
                "test_selective_risk": round(risk, 5), "risk_target": ALPHA,
                "risk_met": bool(risk <= ALPHA),
                "test_macro_retained_skill": test_block["macro_retained_skill"],
                "no_abstention_macro_retained_skill": no_abstain["macro_retained_skill"],
                "no_abstention_route_accuracy": no_abstain["route_accuracy"],
                "posthoc_tau": round(tau_post, 6), "posthoc_coverage": post["coverage"],
                "per_lane_test": {l: test_block[l] for l in external.LANES}}
    return {"provenance": "run", "alpha": ALPHA, "delta": DELTA, "grid_size": int(len(GRID)),
            "n_calibration": int(calib.sum()), "n_test": int((~calib).sum()),
            "method": "grouped 50/50 calibration/test split; Learn-then-Test with Bonferroni "
                      "correction over the threshold grid and one-sided Clopper-Pearson bounds on the "
                      "selective misroute rate",
            "scenarios": {k: {"calibrate_on": v[0], "deploy_on": v[1]} for k, v in scenarios.items()},
            "gates": out, "runtime_seconds": round(time.time() - start, 2)}

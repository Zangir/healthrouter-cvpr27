"""Executed gate-to-specialist dispatch on unseen-institution images."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .data import cvc_images, external_router_sources, montgomery
from .metrics import bootstrap_auc, mean_interval, rounded_ci
from .models import FileDataset, convnext_model, device, image_transform, predict

# route index -> (route name, executable lane, native output type)
REGISTRY = {0: ("chest_xray", "tb_convnext_tiny_full.pt", "score"),
            1: ("endoscopy", "kvasir_yolo_seg", "mask"),
            2: ("pathology", "pcam_convnext_tiny_full.pt", "score"),
            3: ("fundus", "aptos_convnext_tiny.pt", "grade"),
            4: ("mammography", "cbis_convnext_tiny.pt", "score")}
THRESHOLDS = [0.0, 0.5, 0.9, 0.99, 0.999]


def _gate_predictions(root: str, files: list[str], variant: str):
    model = convnext_model(5, Path(root) / "router_convnext_tiny.pt")
    scores, _ = predict(model, FileDataset(files, np.zeros(len(files)),
                        image_transform(224, variant)), batch_size=64, multiclass=True)
    return scores.argmax(1), scores.max(1)


def _chest_lane(root: str) -> tuple[np.ndarray, np.ndarray]:
    files, labels = montgomery(root)
    model = convnext_model(1, Path(root) / "tb_convnext_tiny_full.pt")
    score, y = predict(model, FileDataset(files, labels, image_transform(384)), batch_size=32)
    return score, y


def _endoscopy_lane(root: str) -> np.ndarray:
    """Per-image Dice for the registered segmenter on every CVC-ClinicDB frame."""
    import cv2
    from ultralytics import YOLO
    segmenter = YOLO(Path(root) / "runs/segment/runs/kvasir_seg/weights/best.pt")
    dice = []
    for image_path in cvc_images(root):
        mask_path = Path(image_path).parent.parent / "masks" / Path(image_path).name
        ground = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if ground is None:
            ground = cv2.imread(str(mask_path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        ground = (ground > 127).astype(np.uint8)
        height, width = ground.shape
        output = segmenter.predict(image_path, device=str(device()), verbose=False,
                                   conf=0.25, retina_masks=True)[0]
        pred = np.zeros((height, width), np.uint8)
        if output.masks is not None:
            for mask in output.masks.data.cpu().numpy():
                pred |= (cv2.resize(mask, (width, height)) > 0.5).astype(np.uint8)
        total = pred.sum() + ground.sum()
        dice.append(1.0 if total == 0 else 2.0 * (pred & ground).sum() / total)
    return np.asarray(dice)


def _compose(route, confidence, reference, chest_score, chest_y, endoscopy_dice,
             threshold: float) -> dict:
    """Score the executed cascade: a wrong lane yields no valid output for that item."""
    covered = confidence >= threshold
    chest = reference == 0
    endoscopy = reference == 1
    chest_ok = covered[chest] & (route[chest] == 0)
    endoscopy_ok = covered[endoscopy] & (route[endoscopy] == 1)
    served_dice = np.where(endoscopy_ok, endoscopy_dice, 0.0)
    covered_endoscopy = covered & endoscopy
    block = {"threshold": threshold,
             "coverage": round(float(covered.mean()), 5),
             "selective_route_accuracy": round(float(
                 (route[covered] == reference[covered]).mean()), 5),
             "valid_output_rate": round(float((chest_ok.sum() + endoscopy_ok.sum())
                                             / len(reference)), 5),
             "chest_dispatched": int(chest_ok.sum()), "chest_total": int(chest.sum()),
             "endoscopy_dispatched": int(endoscopy_ok.sum()),
             "endoscopy_total": int(endoscopy.sum()),
             "endoscopy_covered_mean_dice": rounded_ci(mean_interval(
                 np.where(route[covered_endoscopy] == 1,
                          endoscopy_dice[covered[endoscopy]], 0.0)), 5)
             if covered_endoscopy.any() else None,
             "endoscopy_served_mean_dice": rounded_ci(mean_interval(served_dice), 5)}
    if len(np.unique(chest_y[chest_ok])) == 2:
        block["chest_dispatched_auroc"] = rounded_ci(
            bootstrap_auc(chest_y[chest_ok], chest_score[chest_ok]), 5)
    else:
        block["chest_dispatched_auroc"] = None
    return block


def evaluate(root: str) -> dict:
    start = time.time()
    files, reference, counts = external_router_sources(root)
    chest_score, chest_y = _chest_lane(root)
    endoscopy_dice = _endoscopy_lane(root)
    oracle = {"chest_auroc": rounded_ci(bootstrap_auc(chest_y, chest_score), 5),
              "endoscopy_mean_dice": rounded_ci(mean_interval(endoscopy_dice), 5),
              "definition": "reference routing; every item reaches its correct lane"}

    conditions = {}
    for variant in ["standard", "grayscale"]:
        route, confidence = _gate_predictions(root, files, variant)
        wrong = route != reference
        conditions[variant] = {
            "gate_accuracy": round(float((~wrong).mean()), 5),
            "misrouted_items": int(wrong.sum()),
            "misrouted_lanes": sorted({REGISTRY[int(r)][0] for r in route[wrong]}),
            "mean_confidence_correct": round(float(confidence[~wrong].mean()), 5),
            "mean_confidence_misrouted": (round(float(confidence[wrong].mean()), 5)
                                          if wrong.any() else None),
            "selective_dispatch": [
                _compose(route, confidence, reference, chest_score, chest_y,
                         endoscopy_dice, t) for t in THRESHOLDS],
        }
    full = {v: conditions[v]["selective_dispatch"][0] for v in conditions}
    return {
        "provenance": "run",
        "protocol": "gate prediction selects a registry lane; the selected lane is executed "
                    "and only schema-compatible outputs are scored",
        "n_items": int(len(reference)), "source_counts": counts,
        "executable_lanes": {str(k): v[1] for k, v in REGISTRY.items()},
        "oracle_routing": oracle, "conditions": conditions,
        "end_to_end_vs_oracle": {
            "standard_endoscopy_dice_loss": round(
                oracle["endoscopy_mean_dice"]["value"]
                - full["standard"]["endoscopy_served_mean_dice"]["value"], 5),
            "grayscale_endoscopy_dice_loss": round(
                oracle["endoscopy_mean_dice"]["value"]
                - full["grayscale"]["endoscopy_served_mean_dice"]["value"], 5),
            "grayscale_valid_output_rate": full["grayscale"]["valid_output_rate"],
        },
        "runtime_seconds": round(time.time() - start, 2)}

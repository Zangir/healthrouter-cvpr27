"""Inference-only revalidation of the five supplied specialist checkpoints."""
from __future__ import annotations

import glob
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms

from .data import (aptos_split, cbis_split, cvc_images, data_dir, montgomery,
                   nonoverlap_mask, pcam_arrays, pcam_test_nonoverlap_mask)
from .metrics import bootstrap_accuracy, bootstrap_auc, bootstrap_qwk, mean_interval, rounded_ci
from .models import ArrayDataset, FileDataset, MEAN, STD, convnext_model, device, image_transform, predict


def _binary(root, checkpoint, files, labels, size, n_boot=10_000):
    model = convnext_model(1, Path(root) / checkpoint)
    score, y = predict(model, FileDataset(files, labels, image_transform(size)),
                       batch_size=20 if size >= 512 else 32)
    return rounded_ci(bootstrap_auc(y, score, n_boot), 5), score, y


def _segmentation_scores(model, images: list[str], mask_lookup):
    scores = []
    for image_path in images:
        mask_path = mask_lookup(image_path) / Path(image_path).name
        if not mask_path.exists():
            mask_path = mask_path.with_suffix(".png")
        ground = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if ground is None:
            continue
        ground = (ground > 127).astype(np.uint8)
        height, width = ground.shape
        output = model.predict(image_path, device=str(device()), verbose=False, conf=0.25,
                               retina_masks=True)[0]
        pred = np.zeros((height, width), np.uint8)
        if output.masks is not None:
            for mask in output.masks.data.cpu().numpy():
                pred |= (cv2.resize(mask, (width, height)) > 0.5).astype(np.uint8)
        denominator = pred.sum() + ground.sum()
        scores.append(1.0 if denominator == 0 else 2.0 * (pred & ground).sum() / denominator)
    return np.asarray(scores)


def evaluate(root: str) -> dict:
    start = time.time()
    base = data_dir(root)
    mont_files, mont_y = montgomery(root)
    tb, _, _ = _binary(root, "tb_convnext_tiny_full.pt", mont_files, mont_y, 384)

    _, _, aptos_files, aptos_y = aptos_split(root)
    aptos_model = convnext_model(5, Path(root) / "aptos_convnext_tiny.pt")
    aptos_score, aptos_label = predict(aptos_model, FileDataset(aptos_files, aptos_y,
                                       image_transform(384)), batch_size=32, multiclass=True)
    aptos_pred = aptos_score.argmax(1)
    aptos_train, _, _, _ = aptos_split(root)
    aptos_keep = nonoverlap_mask(aptos_train, aptos_files)
    aptos = {"raw_qwk": rounded_ci(bootstrap_qwk(aptos_label, aptos_pred), 5),
             "clean_qwk": rounded_ci(bootstrap_qwk(aptos_label[aptos_keep],
                                                    aptos_pred[aptos_keep]), 5),
             "raw_accuracy": round(float(accuracy_score(aptos_label, aptos_pred)), 5),
             "clean_accuracy": round(float(accuracy_score(aptos_label[aptos_keep],
                                                           aptos_pred[aptos_keep])), 5),
             "n_raw": len(aptos_label), "n_clean": int(aptos_keep.sum()),
             "cross_split_duplicates": int((~aptos_keep).sum())}

    _, _, cbis_files, cbis_y = cbis_split(root)
    cbis_train, _, _, _ = cbis_split(root)
    cbis_raw, cbis_score, cbis_label = _binary(
        root, "cbis_convnext_tiny.pt", cbis_files, cbis_y, 512)
    cbis_keep = nonoverlap_mask(cbis_train, cbis_files)
    cbis = {"raw": cbis_raw,
            "clean": rounded_ci(bootstrap_auc(cbis_label[cbis_keep],
                                               cbis_score[cbis_keep]), 5),
            "n_raw": len(cbis_label), "n_clean": int(cbis_keep.sum()),
            "cross_split_duplicates": int((~cbis_keep).sum())}

    pcam_x, pcam_y = pcam_arrays(root, "test")
    pcam_tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                  transforms.Normalize(MEAN, STD)])
    pcam_model = convnext_model(1, Path(root) / "pcam_convnext_tiny_full.pt")
    pcam_score, pcam_label = predict(pcam_model, ArrayDataset(pcam_x, pcam_y, pcam_tf),
                                     batch_size=384)
    pcam_keep = pcam_test_nonoverlap_mask(root, pcam_x)
    pcam = {"raw": rounded_ci(bootstrap_auc(pcam_label, pcam_score, n_boot=1_000), 5),
            "clean": rounded_ci(bootstrap_auc(pcam_label[pcam_keep],
                                               pcam_score[pcam_keep], n_boot=1_000), 5),
            "n_raw": len(pcam_label), "n_clean": int(pcam_keep.sum()),
            "cross_split_duplicates": int((~pcam_keep).sum())}

    from ultralytics import YOLO
    weight = Path(root) / "runs/segment/runs/kvasir_seg/weights/best.pt"
    segmenter = YOLO(weight)
    kvasir_images = sorted(glob.glob(str(base / "kvasir_yolo/images/val/*.jpg")))
    kvasir_masks = lambda _: base / "Kvasir-SEG/masks"
    kvasir_dice = _segmentation_scores(segmenter, kvasir_images, kvasir_masks)
    cvc_masks = lambda path: Path(path).parent.parent / "masks"
    cvc_dice = _segmentation_scores(segmenter, cvc_images(root), cvc_masks)
    segmentation = {
        "kvasir_validation_mean_dice": rounded_ci(mean_interval(kvasir_dice), 5),
        "kvasir_validation_median_dice": round(float(np.median(kvasir_dice)), 5),
        "kvasir_n": len(kvasir_dice),
        "cvc_external_mean_dice": rounded_ci(mean_interval(cvc_dice), 5),
        "cvc_external_median_dice": round(float(np.median(cvc_dice)), 5),
        "cvc_n": len(cvc_dice),
    }
    return {"provenance": "run", "tb_montgomery_auroc": tb,
            "aptos_random_holdout": aptos, "cbis_local_mirror_test_auroc": cbis,
            "pcam_official_test_auroc": pcam, "segmentation": segmentation,
            "runtime_seconds": round(time.time() - start, 2)}

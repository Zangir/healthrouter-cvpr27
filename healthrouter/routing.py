"""Evaluate the saved modality router under source and acquisition shift."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import external_router_sources, nonoverlap_mask, router_sources
from .metrics import bootstrap_accuracy, rounded_ci
from .models import FileDataset, convnext_model, image_transform, imagenet_features, predict


def _image_statistics(paths: list[str]) -> np.ndarray:
    rows = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        image.thumbnail((64, 64))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        gray = arr.mean(2)
        dx = np.abs(np.diff(gray, axis=1)).mean()
        dy = np.abs(np.diff(gray, axis=0)).mean()
        q = np.quantile(arr.reshape(-1, 3), [0.1, 0.5, 0.9], axis=0).ravel()
        rows.append(np.r_[arr.mean((0, 1)), arr.std((0, 1)), q, dx, dy,
                          width / max(height, 1)])
    return np.asarray(rows)


def _accuracy_block(labels, pred, n_boot=10_000):
    point, lo, hi = bootstrap_accuracy(labels, pred, n_boot)
    return {**rounded_ci((point, lo, hi), 5), "correct": int((pred == labels).sum()),
            "n": int(len(labels))}


def evaluate(root: str, seed: int = 0) -> dict:
    start = time.time()
    train_files, train_y, test_files, test_y, classes = router_sources(root)
    external_files, external_y, source_counts = external_router_sources(root)
    checkpoint = Path(root) / "router_convnext_tiny.pt"
    model = convnext_model(5, checkpoint)

    test_scores, _ = predict(model, FileDataset(test_files, test_y, image_transform(224)),
                             batch_size=64, multiclass=True)
    external_scores, _ = predict(
        model, FileDataset(external_files, external_y, image_transform(224)),
        batch_size=64, multiclass=True)
    test_pred = test_scores.argmax(1)
    external_pred = external_scores.argmax(1)
    keep = nonoverlap_mask(train_files, test_files)

    corruptions = {}
    for variant in ["grayscale", "blur", "low_resolution"]:
        scores, _ = predict(model, FileDataset(external_files, external_y,
                            image_transform(224, variant)), batch_size=64, multiclass=True)
        corruptions[variant] = _accuracy_block(external_y, scores.argmax(1), 2_000)

    train_x = _image_statistics(train_files)
    test_x = _image_statistics(test_files)
    external_x = _image_statistics(external_files)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=seed))
    baseline.fit(train_x, train_y)
    baseline_test = baseline.predict(test_x)
    baseline_external = baseline.predict(external_x)

    frozen_train = imagenet_features(train_files, size=224, batch_size=64)
    frozen_test = imagenet_features(test_files, size=224, batch_size=64)
    frozen_external = imagenet_features(external_files, size=224, batch_size=64)
    frozen = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, C=0.1,
                           random_state=seed)).fit(frozen_train, train_y)
    frozen_test_pred = frozen.predict(frozen_test)
    frozen_external_pred = frozen.predict(frozen_external)

    scale, simple_scale = {}, {}
    rng = np.random.RandomState(2026)
    for n_per_class in [10, 25, 50, 100, 200, 400]:
        scores, simple_scores = [], []
        for repeat in range(5):
            chosen = []
            for class_id in range(5):
                pool = np.where(train_y == class_id)[0]
                chosen.extend(rng.choice(pool, min(n_per_class, len(pool)), replace=False))
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, C=0.1,
                                  random_state=repeat)).fit(frozen_train[chosen], train_y[chosen])
            simple = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000,
                                   random_state=repeat)).fit(train_x[chosen], train_y[chosen])
            scores.append(float((probe.predict(frozen_external) == external_y).mean()))
            simple_scores.append(float((simple.predict(external_x) == external_y).mean()))
        scale[str(n_per_class)] = {"mean": round(float(np.mean(scores)), 5),
                                   "std": round(float(np.std(scores)), 5),
                                   "replicates": [round(x, 5) for x in scores]}
        simple_scale[str(n_per_class)] = {"mean": round(float(np.mean(simple_scores)), 5),
                                          "std": round(float(np.std(simple_scores)), 5)}

    trace = []
    for i in np.linspace(0, len(external_files) - 1, 5, dtype=int):
        pred = int(external_pred[i])
        trace.append({"file": Path(external_files[i]).name, "true_route": classes[int(external_y[i])],
                      "predicted_route": classes[pred],
                      "confidence": round(float(external_scores[i, pred]), 5)})
    return {
        "provenance": "run", "seed": seed, "classes": classes,
        "in_source": _accuracy_block(test_y, test_pred),
        "in_source_after_exact_dedup": _accuracy_block(test_y[keep], test_pred[keep]),
        "in_source_cross_split_duplicates": int((~keep).sum()),
        "unseen_institution_two_modalities": _accuracy_block(external_y, external_pred),
        "unseen_source_counts": source_counts,
        "simple_feature_baseline_in_source": _accuracy_block(test_y, baseline_test),
        "simple_feature_baseline_external": _accuracy_block(external_y, baseline_external),
        "frozen_imagenet_router_in_source": _accuracy_block(test_y, frozen_test_pred),
        "frozen_imagenet_router_external": _accuracy_block(external_y, frozen_external_pred),
        "corruption_ablations": corruptions, "data_scale_ablation": scale,
        "simple_feature_data_scale": simple_scale,
        "route_trace": trace, "runtime_seconds": round(time.time() - start, 2),
    }

"""Assemble canonical results strictly from persisted run records."""
from __future__ import annotations

import json
from pathlib import Path


def load_results(results_dir: str | Path) -> dict:
    base = Path(results_dir)
    required = ["router.json", "tb_probes.json", "specialists.json"]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run required experiments first: {missing}")
    return {name[:-5]: json.loads((base / name).read_text()) for name in required}


def aggregate(results_dir: str | Path) -> dict:
    runs = load_results(results_dir)
    router = runs["router"]
    probes = runs["tb_probes"]
    specialists = runs["specialists"]
    return {
        "artifact": "HealthRouter", "provenance": "run",
        "headline": {
            "rad_dino_fold_mean_auroc": probes["rad_dino_probe"]["fold_mean"],
            "imagenet_convnext_fold_mean_auroc": probes["imagenet_convnext_probe"]["fold_mean"],
            "paired_oof_auroc_difference": probes["paired_oof_auroc_difference"],
        },
        "runtime_seconds_total": round(
            router["runtime_seconds"] + probes["runtime_seconds"]
            + specialists["runtime_seconds"], 2
        ),
        "routing": router, "specialists": specialists,
        "user_provided_prior": {
            "tb_finetune_five_fold_mean_auroc": 0.9545,
            "tb_finetune_five_fold_std_auroc": 0.012,
            "pcam_100k_best_test_auroc": 0.9636,
            "note": "Preliminary seed results not re-trained in this release."
        },
    }


def ablations(results_dir: str | Path) -> dict:
    runs = load_results(results_dir)
    router, probes, specialists = runs["router"], runs["tb_probes"], runs["specialists"]
    return {
        "representation_package": {
            "imagenet_convnext_fold_mean_auroc": probes["imagenet_convnext_probe"]["fold_mean"],
            "rad_dino_fold_mean_auroc": probes["rad_dino_probe"]["fold_mean"],
            "paired_oof_auroc_difference": probes["paired_oof_auroc_difference"],
        },
        "routing_source_shift": {
            "learned_convnext_router_in_source_accuracy": router["in_source"]["value"],
            "learned_convnext_router_external_accuracy": router["unseen_institution_two_modalities"]["value"],
            "frozen_imagenet_convnext_router_in_source_accuracy": router["frozen_imagenet_router_in_source"]["value"],
            "frozen_imagenet_convnext_router_external_accuracy": router["frozen_imagenet_router_external"]["value"],
            "rgb_statistics_router_in_source_accuracy": router["simple_feature_baseline_in_source"]["value"],
            "rgb_statistics_router_external_accuracy": router["simple_feature_baseline_external"]["value"],
        },
        "routing_corruptions": router["corruption_ablations"],
        "frozen_imagenet_convnext_router_data_scale": router["data_scale_ablation"],
        "duplicate_exclusion": {
            "aptos_qwk_raw": specialists["aptos_random_holdout"]["raw_qwk"]["value"],
            "aptos_qwk_clean": specialists["aptos_random_holdout"]["clean_qwk"]["value"],
            "cbis_auroc_raw": specialists["cbis_local_mirror_test_auroc"]["raw"]["value"],
            "cbis_auroc_clean": specialists["cbis_local_mirror_test_auroc"]["clean"]["value"],
            "pcam_auroc_raw": specialists["pcam_official_test_auroc"]["raw"]["value"],
            "pcam_auroc_clean": specialists["pcam_official_test_auroc"]["clean"]["value"],
        },
    }

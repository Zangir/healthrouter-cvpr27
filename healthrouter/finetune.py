"""Does the frozen-probe ranking survive full fine-tuning?

Same pools, same seed-42 folds, same 224-px square pixels as the matched grid. One recipe
for every package (AdamW, backbone lr 3e-5, head lr 1e-3, one-cycle schedule, 6 epochs, batch
16 accumulated from micro-batches of 4, flip + mild random-resized-crop), last epoch scored, no checkpoint selection on test.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from . import packages, pools
from .grid2 import _folds
from .metrics import paired_metric_difference, qwk, rounded_ci
from .models import device

EPOCHS, BATCH, LR_BACKBONE, LR_HEAD = 6, 16, 3e-5, 1e-3
MICRO = 4  # micro-batch; gradients accumulated to BATCH (fits the 4.8 GB MPS cap)
DOMAINS = {"chest_xray": "auroc", "fundus": "qwk"}
FT_PACKAGES = ["imagenet_convnext", "dinov2_base", "rad_dino"]


class _Net(nn.Module):
    def __init__(self, package: str, n_out: int):
        super().__init__()
        _, _, key, (mean, std) = packages.PACKAGES[package]
        self.backbone, self.run = packages._load(key)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        with torch.no_grad():
            dim = self.run(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        return self.head(self.run((x - self.mean) / self.std))


def _augment(x: torch.Tensor, rng: np.random.RandomState) -> torch.Tensor:
    if rng.rand() < 0.5:
        x = x.flip(3)
    s = int(224 * rng.uniform(0.85, 1.0))
    i, j = rng.randint(0, 224 - s + 1, size=2)
    return nn.functional.interpolate(x[:, :, i:i + s, j:j + s], size=224, mode="bilinear",
                                     align_corners=False)


def _train_fold(package, pixels, y, train, test, n_out, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = _Net(package, n_out).to(device())
    if hasattr(net.backbone, "gradient_checkpointing_enable"):
        net.backbone.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": LR_BACKBONE},
                             {"params": net.head.parameters(), "lr": LR_HEAD}], weight_decay=0.05)
    steps = EPOCHS * int(np.ceil(len(train) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_BACKBONE, LR_HEAD],
                                                total_steps=steps, pct_start=0.1)
    tensor = lambda idx: torch.from_numpy(np.ascontiguousarray(pixels[idx])).permute(
        0, 3, 1, 2).float().div(255.0).to(device())
    net.train()
    for _ in range(EPOCHS):
        order = rng.permutation(train)
        for s in range(0, len(order), BATCH):
            idx = np.sort(order[s:s + BATCH])
            if len(idx) < 2:
                continue
            opt.zero_grad()
            x = _augment(tensor(idx), rng)
            target = torch.from_numpy(y[idx]).long().to(device())
            for m in range(0, len(idx), MICRO):
                loss = nn.functional.cross_entropy(net(x[m:m + MICRO]), target[m:m + MICRO],
                                                   reduction="sum") / len(idx)
                loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.inference_mode():
        prob = np.concatenate([net(tensor(test[s:s + 8])).softmax(1).float().cpu().numpy()
                               for s in range(0, len(test), 8)])
    del net, opt
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    return prob


def evaluate(root: str) -> dict:
    start = time.time()
    grid = pools.grid_pools(root)
    out = {}
    for domain, metric in DOMAINS.items():
        pixels, y = grid[domain]["cached"][224], grid[domain]["y"].astype(int)
        n_out = len(np.unique(y))
        score = (lambda yy, p: roc_auc_score(yy, p[:, 1])) if metric == "auroc" else \
                (lambda yy, p: qwk(yy, p.argmax(1)))
        probs, entries = {}, {}
        for package in FT_PACKAGES:
            t0 = time.time()
            prob = np.zeros((len(y), n_out))
            folds = []
            for k, (train, test) in enumerate(_folds(y, 42)):
                part = pools.cache_dir() / f"finetune_{domain}_{package}_fold{k}.npy"
                if not part.exists():  # resumable: each fold's predictions cached on disk
                    np.save(part, _train_fold(package, pixels, y, train, test, n_out, 42 + k))
                prob[test] = np.load(part)
                folds.append(float(score(y[test], prob[test])))
            probs[package] = prob
            entries[package] = {"oof_score": round(float(score(y, prob)), 5),
                                "fold_scores": [round(f, 5) for f in folds],
                                "seconds": round(time.time() - t0, 1)}
            print(f"  finetune {domain}/{package}: {entries[package]}", flush=True)
        contrasts = {}
        for a, b in [("rad_dino", "dinov2_base"), ("rad_dino", "imagenet_convnext"),
                     ("dinov2_base", "imagenet_convnext")]:
            contrasts[f"{a}_minus_{b}"] = rounded_ci(paired_metric_difference(
                y, probs[a], probs[b], score, n_boot=2_000), 5)
        out[domain] = {"metric": metric, "n_items": int(len(y)), "packages": entries,
                       "best_package": max(entries, key=lambda p: entries[p]["oof_score"]),
                       "contrasts": contrasts}
    return {"provenance": "run", "protocol": __doc__.strip().splitlines()[2:5],
            "resolution": 224, "fold_seed": 42, "domains": out,
            "runtime_seconds": round(time.time() - start, 2)}

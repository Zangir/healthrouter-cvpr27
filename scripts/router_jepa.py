#!/usr/bin/env python3
"""I-JEPA router-feature ablation: mean-pooled I-JEPA ViT-H/14 embeddings of the in-source
pools (same square 224 px pixels as every encoder). Checkpointed per pool; --budget seconds."""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch

from healthrouter.collection import data
from healthrouter.models import device

NAME = "facebook/ijepa_vith14_1k"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=480)
    p.add_argument("--tasks", nargs="+", default=data.MED_TASKS)
    a = p.parse_args()
    start = time.time()
    todo = [t for t in a.tasks if not data.feature_path(t, "ijepa", 224).exists()]
    if not todo:
        print("ALL DONE")
        return
    from transformers import IJepaModel
    model = IJepaModel.from_pretrained(NAME).to(device()).eval()
    params = round(sum(q.numel() for q in model.parameters()) / 1e6, 1)
    mean = torch.tensor(MEAN, device=device()).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device()).view(1, 3, 1, 1)
    for t in todo:
        pool = np.load(glob.glob(str(data.cache_dir() / f"pool_{t}_224_*.npy"))[0], mmap_mode="r")
        _, y = data.load(t, "imagenet_convnext", 224)
        part = data.cache_dir() / "features" / f"{t}_ijepa_224.partial.npy"
        feats = list(np.load(part)) if part.exists() else []
        t0 = time.time()
        with torch.inference_mode():
            for s in range(len(feats), len(pool), 8):
                x = torch.from_numpy(np.ascontiguousarray(pool[s:s + 8])).to(device())
                x = (x.permute(0, 3, 1, 2).float() / 255.0 - mean) / std
                feats.extend(model(pixel_values=x).last_hidden_state.mean(1).float().cpu().numpy())
                if time.time() - start > a.budget:
                    np.save(part, np.stack(feats))
                    print(f"{t}: {len(feats)}/{len(pool)} saved; budget reached", flush=True)
                    return
        np.savez(data.feature_path(t, "ijepa", 224), x=np.stack(feats), y=y, params=params,
                 seconds=time.time() - t0)
        part.unlink(missing_ok=True)
        print(f"{t}: done {len(feats)} in {time.time() - t0:.0f}s", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()

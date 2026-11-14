"""Run every registered specialist on its own external cohort (clean inputs, frozen)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import external
from .cascade import _endoscopy_lane
from .models import MEAN, STD, convnext_model, device

# lane -> (checkpoint, outputs, input size, output type)
SPECIALISTS = {"chest_xray": ("tb_convnext_tiny_full.pt", 1, 384, "score"),
               "pathology": ("pcam_convnext_tiny_full.pt", 1, 96, "score"),
               "fundus": ("aptos_convnext_tiny.pt", 5, 384, "grade"),
               "mammography": ("cbis_convnext_tiny.pt", 1, 512, "score"),
               "endoscopy": ("kvasir_yolo_seg", None, None, "mask")}


class _Items(Dataset):
    def __init__(self, items, size):
        self.items = items
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(external.decode(self.items[i]))


def _classify(root: str, lane: str, items) -> np.ndarray:
    checkpoint, outputs, size, _ = SPECIALISTS[lane]
    model = convnext_model(outputs, Path(root) / checkpoint)
    scores = []
    with torch.inference_mode():
        for x in DataLoader(_Items(items, size), batch_size=4 if size >= 384 else 32):
            logits = model(x.to(device()))
            out = logits.softmax(1) if outputs > 1 else logits.sigmoid().squeeze(1)
            scores.append(out.float().cpu().numpy())
    return np.concatenate(scores)


def lane_outputs(root: str, sources: dict) -> dict:
    """{lane: per-item native output}: score, 5-way grade probabilities, or Dice."""
    from . import pools
    out = {}
    for lane, source in sources.items():
        part = pools.cache_dir() / f"lane_{lane}_{len(source['y'])}.npy"
        if part.exists():  # resumable: specialist outputs cached once per cohort
            out[lane] = np.load(part)
        else:
            out[lane] = _endoscopy_lane(root) if lane == "endoscopy" else \
                _classify(root, lane, source["items"])
            np.save(part, out[lane])
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  lane {lane}: {len(out[lane])} outputs", flush=True)
    return out

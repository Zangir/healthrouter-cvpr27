"""Load the Mammo-CLIP EfficientNet-B2 image encoder without unsafe unpickling.

The released checkpoint is a torch zip archive whose pickle also stores omegaconf
configuration objects. Instead of ``torch.load(weights_only=False)`` (arbitrary code), a
restricted unpickler rebuilds only tensors and plain containers; every other class becomes
an inert stub, so no foreign code runs.
"""
from __future__ import annotations

import collections
import pickle
import zipfile

import torch
from torch import nn

REPO = "shawn24/Mammo-CLIP"
FILE = "Pre-trained-checkpoints/b2-model-best-epoch-10.tar"
ALLOWED = {("collections", "OrderedDict"): collections.OrderedDict,
           ("collections", "defaultdict"): collections.defaultdict,
           ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
           ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
           ("builtins", "set"): set}
STORAGES = {"FloatStorage": torch.float32, "HalfStorage": torch.float16,
            "LongStorage": torch.int64, "IntStorage": torch.int32,
            "BoolStorage": torch.bool, "DoubleStorage": torch.float64}


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _read_state_dict(path: str) -> dict:
    archive = zipfile.ZipFile(path)
    prefix = archive.namelist()[0].split("/")[0]

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in ALLOWED:
                return ALLOWED[(module, name)]
            if module == "torch" and name in STORAGES:
                return name
            return type(name, (_Stub,), {})

        def persistent_load(self, pid):
            _, storage_type, key, _, numel = pid
            dtype = STORAGES[storage_type]
            raw = bytearray(archive.read(f"{prefix}/data/{key}"))
            untyped = torch.frombuffer(raw, dtype=torch.uint8).untyped_storage() if raw \
                else torch.UntypedStorage(0)
            return torch.storage.TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)

    return Restricted(archive.open(f"{prefix}/data.pkl")).load()["model"]


class _PooledEncoder(nn.Module):
    """EfficientNet-B2 trunk (efficientnet_pytorch layout) + global average pooling."""

    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk

    def forward(self, x):
        if x.shape[1] != self.trunk._conv_stem.in_channels:
            x = x.mean(1, keepdim=True)
        return self.trunk.extract_features(x).mean(dim=(2, 3))


def load_image_encoder() -> nn.Module:
    from efficientnet_pytorch import EfficientNet
    from huggingface_hub import hf_hub_download
    state = _read_state_dict(hf_hub_download(REPO, FILE))
    prefix = "image_encoder."
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    channels = weights["_conv_stem.weight"].shape[1]
    trunk = EfficientNet.from_name("efficientnet-b2", in_channels=channels)
    missing, _ = trunk.load_state_dict(weights, strict=False)
    if [m for m in missing if not m.startswith("_fc")]:
        raise RuntimeError(f"Mammo-CLIP weights incomplete: {missing[:5]}")
    return _PooledEncoder(trunk)

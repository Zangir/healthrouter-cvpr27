"""Fixed, leakage-aware indexes for the local HealthRouter benchmarks."""
from __future__ import annotations

import glob
import hashlib
from pathlib import Path

import numpy as np


def nonoverlap_mask(train_files: list[str], test_files: list[str]) -> np.ndarray:
    """Exclude byte-identical test images found in training; never relabel."""
    def digest(path: str) -> bytes:
        handle = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                handle.update(block)
        return handle.digest()
    train_hashes = {digest(path) for path in train_files}
    return np.array([digest(path) not in train_hashes for path in test_files])


def data_dir(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidate = root / "data"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Expected benchmark data at {candidate}")
    return candidate


def shenzhen(root: str | Path) -> tuple[list[str], np.ndarray]:
    files = sorted(glob.glob(str(data_dir(root) / "ChinaSet_AllFiles/CXR_png/*.png")))
    labels = np.array([int(Path(f).stem.rsplit("_", 1)[1]) for f in files])
    if len(files) != 662:
        raise RuntimeError(f"Expected 662 Shenzhen images, found {len(files)}")
    return files, labels


def tb_folds(labels: np.ndarray, k: int = 5, seed: int = 42):
    rng = np.random.RandomState(seed)
    pos = rng.permutation(np.where(labels == 1)[0])
    neg = rng.permutation(np.where(labels == 0)[0])
    for fold in range(k):
        test = np.concatenate([pos[fold::k], neg[fold::k]])
        train = np.setdiff1d(np.arange(len(labels)), test)
        yield train, test


def stratified_folds(labels: np.ndarray, k: int = 5, seed: int = 42):
    """Class-stratified folds for any label cardinality (binary uses tb_folds)."""
    rng = np.random.RandomState(seed)
    by_class = [rng.permutation(np.where(labels == c)[0]) for c in np.unique(labels)]
    for fold in range(k):
        test = np.concatenate([members[fold::k] for members in by_class])
        yield np.setdiff1d(np.arange(len(labels)), test), test


def montgomery(root: str | Path) -> tuple[list[str], np.ndarray]:
    files = sorted(glob.glob(str(data_dir(root) / "MontgomerySet/CXR_png/*.png")))
    labels = np.array([int(Path(f).stem.rsplit("_", 1)[1]) for f in files])
    return files, labels


def aptos_split(root: str | Path, seed: int = 42):
    files = sorted(glob.glob(str(data_dir(root) / "aptos_img/*/*.jpg")))
    labels = np.array([int(Path(f).parent.name) for f in files])
    order = np.random.RandomState(seed).permutation(len(files))
    cut = int(0.8 * len(files))
    return ([files[i] for i in order[:cut]], labels[order[:cut]],
            [files[i] for i in order[cut:]], labels[order[cut:]])


def cbis_split(root: str | Path):
    files = glob.glob(str(data_dir(root) / "cbis_ddsm_1024/**/*.png"), recursive=True)
    def label(path: str) -> int:
        return int("/cancer" in path.replace("not_cancer", "NC"))
    train = [f for f in files if "train" in f.lower()]
    test = [f for f in files if "test" in f.lower()]
    return train, np.array([label(f) for f in train]), test, np.array([label(f) for f in test])


def cvc_images(root: str | Path) -> list[str]:
    return sorted(glob.glob(str(data_dir(root) / "cvc_clinicdb/*/images/*")))


def router_sources(root: str | Path):
    """Reconstruct the seed router's source pool and exact random split."""
    base = data_dir(root)
    sources = {
        "chest_xray": glob.glob(str(base / "ChinaSet_AllFiles/CXR_png/*.png")),
        "endoscopy": glob.glob(str(base / "Kvasir-SEG/images/*.jpg")),
        "pathology": glob.glob(str(base / "router_pcam_tmp/*.png")),
        "fundus": glob.glob(str(base / "aptos_img/*/*.jpg")),
        "mammography": glob.glob(str(base / "cbis_ddsm_1024/**/*.png"), recursive=True),
    }
    rng = np.random.RandomState(42)
    files, labels, names = [], [], []
    for class_id, (name, paths) in enumerate(sources.items()):
        paths = list(paths)
        rng.shuffle(paths)
        paths = paths[:500]
        if len(paths) != 500:
            raise RuntimeError(f"Router source {name} has {len(paths)} images")
        files.extend(paths)
        labels.extend([class_id] * len(paths))
        names.append(name)
    order = rng.permutation(len(files))
    cut = int(0.8 * len(files))
    tr = order[:cut]
    te = order[cut:]
    return ([files[i] for i in tr], np.array([labels[i] for i in tr]),
            [files[i] for i in te], np.array([labels[i] for i in te]), names)


def external_router_sources(root: str | Path):
    base = data_dir(root)
    chest = sorted(glob.glob(str(base / "MontgomerySet/CXR_png/*.png")))
    endoscopy = cvc_images(root)
    files = chest + endoscopy
    labels = np.array([0] * len(chest) + [1] * len(endoscopy))
    return files, labels, {"chest_xray": len(chest), "endoscopy": len(endoscopy)}


def pcam_arrays(root: str | Path, split: str):
    import h5py
    base = data_dir(root)
    with h5py.File(base / f"camelyonpatch_level_2_split_{split}_y.h5") as handle:
        labels = handle["y"][:].reshape(-1).astype(np.float32)
    with h5py.File(base / f"camelyonpatch_level_2_split_{split}_x.h5") as handle:
        images = handle["x"][:]
    return images, labels


def pcam_test_nonoverlap_mask(root: str | Path, test_images: np.ndarray) -> np.ndarray:
    """Stream PCam train patches and flag exact cross-split duplicates."""
    import h5py
    path = data_dir(root) / "camelyonpatch_level_2_split_train_x.h5"
    hashes: set[bytes] = set()
    with h5py.File(path) as handle:
        dataset = handle["x"]
        for start in range(0, len(dataset), 4096):
            for image in dataset[start:start + 4096]:
                hashes.add(hashlib.blake2b(image.tobytes(), digest_size=16).digest())
    return np.array([hashlib.blake2b(image.tobytes(), digest_size=16).digest() not in hashes
                     for image in test_images])

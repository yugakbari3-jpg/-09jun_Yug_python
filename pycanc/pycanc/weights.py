"""
Loading of the official Sybil ensemble (5 checkpoints + calibrators) released by
the Barzilay lab at MIT:  https://github.com/reginabarzilaygroup/Sybil/releases

The checkpoints contain an ``argparse.Namespace`` and numpy scalars next to the
weights, so we unpickle them with ``weights_only=True`` and an explicit
allow-list instead of falling back to arbitrary-code unpickling.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from .model import PyCancNet

CHECKPOINT_URL = "https://github.com/reginabarzilaygroup/Sybil/releases/download/v1.5.0/sybil_checkpoints.zip"
CHECKPOINT_IDS = [
    "28a7cd44f5bcd3e6cc760b65c7e0d54d",
    "56ce1a7d241dc342982f5466c4a9d7ef",
    "64a91b25f84141d32852e75a3aec7305",
    "65fd1f04cb4c5847d86a9ed8ba31ac1a",
    "624407ef8e3a2a009f9fa51f9846fe9a",
]


def _default_dir() -> Path:
    env = os.environ.get("PYCANC_CHECKPOINT_DIR") or os.environ.get("SYBIL_CHECKPOINT_DIR")
    if env:
        return Path(env)
    legacy = Path.home() / ".sybil"  # where earlier versions downloaded to
    if (legacy / f"{CHECKPOINT_IDS[0]}.ckpt").exists():
        return legacy
    return Path.home() / ".pycanc"


DEFAULT_DIR = _default_dir()


def _safe_globals():
    from numpy._core.multiarray import _reconstruct, scalar

    return [
        argparse.Namespace,
        np.ndarray,
        np.dtype,
        (scalar, "numpy.core.multiarray.scalar"),
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        (type(np.dtype("float64")), "numpy.dtypes.Float64DType"),
        (type(np.dtype("float32")), "numpy.dtypes.Float32DType"),
        (type(np.dtype("int64")), "numpy.dtypes.Int64DType"),
    ]


def download_checkpoints(target: Path = DEFAULT_DIR, progress=print) -> Path:
    target = Path(target)
    if all((target / f"{c}.ckpt").exists() for c in CHECKPOINT_IDS):
        return target
    target.mkdir(parents=True, exist_ok=True)
    zpath = target / "sybil_checkpoints.zip"
    progress(f"Downloading official MIT Sybil checkpoints (~700 MB) to {target} ...")
    urllib.request.urlretrieve(CHECKPOINT_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or "/" in name:
                continue
            z.extract(name, target)
    zpath.unlink()
    return target


def load_checkpoint(path: Path) -> PyCancNet:
    with torch.serialization.safe_globals(_safe_globals()):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    model = PyCancNet()
    model.load_state_dict(sd, strict=True)
    return model.eval()


# --------------------------------------------------------------------------- #
#  Calibration (Platt scaling -> isotonic regression, averaged over CV folds)
# --------------------------------------------------------------------------- #
class SimpleIsotonic:
    def __init__(self, coef, intercept, x0, y0, x_min=-np.inf, x_max=np.inf):
        self.coef = np.asarray(coef, dtype=float) if coef is not None else None
        self.intercept = np.asarray(intercept, dtype=float) if intercept is not None else 0.0
        self.x0, self.y0 = np.asarray(x0, float), np.asarray(y0, float)
        self.x_min, self.x_max = x_min, x_max

    def __call__(self, p: np.ndarray) -> np.ndarray:
        t = p.reshape(-1, 1)
        if self.coef is not None:
            t = t @ self.coef + self.intercept
        t = np.clip(t.ravel(), self.x_min, self.x_max)
        return np.interp(t, self.x0, self.y0)


class Calibrator:
    def __init__(self, per_year: dict[str, list[SimpleIsotonic]]):
        self.per_year = per_year

    @classmethod
    def from_json(cls, path: Path) -> "Calibrator":
        raw = json.loads(Path(path).read_text())
        return cls({k: [SimpleIsotonic(**c) for c in v] for k, v in raw.items()})

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        """probs: (N, 6) raw sigmoid outputs -> calibrated (N, 6)."""
        out = np.zeros_like(probs)
        for y in range(probs.shape[1]):
            cals = self.per_year[f"Year{y + 1}"]
            out[:, y] = np.mean([c(probs[:, y]) for c in cals], axis=0)
        return out


def load_ensemble(directory: Path = DEFAULT_DIR, download: bool = False):
    directory = Path(directory)
    if download:
        download_checkpoints(directory)
    models = [load_checkpoint(directory / f"{c}.ckpt") for c in CHECKPOINT_IDS
              if (directory / f"{c}.ckpt").exists()]
    # sybil_1..5 correspond to CHECKPOINT_IDS in order; "ensemble" to their mean.
    calibrators = {}
    for key, fname in [("ensemble", "sybil_ensemble_simple_calibrator.json")] + \
            [(i, f"sybil_{i + 1}_simple_calibrator.json") for i in range(len(CHECKPOINT_IDS))]:
        if (directory / fname).exists():
            calibrators[key] = Calibrator.from_json(directory / fname)
    return models, calibrators

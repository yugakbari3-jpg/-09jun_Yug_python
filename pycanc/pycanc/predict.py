"""Inference engine: ensemble, calibration, attention maps and hotspots."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .model import PyCancNet
from .preprocess import Prepared
from .weights import DEFAULT_DIR, CHECKPOINT_IDS, load_ensemble

STAGES = ["stem", "layer1", "layer2", "layer3", "layer4", "pool", "head"]


class PyCancEngine:
    def __init__(self, checkpoint_dir: Path | str = DEFAULT_DIR, device: str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lock = threading.Lock()
        self.reload()

    # ------------------------------------------------------------------ #
    def reload(self):
        models, cals = load_ensemble(self.checkpoint_dir)
        self.official = len(models) > 0
        if not models:
            torch.manual_seed(0)
            models = [PyCancNet().eval()]
        self.models = [m.to(self.device) for m in models]
        self.calibrators = cals

    @property
    def info(self) -> dict:
        return {
            "official_weights": self.official,
            "num_models": len(self.models),
            "calibrated": bool(self.calibrators),
            "device": str(self.device),
            "checkpoint_dir": str(self.checkpoint_dir),
            "expected_checkpoints": len(CHECKPOINT_IDS),
            "threads": torch.get_num_threads(),
        }

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict(self, prep: Prepared, n_models: int | None = None,
                progress: Callable[[dict], None] | None = None) -> dict:
        n = min(n_models or len(self.models), len(self.models))
        x = prep.tensor.to(self.device)
        raw, attn, vol_attn = [], [], []
        with self.lock:
            for i, model in enumerate(self.models[:n]):
                hooks = []
                if progress:
                    def mk(stage, i=i):
                        return lambda *a: progress({"model": i, "of": n, "stage": stage})
                    enc = model.image_encoder
                    for j, stage in enumerate(STAGES[:5]):
                        hooks.append(enc[j].register_forward_pre_hook(mk(stage)))
                    hooks.append(model.pool.register_forward_pre_hook(mk("pool")))
                    hooks.append(model.prob_of_failure_layer.register_forward_pre_hook(mk("head")))
                out = model(x)
                for h in hooks:
                    h.remove()
                raw.append(torch.sigmoid(out["logit"]).float().cpu().numpy()[0])
                T = out["image_attention_1"].shape[1]
                side = int(round(out["image_attention_1"].shape[2] ** 0.5))
                img_a = torch.exp(out["image_attention_1"]).view(T, side, side)
                vol_a = torch.exp(out["volume_attention_1"]).view(T)
                attn.append((img_a * vol_a[:, None, None]).float().cpu().numpy())
                vol_attn.append(vol_a.float().cpu().numpy())

        raw = np.stack(raw)                       # (n, 6)
        mean = raw.mean(0, keepdims=True)
        # The ensemble calibrator was fit on the 5-model mean and each sybil_i
        # calibrator on a single model; other subset sizes reuse the ensemble one.
        cal = self.calibrators.get(0 if n == 1 else "ensemble")
        calibrated = cal(mean)[0] if cal is not None else mean[0]
        A = np.mean(attn, 0)
        A = A / (A.max() + 1e-12)
        V = np.mean(vol_attn, 0)

        return {
            "risk": [float(v) for v in calibrated],
            "raw_ensemble": [float(v) for v in mean[0]],
            "per_model": raw.round(6).tolist(),
            "attention": A.astype(np.float32),          # (25, 16, 16) normalised
            "slice_attention": (V / (V.max() + 1e-12)).astype(np.float32).tolist(),
            "hotspots": hotspots(A),
            "models_used": n,
            "official_weights": self.official,
            "calibrated": cal is not None,
        }


def hotspots(A: np.ndarray, k: int = 5, depth: int = 200, size: int = 256) -> list[dict]:
    """Local maxima of the low-res attention grid, mapped to model-input voxels."""
    T, H, W = A.shape
    pad = np.pad(A, 1, constant_values=-1)
    neigh = np.max(np.stack([pad[1 + dz:T + 1 + dz, 1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
                             for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                             if (dz, dy, dx) != (0, 0, 0)]), 0)
    peaks = np.argwhere(A >= neigh)
    peaks = sorted(peaks.tolist(), key=lambda p: -A[tuple(p)])[:k]
    return [{"z": int((t + 0.5) * depth / T), "y": int((y + 0.5) * size / H), "x": int((x + 0.5) * size / W),
             "score": float(A[t, y, x])} for t, y, x in peaks]

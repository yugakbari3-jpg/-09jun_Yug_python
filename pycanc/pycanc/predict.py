"""Inference engine: device selection, ensemble, test-time augmentation,
calibration, uncertainty, attention maps and hotspots."""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .model import PyCancNet
from .preprocess import Prepared
from .weights import CHECKPOINT_IDS, DEFAULT_DIR, Calibrator, load_ensemble

STAGES = ["stem", "layer1", "layer2", "layer3", "layer4", "pool", "head"]
LOCAL_CALIBRATOR = "local_calibrator.json"

# Test-time augmentations: (name, shift_y px, shift_x px, rotation deg) on the 256 grid.
# Small in-plane shifts / rotations the network is robust to; the identity pass
# always comes first and is the one used for attention maps.
TTA = [("identity", 0, 0, 0), ("shift+", 6, 6, 0), ("shift-", -6, -6, 0), ("rot+", 0, 0, 5), ("rot-", 0, 0, -5)]


def pick_device(requested: str | None = None) -> torch.device:
    """cuda > Apple GPU (mps, if 3D convs work) > cpu."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            conv = torch.nn.Conv3d(1, 2, 3).to("mps")
            conv(torch.zeros(1, 1, 4, 4, 4, device="mps")).cpu()
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def augment(x: torch.Tensor, shift_y: int, shift_x: int, rot_deg: float) -> torch.Tensor:
    """In-plane affine on (1, 3, T, H, W); pads with 0 like the reference CropOrPad."""
    if not (shift_y or shift_x or rot_deg):
        return x
    _, C, T, H, W = x.shape
    a = math.radians(rot_deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), -2 * shift_x / W],
                          [math.sin(a), math.cos(a), -2 * shift_y / H]], dtype=x.dtype, device=x.device)
    sl = x[0, :1].permute(1, 0, 2, 3)                                   # T, 1, H, W (channels are identical)
    grid = F.affine_grid(theta.expand(T, 2, 3), sl.shape, align_corners=False)
    out = F.grid_sample(sl, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.permute(1, 0, 2, 3)[None].expand(1, C, T, H, W).contiguous()


class PyCancEngine:
    def __init__(self, checkpoint_dir: Path | str = DEFAULT_DIR, device: str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = pick_device(device or os.environ.get("PYCANC_DEVICE"))
        self.lock = threading.Lock()
        self.reload()

    # ------------------------------------------------------------------ #
    def reload(self):
        models, cals = load_ensemble(self.checkpoint_dir)
        self.official = len(models) == len(CHECKPOINT_IDS)
        self.has_weights = len(models) > 0
        if not models:
            torch.manual_seed(0)
            models = [PyCancNet().eval()]
        self.models = [m.to(self.device) for m in models]
        self.calibrators = cals if self.official else {}
        local = self.checkpoint_dir / LOCAL_CALIBRATOR
        self.local_calibrator = Calibrator.from_json(local) if local.exists() else None
        from .fusion import load_fusion
        self.fusion = load_fusion(self.checkpoint_dir)

    @property
    def info(self) -> dict:
        return {
            "official_weights": self.official,
            "num_models": len(self.models),
            "calibrated": bool(self.calibrators) or self.local_calibrator is not None,
            "local_calibration": self.local_calibrator is not None,
            "fusion": self.fusion is not None,
            "device": self.device.type,
            "half_precision": self.device.type == "cuda",
            "checkpoint_dir": str(self.checkpoint_dir),
            "expected_checkpoints": len(CHECKPOINT_IDS),
            "threads": torch.get_num_threads(),
            "max_tta": len(TTA),
        }

    # ------------------------------------------------------------------ #
    def _forward(self, model, x):
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                return model(x)
        return model(x)

    @torch.inference_mode()
    def predict(self, prep: Prepared, n_models: int | None = None, tta: int = 1,
                progress: Callable[[dict], None] | None = None) -> dict:
        n = min(n_models or len(self.models), len(self.models))
        tta = max(1, min(int(tta or 1), len(TTA)))
        x0 = prep.tensor.to(self.device)
        raw = np.zeros((n, tta, 6))
        attn, vol_attn = [], []
        with self.lock:
            for k, (_, sy, sx, rot) in enumerate(TTA[:tta]):
                x = augment(x0, sy, sx, rot)
                for i, model in enumerate(self.models[:n]):
                    hooks = []
                    if progress:
                        def mk(stage, i=i, k=k):
                            return lambda *a: progress({"model": i, "of": n, "pass": k, "passes": tta, "stage": stage})
                        enc = model.image_encoder
                        for j, stage in enumerate(STAGES[:5]):
                            hooks.append(enc[j].register_forward_pre_hook(mk(stage)))
                        hooks.append(model.pool.register_forward_pre_hook(mk("pool")))
                        hooks.append(model.prob_of_failure_layer.register_forward_pre_hook(mk("head")))
                    out = self._forward(model, x)
                    for h in hooks:
                        h.remove()
                    raw[i, k] = torch.sigmoid(out["logit"].float()).cpu().numpy()[0]
                    if k == 0:
                        T = out["image_attention_1"].shape[1]
                        side = int(round(out["image_attention_1"].shape[2] ** 0.5))
                        img_a = torch.exp(out["image_attention_1"].float()).view(T, side, side)
                        vol_a = torch.exp(out["volume_attention_1"].float()).view(T)
                        attn.append((img_a * vol_a[:, None, None]).cpu().numpy())
                        vol_attn.append(vol_a.cpu().numpy())

        per_model = raw.mean(1)                                   # (n, 6), TTA-averaged
        mean = per_model.mean(0, keepdims=True)
        calibrated, method = self._calibrate(per_model, mean)
        per_model_cal = np.stack([self._calibrate_one(i, per_model[i:i + 1]) for i in range(n)])

        # uncertainty: spread between ensemble members + spread between TTA passes
        model_std = per_model_cal.std(0) if n > 1 else np.zeros(6)
        tta_cal = np.stack([self._calibrate(raw[:, k], raw[:, k].mean(0, keepdims=True))[0] for k in range(tta)])
        tta_std = tta_cal.std(0) if tta > 1 else np.zeros(6)
        std = np.sqrt(model_std ** 2 + tta_std ** 2)

        A = np.mean(attn, 0)
        A = A / (A.max() + 1e-12)
        V = np.mean(vol_attn, 0)
        return {
            "risk": [float(v) for v in calibrated],
            "risk_low": [float(max(0.0, c - 1.96 * s)) for c, s in zip(calibrated, std)],
            "risk_high": [float(min(1.0, c + 1.96 * s)) for c, s in zip(calibrated, std)],
            "uncertainty": {"std": std.tolist(), "model_std": model_std.tolist(), "tta_std": tta_std.tolist()},
            "raw_ensemble": [float(v) for v in mean[0]],
            "per_model": per_model.round(6).tolist(),
            "per_model_calibrated": per_model_cal.round(6).tolist(),
            "attention": A.astype(np.float32),          # (25, 16, 16) normalised
            "slice_attention": (V / (V.max() + 1e-12)).astype(np.float32).tolist(),
            "hotspots": hotspots(A),
            "models_used": n,
            "tta_passes": tta,
            "official_weights": self.official,
            "calibrated": method != "none",
            "calibration": method,
            "device": self.device.type,
        }

    # ------------------------------------------------------------------ #
    def _calibrate_one(self, i: int, p: np.ndarray) -> np.ndarray:
        cal = self.calibrators.get(i)
        return cal(p)[0] if cal is not None else p[0]

    def _calibrate(self, per_model: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, str]:
        """Local calibrator if fitted; the official ensemble calibrator for the
        full 5-model mean; otherwise each member's own calibrator, averaged."""
        n = per_model.shape[0]
        if self.local_calibrator is not None:
            return self.local_calibrator(mean)[0], "local"
        if not self.calibrators:
            return mean[0], "none"
        if n == len(CHECKPOINT_IDS) and "ensemble" in self.calibrators:
            return self.calibrators["ensemble"](mean)[0], "ensemble"
        return np.mean([self._calibrate_one(i, per_model[i:i + 1]) for i in range(n)], 0), "per-model"


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

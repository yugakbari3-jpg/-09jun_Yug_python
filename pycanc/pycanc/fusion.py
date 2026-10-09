"""
Image + clinical fusion: per-year logistic regression on
    [logit(PyCanc risk_t), logit(PLCOm2012 6y risk)]
fitted on local labelled data by `python -m pycanc.evaluate --fit-fusion`.

No coefficients ship with PyCanc: a combined score is only shown once you have
fitted one on outcomes from your own population.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FUSION_FILE = "fusion.json"


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 100) -> np.ndarray:
    """Newton-Raphson logistic regression with a small ridge penalty. Returns [b0, b...]."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


class Fusion:
    def __init__(self, weights: dict[str, list[float]], meta: dict | None = None):
        self.weights = {k: np.asarray(v, float) for k, v in weights.items()}
        self.meta = meta or {}

    def __call__(self, image_risk: list[float], plco_risk: float) -> list[float]:
        out = []
        for t, r in enumerate(image_risk):
            w = self.weights[f"Year{t + 1}"]
            z = w[0] + w[1] * _logit(r) + w[2] * _logit(plco_risk)
            out.append(float(1 / (1 + np.exp(-z))))
        return out

    def save(self, directory: Path):
        Path(directory, FUSION_FILE).write_text(json.dumps(
            {"weights": {k: v.tolist() for k, v in self.weights.items()}, "meta": self.meta}, indent=2))


def load_fusion(directory: Path) -> Fusion | None:
    f = Path(directory) / FUSION_FILE
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return Fusion(d["weights"], d.get("meta"))

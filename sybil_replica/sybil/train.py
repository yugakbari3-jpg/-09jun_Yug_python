"""
Training / fine-tuning SybilNet with the objectives from the paper.

    loss = masked survival BCE over years 1..6
         + λ_img * KL(image_attention || nodule-box distribution per slice)
         + λ_vol * KL(volume_attention || annotated-area distribution over slices)

The attention terms only apply to scans that come with a nodule mask
(in NLST, radiologist bounding boxes on the cancer-positive subset).

Data
----
A CSV with columns
    path                    DICOM folder / .zip / .nii(.gz) / .npz
    years_to_cancer         integer years from scan to diagnosis (blank / -1 if none)
    years_to_last_followup  integer years of cancer-free follow-up
    mask_path   (optional)  .npz with `mask` (Z, Y, X) on the same grid as the CT

    python -m sybil.train --csv train.csv --val-csv val.csv --epochs 10
    python -m sybil.train --demo            # tiny synthetic run to check the loop
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import SybilNet
from .preprocess import CTVolume, NUM_IMAGES, _to_model_grid, load_any, preprocess

MAX_FOLLOWUP = 6


def survival_labels(years_to_cancer: float | None, years_to_last_followup: float, max_followup=MAX_FOLLOWUP):
    """Exactly the reference labelling: y_seq is 1 from the diagnosis year on;
    y_mask hides years after censoring for cancer-free patients."""
    has_cancer = years_to_cancer is not None and years_to_cancer >= 0 and years_to_cancer < max_followup
    y_seq = np.zeros(max_followup, np.float32)
    if has_cancer:
        t = int(years_to_cancer)
        y_seq[t:] = 1
    else:
        t = int(min(years_to_last_followup, max_followup - 1))
    y_mask = np.array([1] * (t + 1) + [0] * (max_followup - t - 1), np.float32)
    return y_seq, y_mask, float(has_cancer), t


def survival_loss(logit, y_seq, y_mask):
    return F.binary_cross_entropy_with_logits(logit, y_seq, weight=y_mask, reduction="sum") / torch.clamp(y_mask.sum(), min=1)


def annotation_loss(out, ann, has_ann, lam_img=1.0, lam_vol=1.0):
    """ann: (B, 1, 200, 256, 256) binary nodule mask; has_ann: (B,)"""
    if has_ann.sum() == 0:
        return torch.zeros((), device=ann.device)
    B, _, N, H, W = out["activ"].shape
    total = 0.0
    # spatial attention inside each slice
    gold = F.interpolate(ann, (N, H, W), mode="area") * has_ann[:, None, None, None, None]
    area = gold.sum(dim=(-1, -2), keepdim=True)
    gold = (gold / torch.where(area == 0, torch.ones_like(area), area)).view(B, N, -1)
    n = max(1, int((gold.view(B * N, -1).sum(-1) > 0).sum()))
    pred = out["image_attention_1"] * has_ann[:, None, None]
    total = total + lam_img * (F.kl_div(pred, gold, reduction="none") * (gold > 0)).sum() / n
    # attention across slices (both pooling branches)
    areas = ann.sum(dim=(-1, -2))[:, 0] * has_ann[:, None]                      # B, 200
    areas = F.interpolate(areas[:, None], N, mode="linear", align_corners=True)[:, 0]
    s = areas.sum(-1, keepdim=True)
    gold_v = areas / torch.where(s == 0, torch.ones_like(s), s)
    n = max(1, int((gold_v.sum(-1) > 0).sum()))
    for k in (1, 2):
        pred = out[f"volume_attention_{k}"] * has_ann[:, None]
        total = total + lam_vol * (F.kl_div(pred, gold_v, reduction="none") * (gold_v > 0)).sum() / n
    return total


# --------------------------------------------------------------------------- #
class CTDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _ct(self, row) -> CTVolume:
        return row["ct"] if "ct" in row else load_any(row["path"])

    def __getitem__(self, i):
        row = self.rows[i]
        ct = self._ct(row)
        prep = preprocess(ct)
        ytc = row.get("years_to_cancer")
        ytc = float(ytc) if ytc not in (None, "", "-1") else None
        y_seq, y_mask, y, t = survival_labels(ytc, float(row.get("years_to_last_followup") or 0))
        mask = row.get("mask")
        if mask is None and row.get("mask_path"):
            mask = np.load(row["mask_path"])["mask"]
        has = mask is not None
        ann = _to_model_grid(mask.astype(np.float32), ct.spacing, 0.0)[0] if has else torch.zeros(NUM_IMAGES, 256, 256)
        return {
            "x": prep.tensor[0],
            "y_seq": torch.from_numpy(y_seq), "y_mask": torch.from_numpy(y_mask),
            "y": torch.tensor(y), "time_at_event": torch.tensor(t),
            "ann": (ann > 0.5).float()[None], "has_ann": torch.tensor(float(has)),
        }


def read_csv(path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def demo_rows(n=5, seed=0) -> list[dict]:
    from .phantom import make_phantom

    rows, rng = [], np.random.default_rng(seed)
    for i in range(n):
        nod = bool(i % 2)
        mm = float(rng.uniform(8, 24)) if nod else 0
        ct = make_phantom(seed=int(rng.integers(1e6)), nodule=nod, nodule_mm=mm or 10, shape=(64, 128, 128),
                          spacing=(5.0, 2.8125, 2.8125))
        mask = None
        if nod:
            c = np.array(ct.meta["nodule"]["center_mm"])
            Z, Y, X = ct.hu.shape
            zz, yy, xx = np.meshgrid(np.arange(Z) * 5.0, (np.arange(Y) - Y / 2) * 2.8125, (np.arange(X) - X / 2) * 2.8125, indexing="ij")
            mask = (np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < mm / 2).astype(np.uint8)
        rows.append({"ct": ct, "mask": mask, "years_to_cancer": (1 if mm > 16 else 3) if nod else None,
                     "years_to_last_followup": 6})
    return rows


# --------------------------------------------------------------------------- #
def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    num = den = 0.0
    for i in range(len(risk)):
        if not event[i]:
            continue
        for j in range(len(risk)):
            if time[j] > time[i] or (time[j] == time[i] and not event[j]):
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num / den if den else float("nan")


def scale_input(x, s: float):
    return x if s == 1.0 else F.interpolate(x, scale_factor=s, mode="trilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, loader, device, s=1.0):
    model.eval()
    probs, times, events = [], [], []
    for b in loader:
        out = model(scale_input(b["x"].to(device), s))
        probs.append(torch.sigmoid(out["logit"]).cpu())
        times.append(b["time_at_event"]); events.append(b["y"])
    p, t, e = torch.cat(probs).numpy(), torch.cat(times).numpy(), torch.cat(events).numpy()
    return {"c_index_6y": concordance_index(p[:, -1], t, e)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv"); ap.add_argument("--val-csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--init", help="start from an official checkpoint (.ckpt) to fine-tune")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=5, help="gradient accumulation steps (paper: effective batch 10)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda-img", type=float, default=1.0)
    ap.add_argument("--lambda-vol", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    ap.add_argument("--input-scale", type=float, default=1.0,
                    help="resample inputs by this factor (e.g. 0.5 for a quick CPU run; the paper uses 1.0)")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.demo:
        rows = demo_rows()
        train_rows, val_rows = rows[:3], rows[3:]
        a.epochs, a.batch_size, a.accum, a.workers = min(a.epochs, 1), 1, 1, 0
        a.input_scale = min(a.input_scale, 0.5)
    else:
        if not a.csv:
            ap.error("--csv is required (or use --demo)")
        train_rows, val_rows = read_csv(a.csv), read_csv(a.val_csv) if a.val_csv else []

    if a.init:
        from .weights import load_checkpoint
        model = load_checkpoint(Path(a.init))
        model.dropout.p = a.dropout
    else:
        model = SybilNet(dropout=a.dropout, pretrained_encoder=not a.demo)
    model.to(device)

    tl = DataLoader(CTDataset(train_rows), batch_size=a.batch_size, shuffle=True, num_workers=a.workers)
    vl = DataLoader(CTDataset(val_rows), batch_size=a.batch_size, num_workers=a.workers) if val_rows else None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    best = -math.inf

    for epoch in range(a.epochs):
        model.train()
        opt.zero_grad()
        for step, b in enumerate(tl):
            with torch.autocast(device.type, enabled=a.amp and device.type == "cuda"):
                out = model(scale_input(b["x"].to(device), a.input_scale))
                ls = survival_loss(out["logit"], b["y_seq"].to(device), b["y_mask"].to(device))
                la = annotation_loss(out, b["ann"].to(device), b["has_ann"].to(device), a.lambda_img, a.lambda_vol)
                loss = (ls + la) / a.accum
            scaler.scale(loss).backward()
            if (step + 1) % a.accum == 0 or step + 1 == len(tl):
                scaler.step(opt); scaler.update(); opt.zero_grad()
            print(f"epoch {epoch} step {step}  survival {ls.item():.4f}  attention {float(la):.4f}", flush=True)
        if vl:
            m = evaluate(model, vl, device, a.input_scale)
            print(f"epoch {epoch} val {m}", flush=True)
            score = m["c_index_6y"] if not math.isnan(m["c_index_6y"]) else -epoch
        else:
            score = epoch
        torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, out_dir / "last.ckpt")
        if score > best:
            best = score
            torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, out_dir / "best.ckpt")
    print(f"done; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()

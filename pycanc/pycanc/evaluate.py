"""
Measure PyCanc on labelled scans, and optionally fit a local calibration
and an image + clinical fusion model.

CSV columns (same as training):
    path, years_to_cancer, years_to_last_followup
optional clinical columns (enable PLCOm2012 + fusion):
    age, smoking_status, cigarettes_per_day, smoking_years, years_since_quit,
    education, bmi, copd, personal_cancer_history, family_lung_cancer, race

    python -m pycanc.evaluate --csv test.csv --models 5 --out results/
    python -m pycanc.evaluate --predictions results/predictions.csv --fit-calibration --fit-fusion

Predictions are cached in <out>/predictions.csv, so metrics and fits can be
recomputed without re-running the network.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .clinical import Patient, plcom2012
from .fusion import Fusion, _logit, fit_logistic
from .train import concordance_index, survival_labels

YEARS = 6
CLINICAL = ["age", "smoking_status", "cigarettes_per_day", "smoking_years", "years_since_quit",
            "education", "bmi", "copd", "personal_cancer_history", "family_lung_cancer", "race"]


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv)); r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_ci(fn, *arrays, n=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(arrays[0]), len(arrays[0]))
        v = fn(*[a[idx] for a in arrays])
        if not math.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def year_labels(rows) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """y (N, 6), mask (N, 6), event (N,), time (N,) from the reference labelling."""
    Y, M, E, T = [], [], [], []
    for r in rows:
        ytc = r.get("years_to_cancer")
        ytc = float(ytc) if ytc not in (None, "", "-1") else None
        y, m, e, t = survival_labels(ytc, float(r.get("years_to_last_followup") or 0))
        Y.append(y); M.append(m); E.append(e); T.append(t)
    return np.array(Y), np.array(M), np.array(E), np.array(T)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for k in range(bins):
        sel = (p >= edges[k]) & ((p < edges[k + 1]) if k < bins - 1 else (p <= edges[k + 1]))
        if sel.sum():
            out.append({"bin": k + 1, "n": int(sel.sum()), "predicted": float(p[sel].mean()), "observed": float(y[sel].mean())})
    return out


def metrics(risk: np.ndarray, Y, M, E, T, ci=True) -> dict:
    res = {"per_year": []}
    for t in range(YEARS):
        sel = M[:, t] == 1
        p, y = risk[sel, t], Y[sel, t]
        a = auc(p, y)
        row = {"year": t + 1, "n": int(sel.sum()), "cancers": int(y.sum()), "auc": a,
               "brier": float(np.mean((p - y) ** 2)) if len(p) else float("nan"),
               "mean_predicted": float(p.mean()) if len(p) else float("nan"),
               "observed_rate": float(y.mean()) if len(p) else float("nan")}
        if ci and y.sum() and (1 - y).sum():
            row["auc_95ci"] = bootstrap_ci(auc, p, y)
        res["per_year"].append(row)
    res["c_index_6y"] = concordance_index(risk[:, -1], T, E)
    sel = M[:, -1] == 1
    if sel.sum():
        res["calibration_6y"] = calibration_table(risk[sel, -1], Y[sel, -1])
    return res


# --------------------------------------------------------------------------- #
#  Isotonic regression (pool adjacent violators) -> local calibrator
# --------------------------------------------------------------------------- #
def isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float)
    blocks = [[v, 1.0] for v in ys]  # value, weight
    idx = [[i] for i in range(len(ys))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            w = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [(blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / w, w]
            idx[i] += idx.pop(i + 1); blocks.pop(i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.empty(len(ys))
    for (v, _), ids in zip(blocks, idx):
        fitted[ids] = v
    ux = np.unique(xs)
    uy = np.array([fitted[xs == u].mean() for u in ux])
    return ux, uy


def fit_local_calibrator(raw: np.ndarray, Y, M) -> dict:
    out = {}
    for t in range(YEARS):
        sel = M[:, t] == 1
        x0, y0 = isotonic_fit(raw[sel, t], Y[sel, t])
        out[f"Year{t + 1}"] = [{"coef": None, "intercept": None, "x0": x0.tolist(), "y0": y0.tolist(),
                                "x_min": float(x0.min()), "x_max": float(x0.max())}]
    return out


# --------------------------------------------------------------------------- #
def patient_from_row(r: dict) -> Patient | None:
    if not r.get("age"):
        return None
    b = lambda k: str(r.get(k, "")).strip().lower() in ("1", "true", "yes", "y")
    f = lambda k, d: float(r[k]) if r.get(k) not in (None, "") else d
    return Patient(age=float(r["age"]), smoking_status=(r.get("smoking_status") or "current").lower(),
                   cigarettes_per_day=f("cigarettes_per_day", 20), smoking_years=f("smoking_years", 30),
                   years_since_quit=f("years_since_quit", 0), education=r.get("education") or "high_school",
                   bmi=f("bmi", 27), copd=b("copd"), personal_cancer_history=b("personal_cancer_history"),
                   family_lung_cancer=b("family_lung_cancer"), race=(r.get("race") or "white").lower())


def run_predictions(rows, n_models, tta, out_csv: Path):
    from .predict import PyCancEngine
    from .preprocess import load_any, preprocess

    engine = PyCancEngine()
    if not engine.official:
        print("WARNING: official weights not found; metrics will be meaningless.")
    fields = list(rows[0].keys()) + [f"raw_{t + 1}" for t in range(YEARS)] + [f"risk_{t + 1}" for t in range(YEARS)]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            try:
                res = engine.predict(preprocess(load_any(r["path"])), n_models, tta)
            except Exception as e:
                print(f"  [{i + 1}/{len(rows)}] SKIP {r['path']}: {e}")
                continue
            r = dict(r)
            r.update({f"raw_{t + 1}": res["raw_ensemble"][t] for t in range(YEARS)})
            r.update({f"risk_{t + 1}": res["risk"][t] for t in range(YEARS)})
            w.writerow(r); f.flush()
            print(f"  [{i + 1}/{len(rows)}] {Path(r['path']).name}: 6y {res['risk'][5] * 100:.2f}%")


def print_report(name, m):
    print(f"\n== {name} ==")
    print(" year     n  cancers    AUC   (95% CI)          Brier   pred%  obs%")
    for r in m["per_year"]:
        ci = r.get("auc_95ci", (float("nan"), float("nan")))
        print(f"  {r['year']}  {r['n']:6d}  {r['cancers']:6d}  {r['auc']:.3f}  ({ci[0]:.3f}-{ci[1]:.3f})  "
              f"{r['brier']:.4f}  {r['mean_predicted'] * 100:5.2f}  {r['observed_rate'] * 100:5.2f}")
    print(f"  6-year C-index: {m['c_index_6y']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="labelled scans to run the network on")
    ap.add_argument("--predictions", help="reuse a predictions.csv from an earlier run")
    ap.add_argument("--models", type=int, default=None)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--out", default="pycanc_eval")
    ap.add_argument("--fit-calibration", action="store_true", help="fit an isotonic calibrator on these outcomes")
    ap.add_argument("--fit-fusion", action="store_true", help="fit image + PLCOm2012 fusion on these outcomes")
    ap.add_argument("--save-to", default=None, help="where to save fitted files (default: checkpoint dir)")
    a = ap.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pred_csv = Path(a.predictions) if a.predictions else out / "predictions.csv"
    if not a.predictions:
        if not a.csv:
            ap.error("--csv or --predictions is required")
        with open(a.csv) as f:
            run_predictions(list(csv.DictReader(f)), a.models, a.tta, pred_csv)
    with open(pred_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("no predictions")

    Y, M, E, T = year_labels(rows)
    raw = np.array([[float(r[f"raw_{t + 1}"]) for t in range(YEARS)] for r in rows])
    risk = np.array([[float(r[f"risk_{t + 1}"]) for t in range(YEARS)] for r in rows])
    report = {"n_scans": len(rows), "cancers_within_6y": int(E.sum()), "pycanc": metrics(risk, Y, M, E, T)}
    print_report("PyCanc (image)", report["pycanc"])

    patients = [patient_from_row(r) for r in rows]
    plco = np.array([plcom2012(p)["risk"] if p else np.nan for p in patients], dtype=float)
    has_plco = ~np.isnan(plco)
    if has_plco.sum() >= 10:
        sub = lambda arr: arr[has_plco]
        plco_mat = np.repeat(plco[has_plco, None], YEARS, 1)
        report["plcom2012"] = metrics(plco_mat, sub(Y), sub(M), sub(E), sub(T))
        print_report("PLCOm2012 (clinical, same 6-year score for every year)", report["plcom2012"])

    save_dir = Path(a.save_to) if a.save_to else None
    if a.fit_calibration:
        from .weights import DEFAULT_DIR
        from .predict import LOCAL_CALIBRATOR
        d = save_dir or DEFAULT_DIR
        Path(d, LOCAL_CALIBRATOR).write_text(json.dumps(fit_local_calibrator(raw, Y, M)))
        print(f"\nSaved local calibrator -> {Path(d, LOCAL_CALIBRATOR)}  (fit on {len(rows)} scans; "
              "evaluate on a *different* set to judge it)")
    if a.fit_fusion:
        if has_plco.sum() < 30:
            raise SystemExit("fusion needs clinical columns for at least 30 scans")
        from .weights import DEFAULT_DIR
        weights = {}
        for t in range(YEARS):
            sel = has_plco & (M[:, t] == 1)
            X = np.column_stack([_logit(risk[sel, t]), _logit(plco[sel])])
            weights[f"Year{t + 1}"] = fit_logistic(X, Y[sel, t]).tolist()
        fus = Fusion(weights, {"n": int(has_plco.sum()), "source": str(pred_csv)})
        fused = np.array([fus(risk[i].tolist(), plco[i]) for i in np.where(has_plco)[0]])
        report["fusion_in_sample"] = metrics(fused, Y[has_plco], M[has_plco], E[has_plco], T[has_plco], ci=False)
        print_report("Fusion (in-sample, optimistic)", report["fusion_in_sample"])
        fus.save(save_dir or DEFAULT_DIR)
        print(f"Saved fusion model -> {Path(save_dir or DEFAULT_DIR, 'fusion.json')}")

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nReport -> {out / 'report.json'}")


if __name__ == "__main__":
    main()

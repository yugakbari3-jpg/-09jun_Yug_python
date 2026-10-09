"""
Command line interface.

    python -m pycanc.cli download                 # fetch the official MIT ensemble (~700 MB)
    python -m pycanc.cli predict scan.zip         # DICOM zip / folder, NIfTI or .npz
    python -m pycanc.cli predict --phantom        # synthetic chest CT with a nodule
    python -m pycanc.cli predict scan.zip --models 5 --tta 3 --age 64 --cigs 20 --smoking-years 40
    python -m pycanc.cli serve                    # web workstation on http://127.0.0.1:8000
    python -m pycanc.evaluate --csv test.csv      # AUC / C-index / calibration on labelled scans
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pycanc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download official checkpoints")
    d.add_argument("--dir", default=None)

    p = sub.add_parser("predict", help="predict 1..6 year lung cancer risk")
    p.add_argument("path", nargs="?")
    p.add_argument("--phantom", action="store_true", help="use a synthetic phantom instead of a file")
    p.add_argument("--models", type=int, default=None, help="ensemble size (default: all)")
    p.add_argument("--dir", default=None, help="checkpoint directory")
    p.add_argument("--tta", type=int, default=1, help="test-time augmentation passes (1-5)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    p.add_argument("--json", action="store_true")
    c = p.add_argument_group("clinical history (adds PLCOm2012)")
    c.add_argument("--age", type=float)
    c.add_argument("--smoking", default="current", choices=["current", "former", "never"])
    c.add_argument("--cigs", type=float, default=20, help="cigarettes per day")
    c.add_argument("--smoking-years", type=float, default=30)
    c.add_argument("--years-quit", type=float, default=0)
    c.add_argument("--bmi", type=float, default=27)
    c.add_argument("--copd", action="store_true")
    c.add_argument("--family-history", action="store_true")

    s = sub.add_parser("serve", help="start the web workstation")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = ap.parse_args(argv)

    if a.cmd == "download":
        from . import weights
        print("Saved to", weights.download_checkpoints(a.dir or weights.DEFAULT_DIR))
        return

    if a.cmd == "serve":
        import os
        os.environ["HOST"], os.environ["PORT"] = a.host, str(a.port)
        from web.server import main as serve
        serve()
        return

    from .predict import PyCancEngine
    from .preprocess import load_any, preprocess
    from . import weights

    if not a.path and not a.phantom:
        ap.error("give a path or --phantom")
    if a.phantom:
        from .phantom import make_phantom
        ct = make_phantom()
    else:
        ct = load_any(a.path)
    engine = PyCancEngine(a.dir or weights.DEFAULT_DIR, device=a.device)
    if not engine.official:
        print("WARNING: official checkpoints not found -> untrained weights; run `python -m pycanc.cli download`.",
              file=sys.stderr)
    t = time.time()
    from .preprocess import quality_checks
    for w in quality_checks(ct):
        print("NOTE:", w, file=sys.stderr)
    res = engine.predict(preprocess(ct), a.models, a.tta,
                         progress=lambda d: print(f"\r  pass {d['pass'] + 1}/{d['passes']} · model {d['model'] + 1}/{d['of']}  {d['stage']:<8}", end="", file=sys.stderr))
    print(file=sys.stderr)
    res.pop("attention")
    res["seconds"] = round(time.time() - t, 1)
    if a.age:
        from .clinical import Patient, plcom2012
        clin = plcom2012(Patient(age=a.age, smoking_status=a.smoking, cigarettes_per_day=a.cigs, smoking_years=a.smoking_years,
                                 years_since_quit=a.years_quit, bmi=a.bmi, copd=a.copd, family_lung_cancer=a.family_history))
        clin.pop("inputs", None)
        if engine.fusion is not None and clin["risk"] is not None:
            clin["fused"] = engine.fusion(res["risk"], clin["risk"])
        res["clinical"] = clin
    if a.json:
        print(json.dumps(res, indent=2))
        return
    print(f"\nPyCanc · {res['models_used']} model(s) × {res['tta_passes']} TTA · {res['calibration']} calibration · "
          f"{res['device']} · {res['seconds']} s")
    spread = res["models_used"] > 1 or res["tta_passes"] > 1
    for i, r in enumerate(res["risk"]):
        bar = "█" * max(1, round(r * 200)) if r > 0 else ""
        rng = f"  ({res['risk_low'][i] * 100:.2f}–{res['risk_high'][i] * 100:.2f})" if spread else ""
        print(f"  year {i + 1}: {r * 100:6.2f}%{rng}  {bar[:50]}")
    if "clinical" in res:
        c = res["clinical"]
        if c["risk"] is None:
            print("  PLCOm2012:", c["note"])
        else:
            print(f"  PLCOm2012 6-year clinical risk: {c['risk'] * 100:.2f}% "
                  f"({'≥' if c['eligible'] else '<'} 1.51% screening threshold)")
            if c.get("fused"):
                print(f"  combined image + clinical (local fusion): {c['fused'][5] * 100:.2f}%")
    print("  top attention:", ", ".join(f"slice {h['z']} ({'R' if h['x'] < 128 else 'L'}) {h['score']:.2f}" for h in res["hotspots"][:3]))


if __name__ == "__main__":
    main()

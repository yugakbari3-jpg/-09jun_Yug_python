"""
Command line interface.

    python -m sybil.cli download                 # fetch the official MIT ensemble (~700 MB)
    python -m sybil.cli predict scan.zip         # DICOM zip / folder, NIfTI or .npz
    python -m sybil.cli predict --phantom        # synthetic chest CT with a nodule
    python -m sybil.cli serve                    # web workstation on http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sybil", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download official checkpoints")
    d.add_argument("--dir", default=None)

    p = sub.add_parser("predict", help="predict 1..6 year lung cancer risk")
    p.add_argument("path", nargs="?")
    p.add_argument("--phantom", action="store_true", help="use a synthetic phantom instead of a file")
    p.add_argument("--models", type=int, default=None, help="ensemble size (default: all)")
    p.add_argument("--dir", default=None, help="checkpoint directory")
    p.add_argument("--json", action="store_true")

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

    from .predict import SybilEngine
    from .preprocess import load_any, preprocess
    from . import weights

    if not a.path and not a.phantom:
        ap.error("give a path or --phantom")
    if a.phantom:
        from .phantom import make_phantom
        ct = make_phantom()
    else:
        ct = load_any(a.path)
    engine = SybilEngine(a.dir or weights.DEFAULT_DIR)
    if not engine.official:
        print("WARNING: official checkpoints not found -> untrained weights; run `python -m sybil.cli download`.",
              file=sys.stderr)
    t = time.time()
    res = engine.predict(preprocess(ct), a.models,
                         progress=lambda d: print(f"\r  model {d['model'] + 1}/{d['of']}  {d['stage']:<8}", end="", file=sys.stderr))
    print(file=sys.stderr)
    res.pop("attention")
    res["seconds"] = round(time.time() - t, 1)
    if a.json:
        print(json.dumps(res, indent=2))
        return
    print(f"\nSybil · {res['models_used']} model(s) · {'calibrated' if res['calibrated'] else 'raw'} · {res['seconds']} s")
    for i, r in enumerate(res["risk"]):
        bar = "█" * max(1, round(r * 200)) if r > 0 else ""
        print(f"  year {i + 1}: {r * 100:6.2f}%  {bar[:60]}")
    print("  top attention:", ", ".join(f"slice {h['z']} ({'R' if h['x'] < 128 else 'L'}) {h['score']:.2f}" for h in res["hotspots"][:3]))


if __name__ == "__main__":
    main()

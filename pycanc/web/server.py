"""
PyCanc web server.

    python -m web.server            # http://localhost:8000
"""
from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pycanc import weights
from pycanc.phantom import make_phantom
from pycanc.predict import PyCancEngine
from pycanc.clinical import Patient, plcom2012
from pycanc.preprocess import load_any, preprocess, quality_checks, to_model_coords

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="PyCanc")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

engine = PyCancEngine(weights.DEFAULT_DIR)
pool = ThreadPoolExecutor(max_workers=1)
cases: dict[str, dict] = {}
download_state = {"state": "idle", "message": ""}


def _new_case(ct, name: str) -> dict:
    prep = preprocess(ct)
    cid = uuid.uuid4().hex[:10]
    case = {
        "id": cid,
        "name": name,
        "created": time.time(),
        "meta": ct.meta,
        "spacing": list(ct.spacing),
        "shape": list(ct.hu.shape),
        "valid": list(prep.valid),
        "warnings": quality_checks(ct),
        "prep": prep,
        "job": {"state": "idle"},
    }
    # map phantom nodule into model space for ground-truth display
    nod = (ct.meta or {}).get("nodule")
    if nod:
        nod["model_zyx"] = to_model_coords(ct, *nod["center_mm"])
    cases[cid] = case
    return case


def _public(case: dict) -> dict:
    return {k: v for k, v in case.items() if k not in ("prep",)} | {
        "job": {k: v for k, v in case["job"].items() if k != "result_raw"}
    }


# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return engine.info | {"download": download_state}


class PhantomParams(BaseModel):
    seed: int = 7
    nodule: bool = True
    nodule_mm: float = 14
    nodule_type: str = "solid"
    spiculated: bool = True
    nodule_side: str = "right"
    nodule_level: float = 0.35


@app.post("/api/cases/phantom")
def create_phantom(p: PhantomParams):
    ct = make_phantom(**p.model_dump())
    label = f"Phantom #{p.seed} · " + (f"{p.nodule_mm:g} mm {p.nodule_type} nodule" if p.nodule else "no nodule")
    return _public(_new_case(ct, label))


@app.post("/api/cases/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        ct = load_any(data, file.filename or "")
    except Exception as e:  # surface parser errors to the UI
        raise HTTPException(400, f"Could not read {file.filename}: {e}")
    return _public(_new_case(ct, file.filename or "upload"))


@app.get("/api/cases")
def list_cases():
    return [_public(c) for c in sorted(cases.values(), key=lambda c: -c["created"])]


def _case(cid: str) -> dict:
    if cid not in cases:
        raise HTTPException(404, "case not found")
    return cases[cid]


@app.get("/api/cases/{cid}")
def get_case(cid: str):
    return _public(_case(cid))


@app.get("/api/cases/{cid}/volume/{kind}")
def get_volume(cid: str, kind: str):
    """kind: lung (model input), wide (HU -1000..1500), mask (lung segmentation)."""
    prep = _case(cid)["prep"]
    arr = {"lung": prep.display, "wide": prep.wide, "mask": prep.lung_mask}.get(kind)
    if arr is None:
        raise HTTPException(404, "unknown volume kind")
    return Response(arr.tobytes(), media_type="application/octet-stream", headers={"X-Shape": "200,256,256"})


class PredictParams(BaseModel):
    n_models: int | None = None
    tta: int = 1


@app.post("/api/cases/{cid}/predict")
def predict(cid: str, p: PredictParams):
    case = _case(cid)
    if case["job"].get("state") in ("queued", "running"):
        return case["job"]
    job = {"state": "queued", "progress": {}, "started": time.time()}
    case["job"] = job

    def run():
        job["state"] = "running"
        try:
            res = engine.predict(case["prep"], p.n_models, p.tta, progress=lambda d: job.__setitem__("progress", d))
            attn = res.pop("attention")
            res["attention_b64"] = base64.b64encode(np.clip(attn * 255, 0, 255).astype(np.uint8).tobytes()).decode()
            res["attention_shape"] = list(attn.shape)
            res["seconds"] = round(time.time() - job["started"], 1)
            job["result"] = res
            job["state"] = "done"
        except Exception as e:
            job["state"] = "error"
            job["error"] = str(e)

    pool.submit(run)
    return job


@app.get("/api/cases/{cid}/job")
def get_job(cid: str):
    return _case(cid)["job"]


class ClinicalParams(BaseModel):
    patient: dict
    image_risk: list[float] | None = None


@app.post("/api/clinical")
def clinical(p: ClinicalParams):
    """PLCOm2012 6-year risk, plus the fused image + clinical risk when a fusion model has been fitted."""
    try:
        res = plcom2012(Patient(**p.patient))
    except TypeError as e:
        raise HTTPException(400, f"bad patient fields: {e}")
    res["fused"] = None
    if engine.fusion is not None and p.image_risk and res["risk"] is not None:
        res["fused"] = engine.fusion(p.image_risk, res["risk"])
    res["fusion_available"] = engine.fusion is not None
    return res


@app.post("/api/weights/download")
def download_weights():
    if download_state["state"] == "running":
        return download_state

    def run():
        download_state.update(state="running", message="Downloading ~700 MB from GitHub releases…")
        try:
            weights.download_checkpoints(engine.checkpoint_dir, progress=lambda m: download_state.update(message=m))
            engine.reload()
            download_state.update(state="done", message=f"Loaded {len(engine.models)} official models")
        except Exception as e:
            download_state.update(state="error", message=str(e))

    threading.Thread(target=run, daemon=True).start()
    return download_state


def main():
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()

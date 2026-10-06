#!/usr/bin/env python3
"""
SKINSCOPE - photo-based acne and skin analysis with accounts (runs on your computer).

Run:
    python skinscope_app.py
It opens http://127.0.0.1:8765 in your browser. Ctrl+C in the terminal stops it.

Forgot your password?   python skinscope_app.py --reset-password you@example.com

Needs (inside your virtual environment):
    pip install "opencv-python<5" numpy

Privacy: accounts, photos and results stay on THIS computer. Photos are analysed in
memory and never saved; only numbers and lesion positions are stored, in
~/skinscope_data/skinscope_app.db. Passwords are stored as salted hashes.

Honesty: lesion detection is rule-based computer vision (not a trained neural network)
and is not clinically validated. Photos cannot measure vitamin or mineral levels: the
nutrition section is a watch-list built from your diet answers. The breakout outlook is
an estimate. None of this is medical advice.
"""

import argparse
import base64
import csv
import getpass
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
import webbrowser
from datetime import date, datetime, timedelta
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit('Missing libraries. Run:  pip install "opencv-python<5" numpy')

if not hasattr(cv2, "CascadeClassifier"):
    sys.exit(
        "Your OpenCV is too new (no face detector).\n"
        'Fix:  pip uninstall -y opencv-python && pip install "opencv-python<5"'
    )

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ALGO_VERSION = 3
DATA_DIR = Path(os.environ.get("SKINSCOPE_DIR", str(Path.home() / "skinscope_data")))
DB_PATH = DATA_DIR / "skinscope_app.db"
DB = None
DB_LOCK = threading.Lock()
SESSION_DAYS = 30
PBKDF2_ITERS = 240_000
PW_MIN = 8
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

MIN_BRIGHT, MAX_BRIGHT = 30, 85
MAX_IMBALANCE = 8
MIN_SHARP = 12
MIN_FACE_RATIO = 0.18

ZONES = {
    "forehead": (0.28, 0.72, 0.04, 0.20),
    "left_cheek": (0.10, 0.34, 0.48, 0.72),
    "right_cheek": (0.66, 0.90, 0.48, 0.72),
    "nose": (0.42, 0.58, 0.42, 0.66),
    "chin": (0.34, 0.66, 0.84, 0.97),
}
ZONE_LIST = list(ZONES)
LABELS = {"forehead": "Forehead", "left_cheek": "L cheek", "right_cheek": "R cheek",
          "nose": "Nose", "chin": "Chin"}
FIELDS = ("bright", "red", "rel", "shine", "tex", "spots", "pustules", "marks")
FACE_KEYS = ["brightness", "imbalance", "sharpness", "face_ratio", "red_avg", "spots_total"]
VIEWS = ("front", "right", "left")

SENS_TABLE = {0: (2.8, 5.0, 1.25), 1: (2.2, 3.8, 1.0), 2: (1.8, 3.0, 0.8)}

# zone risk engine (personal baseline)
BASE_MIN = 5
LEAD = 12 * 3600
HORIZON = 7 * 86400
W_RED, W_SPOTS, W_SHINE = 0.6, 0.25, 0.15
FLOORS = {"rel": 0.6, "spots": 1.0, "shine": 1.5}
LOW_MAX, WATCH_MAX = 0.8, 1.6
THRESHOLDS = (0.8, 1.2, 1.6, 2.0)

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if CASCADE.empty():
    sys.exit("Could not load OpenCV's face detector file. Reinstall opencv-python<5.")

# Optional trained acne detector (ONNX, YOLOv8/YOLO11 style). Put acne_model.onnx (and optionally
# acne_model.json with {"imgsz": 800, "names": ["acne"]}) in the data folder. If it is missing or
# fails to load, the rule-based detector is used automatically.
MODEL_PATH = DATA_DIR / "acne_model.onnx"
MODEL_META = DATA_DIR / "acne_model.json"
DET_MODE = "auto"  # auto | rules | model
MODEL_CONF = {0: 0.45, 1: 0.30, 2: 0.20}  # confidence threshold by sensitivity
_MODEL = {"tried": False, "obj": None}

DEFAULT_PROFILE = {"diet": "omni", "pregnancy": "unknown", "self_skin_type": "", "goal": "",
                   "avoid": "", "sens": 1, "last_q": {}, "grocery": {}}


class Throttled(ValueError):
    pass


class AuthError(Exception):
    pass


# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------
def init_db():
    global DB
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    DB.row_factory = sqlite3.Row
    DB.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
            salt BLOB NOT NULL, pw_hash BLOB NOT NULL, created REAL NOT NULL,
            profile TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, ts REAL NOT NULL,
            ok INTEGER NOT NULL, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS diary (
            user_id INTEGER NOT NULL, day TEXT NOT NULL, data TEXT NOT NULL,
            PRIMARY KEY (user_id, day));
        CREATE TABLE IF NOT EXISTS breakouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, ts REAL NOT NULL,
            zone TEXT NOT NULL, severity INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, ts REAL NOT NULL,
            detected INTEGER NOT NULL, false_pos INTEGER NOT NULL, missed INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, factor TEXT NOT NULL,
            start_ts REAL NOT NULL, days INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'active',
            stopped_ts REAL, adherence TEXT NOT NULL DEFAULT '{}');
        """
    )
    DB.commit()


def clean(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------
# Accounts and sessions
# ----------------------------------------------------------------------------
FAILS = {}


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)


def _norm_email(email):
    return (email or "").strip().lower()


def _check_rate(key):
    rec = FAILS.get(key)
    if rec and rec[0] >= 5 and time.time() - rec[1] < 600:
        raise Throttled("Too many attempts. Please wait a few minutes and try again.")


def _note_fail(key):
    rec = FAILS.get(key)
    if not rec or time.time() - rec[1] >= 600:
        FAILS[key] = [1, time.time()]
    else:
        rec[0] += 1


def create_user(name, email, password):
    name = (name or "").strip()[:60]
    email = _norm_email(email)
    if not name:
        raise ValueError("Please enter your name.")
    if not EMAIL_RE.match(email) or len(email) > 120:
        raise ValueError("Please enter a valid email address.")
    if len(password or "") < PW_MIN:
        raise ValueError(f"Password must be at least {PW_MIN} characters.")
    salt = os.urandom(16)
    pw = _hash(password, salt)
    try:
        with DB_LOCK:
            cur = DB.execute(
                "INSERT INTO users (email, name, salt, pw_hash, created, profile) VALUES (?, ?, ?, ?, ?, ?)",
                (email, name, salt, pw, time.time(), json.dumps(DEFAULT_PROFILE)))
            DB.commit()
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("An account with that email already exists. Try signing in.")


def authenticate(email, password):
    email = _norm_email(email)
    _check_rate(email)
    with DB_LOCK:
        row = DB.execute("SELECT id, salt, pw_hash FROM users WHERE email = ?", (email,)).fetchone()
    if row:
        ok = hmac.compare_digest(_hash(password or "", bytes(row["salt"])), bytes(row["pw_hash"]))
    else:
        _hash(password or "", b"\x00" * 16)
        ok = False
    if not ok:
        _note_fail(email)
        raise ValueError("Incorrect email or password.")
    FAILS.pop(email, None)
    return row["id"]


def start_session(uid):
    token = secrets.token_urlsafe(32)
    th = hashlib.sha256(token.encode()).hexdigest()
    with DB_LOCK:
        DB.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        DB.execute("INSERT INTO sessions (token_hash, user_id, expires) VALUES (?, ?, ?)",
                   (th, uid, time.time() + SESSION_DAYS * 86400))
        DB.commit()
    return token


def end_session(token):
    if token:
        with DB_LOCK:
            DB.execute("DELETE FROM sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),))
            DB.commit()


def user_for_token(token):
    if not token:
        return None
    th = hashlib.sha256(token.encode()).hexdigest()
    with DB_LOCK:
        row = DB.execute(
            "SELECT u.id, u.email, u.name, u.profile, s.expires FROM sessions s "
            "JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?", (th,)).fetchone()
    if not row or row["expires"] < time.time():
        return None
    return _user_dict(row)


def _user_dict(row):
    try:
        prof = json.loads(row["profile"] or "{}")
    except ValueError:
        prof = {}
    merged = {**DEFAULT_PROFILE, **prof}
    return {"id": row["id"], "email": row["email"], "name": row["name"], "profile": merged}


def get_user(uid):
    with DB_LOCK:
        row = DB.execute("SELECT id, email, name, profile FROM users WHERE id = ?", (uid,)).fetchone()
    return _user_dict(row) if row else None


def public_user(u):
    return {"id": u["id"], "email": u["email"], "name": u["name"], "profile": u["profile"]}


def reset_password_cli(email):
    email = _norm_email(email)
    with DB_LOCK:
        row = DB.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        sys.exit(f"No account found for {email}")
    pw = getpass.getpass("New password (min 8 characters): ")
    if len(pw) < PW_MIN or pw != getpass.getpass("Repeat it: "):
        sys.exit("Passwords did not match or were too short. Nothing changed.")
    salt = os.urandom(16)
    with DB_LOCK:
        DB.execute("UPDATE users SET salt = ?, pw_hash = ? WHERE id = ?", (salt, _hash(pw, salt), row["id"]))
        DB.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
        DB.commit()
    print("Password updated. You can sign in now.")


# ----------------------------------------------------------------------------
# Photo analysis
# ----------------------------------------------------------------------------
def detect_face(gray):
    height, width = gray.shape
    scale = 640.0 / width if width > 640 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1.0 else gray
    small = cv2.equalizeHist(small)
    faces = CASCADE.detectMultiScale(small, scaleFactor=1.15, minNeighbors=6, minSize=(80, 80))
    if len(faces) == 0:
        return None, 0
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x / scale), int(y / scale), int(w / scale), int(h / scale)), len(faces)


def zone_at(nx, ny):
    """Which zone (if any) contains a point given in face-relative 0-1 coordinates."""
    for name, (zx0, zx1, zy0, zy1) in ZONES.items():
        if zx0 <= nx <= zx1 and zy0 <= ny <= zy1:
            return name
    return None


class AcneModel:
    """Runs a YOLO-style ONNX acne detector through OpenCV's DNN module."""

    def __init__(self, path, meta=None):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        meta = meta or {}
        self.imgsz = int(meta.get("imgsz", 800))
        self.names = [str(n).lower() for n in meta.get("names", ["acne"])]

    def _forward(self, blob):
        self.net.setInput(blob)
        return self.net.forward()

    def detect(self, face_bgr, conf_thr=0.3, iou_thr=0.45):
        """Returns detections in face-crop pixel coordinates: dicts with cx, cy, r, conf, t."""
        h, w = face_bgr.shape[:2]
        s = self.imgsz
        scale = min(s / float(w), s / float(h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        canvas = np.full((s, s, 3), 114, np.uint8)
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(face_bgr, (nw, nh), interpolation=interp)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (s, s), swapRB=True)
        out = np.asarray(self._forward(blob))
        if out.ndim == 3:
            out = out[0]
        if out.ndim != 2:
            raise ValueError("Unexpected model output shape %s" % (out.shape,))
        ch = 4 + len(self.names)          # channels per candidate: box (4) + one score per class
        if out.shape[0] == ch and out.shape[1] != ch:
            out = out.T                    # (4+nc, N) -> (N, 4+nc)
        elif out.shape[1] != ch and out.shape[0] < out.shape[1]:
            out = out.T                    # unknown class count: the short axis holds the channels
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= conf_thr
        out, cls, conf = out[keep], cls[keep], conf[keep]
        if len(out) == 0:
            return []
        boxes = []
        for cx, cy, bw, bh in out[:, :4]:
            boxes.append([float(cx - bw / 2.0), float(cy - bh / 2.0), float(bw), float(bh)])
        idx = cv2.dnn.NMSBoxes(boxes, [float(c) for c in conf], float(conf_thr), float(iou_thr))
        idx = np.array(idx).flatten() if len(idx) else []
        dets = []
        for i in idx:
            x, y, bw, bh = boxes[int(i)]
            cx = (x + bw / 2.0 - px) / scale
            cy = (y + bh / 2.0 - py) / scale
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            name = self.names[int(cls[i])] if int(cls[i]) < len(self.names) else "acne"
            dets.append({"cx": cx, "cy": cy, "r": max(bw, bh) / (2.0 * scale), "conf": float(conf[i]),
                         "t": "p" if "pust" in name else "i"})
        return dets


def get_model():
    if DET_MODE == "rules":
        return None
    if not _MODEL["tried"]:
        _MODEL["tried"] = True
        if MODEL_PATH.exists():
            try:
                meta = json.loads(MODEL_META.read_text()) if MODEL_META.exists() else {}
                _MODEL["obj"] = AcneModel(MODEL_PATH, meta)
            except Exception as e:
                print("Could not load the acne model, using rule-based detection instead:", e)
    if DET_MODE == "model" and _MODEL["obj"] is None:
        raise RuntimeError("Detector 'model' requested but no working model found at %s" % MODEL_PATH)
    return _MODEL["obj"]


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def find_lesions(p_lab, fw, sens):
    """Acne-like lesions in one zone patch. Types: i inflamed, p pustule-like, m dark mark."""
    h, w = p_lab.shape[:2]
    if h < 12 or w < 12:
        return []
    mult, floor, lscale = SENS_TABLE.get(sens, SENS_TABLE[1])
    L = p_lab[:, :, 0]
    A = p_lab[:, :, 1]
    k = max(3, _odd(min(max(15, int(fw * 0.12)), min(h, w))))
    a_bg = cv2.medianBlur(A, k).astype(np.float32)
    l_bg = cv2.medianBlur(L, k).astype(np.float32)
    sc = max(1.0, fw / 400.0)
    a_s = cv2.GaussianBlur(A.astype(np.float32), (0, 0), 1.2 * sc)
    l_s = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 1.2 * sc)
    da = a_s - a_bg
    dl = l_bg - l_s

    sig_a = 1.4826 * float(np.median(np.abs(da - np.median(da))))
    sig_l = 1.4826 * float(np.median(np.abs(dl - np.median(dl))))
    thr_a = max(floor, mult * sig_a)
    thr_l = max(14.0 * lscale, 2.6 * sig_l * lscale)

    r_min, r_max = fw * 0.008, fw * 0.045
    a_min, a_max = math.pi * r_min ** 2, math.pi * r_max ** 2
    ks = _odd(round(3 * sc))
    k3 = np.ones((ks, ks), np.uint8)
    kd = _odd(round(9 * sc))
    out = []

    red = (da > thr_a).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k3)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k3)
    red_any = np.zeros((h, w), np.uint8)
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < a_min or area > a_max:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (per * per) if per > 0 else 0.0
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / float(max(1, min(bw, bh)))
        if circ < 0.4 or aspect > 2.5:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [c], -1, 1, -1)
        mean_da = float(da[m > 0].mean())
        if mean_da < thr_a * 1.15:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        kind = "i"
        sub_m = m[by:by + bh, bx:bx + bw]
        sub_l = l_s[by:by + bh, bx:bx + bw]
        yy, xx = np.mgrid[0:bh, 0:bw]
        dist = np.hypot(xx + bx - cx, yy + by - cy)
        centre = (dist < 0.45 * r) & (sub_m > 0)
        rim = (dist >= 0.6 * r) & (sub_m > 0)
        if centre.sum() >= 3 and rim.sum() >= 3:
            if float(sub_l[centre].mean()) - float(sub_l[rim].mean()) > 10.0 * lscale:
                kind = "p"
        out.append({"cx": cx, "cy": cy, "r": r, "t": kind, "s": mean_da / thr_a})
        red_any = np.maximum(red_any, cv2.dilate(m, np.ones((kd, kd), np.uint8)))

    dark = ((dl > thr_l) & (da < thr_a * 0.8)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k3)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k3)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < a_min * 1.5 or area > a_max * 2.2:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (per * per) if per > 0 else 0.0
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / float(max(1, min(bw, bh)))
        if circ < 0.35 or aspect > 2.2:
            continue
        if bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if red_any[min(int(cy), h - 1), min(int(cx), w - 1)]:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [c], -1, 1, -1)
        mean_dl = float(dl[m > 0].mean())
        if mean_dl < thr_l * 1.1:
            continue
        out.append({"cx": cx, "cy": cy, "r": r, "t": "m", "s": mean_dl / thr_l})

    out.sort(key=lambda d: -d["s"])
    return out[:25]


def analyze_face(img, box, sens=1, model=None):
    height, width = img.shape[:2]
    x, y, w, h = box
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
    if x1 - x0 < 60 or y1 - y0 < 60:
        return None
    face = img[y0:y1, x0:x1]
    fh, fw = face.shape[:2]
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    light_all = lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
    a_all = lab[:, :, 1].astype(np.float32) - 128.0
    face_l = float(light_all.mean())
    core = a_all[int(fh * 0.15): int(fh * 0.85), int(fw * 0.15): int(fw * 0.85)]
    face_a = float(core.mean())
    sharp = float(cv2.Laplacian(cv2.resize(gray, (256, 256)), cv2.CV_64F).var())

    model_dets = []
    if model is not None:
        try:
            for d in model.detect(face, MODEL_CONF.get(sens, 0.30)):
                z = zone_at(d["cx"] / fw, d["cy"] / fh)
                if z:
                    model_dets.append((z, d))
        except Exception as e:
            print("Acne model failed on a photo, using rule-based detection:", e)
            model, model_dets = None, []
    zones, lesions = {}, []
    brights, reds = [], []
    spots_total = 0
    for name, (zx0, zx1, zy0, zy1) in ZONES.items():
        sx0, sx1, sy0, sy1 = int(fw * zx0), int(fw * zx1), int(fh * zy0), int(fh * zy1)
        p_lab = lab[sy0:sy1, sx0:sx1]
        p_gray = gray[sy0:sy1, sx0:sx1]
        if p_lab.size == 0 or min(p_lab.shape[:2]) < 6:
            return None
        light = p_lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
        a_ch = p_lab[:, :, 1].astype(np.float32) - 128.0
        bright = float(light.mean())
        red = float(a_ch.mean())
        shine = float((light > face_l + 18).mean() * 100.0)
        tex = float(cv2.Laplacian(cv2.resize(p_gray, (64, 64)), cv2.CV_64F).var())

        n_spots = n_pust = n_marks = 0
        zl = []
        for d in find_lesions(p_lab, fw, sens):
            if model is not None and d["t"] != "m":
                continue   # the model handles active spots; rules still find dark marks
            zl.append((x0 + sx0 + d["cx"], y0 + sy0 + d["cy"], d["r"], d["t"], d["s"]))
        for z2, d in sorted(model_dets, key=lambda zd: -zd[1]["conf"]):
            if z2 == name and sum(1 for l in zl if l[3] != "m") < 40:
                zl.append((x0 + d["cx"], y0 + d["cy"], d["r"], d["t"], d["conf"]))
        for gx, gy, rr, t, sco in zl:
            lesions.append({
                "x": round(gx, 1), "y": round(gy, 1), "r": round(rr, 1), "t": t, "z": name,
                "nx": round((gx - x0) / fw, 4), "ny": round((gy - y0) / fh, 4),
                "nr": round(rr / fw, 4), "s": round(sco, 2)})
            if t == "m":
                n_marks += 1
            else:
                n_spots += 1
                if t == "p":
                    n_pust += 1
        zones[name] = {"box": [x0 + sx0, y0 + sy0, x0 + sx1, y0 + sy1],
                       "bright": round(bright, 2), "red": round(red, 2), "rel": round(red - face_a, 2),
                       "shine": round(shine, 2), "tex": round(tex, 1),
                       "spots": n_spots, "pustules": n_pust, "marks": n_marks}
        brights.append(bright)
        reds.append(red)
        spots_total += n_spots

    face_info = {
        "brightness": round(float(np.mean(brights)), 2),
        "imbalance": round(abs(zones["left_cheek"]["bright"] - zones["right_cheek"]["bright"]), 2),
        "sharpness": round(sharp, 1), "face_ratio": round((x1 - x0) / float(width), 3),
        "red_avg": round(float(np.mean(reds)), 2), "spots_total": spots_total}
    return {"box": [x0, y0, x1 - x0, y1 - y0], "zones": zones, "lesions": lesions, "face": face_info,
            "detector": "model" if model is not None else "rules"}


def quality_notes(m):
    notes = []
    if m["brightness"] < MIN_BRIGHT:
        notes.append("Too dark - use more light")
    elif m["brightness"] > MAX_BRIGHT:
        notes.append("Too bright - reduce glare or flash")
    if m["imbalance"] > MAX_IMBALANCE:
        notes.append("Uneven light - face a window so both cheeks are lit equally")
    if m["sharpness"] < MIN_SHARP:
        notes.append("Blurry - hold the phone steady or clean the lens")
    if m["face_ratio"] < MIN_FACE_RATIO:
        notes.append("Face too small - move closer")
    return notes


def decode_image(data_url):
    if not isinstance(data_url, str) or len(data_url) > 9_000_000:
        raise ValueError("Could not read that photo.")
    raw = data_url.split(",", 1)[1] if data_url.startswith("data:") and "," in data_url else data_url
    try:
        buf = np.frombuffer(base64.b64decode(raw, validate=False), np.uint8)
    except Exception:
        raise ValueError("Could not read that photo.")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read that photo.")
    return img


def analyze_photo(img, sens=1):
    height, width = img.shape[:2]
    base = {"w": width, "h": height}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box, nfaces = detect_face(gray)
    if box is None:
        return {"found": False, "notes": [], **base}
    res = analyze_face(img, box, sens, get_model())
    if res is None:
        return {"found": False, "notes": [], **base}
    res.update({"found": True, "faces": nfaces, **base})
    res["notes"] = quality_notes(res["face"])
    return res


def flatten_view(res):
    m = dict(res["face"])
    for z, q in res["zones"].items():
        for f in FIELDS:
            m[f"{z}_{f}"] = q[f]
    return m


def combine_views(results):
    """Front photo gives the measurements. For each cheek, use whichever photo found more lesions."""
    front = results["front"]
    m = flatten_view(front)
    lesions = []
    for z in ZONES:
        cands = []
        for view, res in results.items():
            if not res or not res.get("found"):
                continue
            if view == "front" or (view == "left" and z == "left_cheek") or (view == "right" and z == "right_cheek"):
                cands.append([l for l in res["lesions"] if l["z"] == z])
        best = max(cands, key=len) if cands else []
        lesions.extend(best)
    counts = {z: {"spots": 0, "pustules": 0, "marks": 0} for z in ZONES}
    packed = []
    for l in lesions:
        c = counts[l["z"]]
        if l["t"] == "m":
            c["marks"] += 1
        else:
            c["spots"] += 1
            if l["t"] == "p":
                c["pustules"] += 1
        packed.append([round(l["nx"], 3), round(l["ny"], 3), round(l["nr"], 3), l["t"], ZONE_LIST.index(l["z"])])
    for z in ZONES:
        m[f"{z}_spots"], m[f"{z}_pustules"], m[f"{z}_marks"] = (
            counts[z]["spots"], counts[z]["pustules"], counts[z]["marks"])
    m["spots_total"] = sum(counts[z]["spots"] for z in ZONES)
    m["pustules_total"] = sum(counts[z]["pustules"] for z in ZONES)
    m["marks_total"] = sum(counts[z]["marks"] for z in ZONES)
    return m, packed


def grade_for(spots):
    if spots <= 2:
        return "Clear / almost clear"
    if spots <= 9:
        return "Mild"
    if spots <= 24:
        return "Moderate"
    return "Severe"


def skin_score(m):
    zs = list(ZONES)
    spots = sum(m[f"{z}_spots"] for z in zs)
    pust = sum(m[f"{z}_pustules"] for z in zs)
    marks = sum(m[f"{z}_marks"] for z in zs)
    shine = float(np.mean([m[f"{z}_shine"] for z in zs]))
    rel_std = float(np.std([m[f"{z}_rel"] for z in zs]))
    parts = {"acne": max(0.0, 100 - 6 * (spots + 0.5 * pust)), "marks": max(0.0, 100 - 8 * marks),
             "oil": max(0.0, 100 - 5 * shine), "even": max(0.0, 100 - 30 * rel_std)}
    overall = 0.5 * parts["acne"] + 0.15 * parts["marks"] + 0.15 * parts["oil"] + 0.2 * parts["even"]
    return {"score": int(round(overall)), "s_acne": int(round(parts["acne"])),
            "s_marks": int(round(parts["marks"])), "s_oil": int(round(parts["oil"])),
            "s_even": int(round(parts["even"])), "grade": grade_for(spots)}


def estimate_skin_type(m):
    t = (m["forehead_shine"] + m["nose_shine"]) / 2.0
    c = (m["left_cheek_shine"] + m["right_cheek_shine"]) / 2.0
    if t >= 6 and c >= 4:
        return "Oily"
    if t >= 6:
        return "Combination"
    if t < 2.5 and c < 2.5:
        return "Normal to dry"
    return "Normal"


# ----------------------------------------------------------------------------
# Data access helpers
# ----------------------------------------------------------------------------
def user_scans(uid):
    with DB_LOCK:
        rows = DB.execute("SELECT id, ts, ok, data FROM scans WHERE user_id = ? ORDER BY ts, id", (uid,)).fetchall()
    out = []
    for r in rows:
        d = json.loads(r["data"])
        d.update({"id": r["id"], "ts": r["ts"], "ok": bool(r["ok"])})
        out.append(d)
    return out


def user_breakouts(uid):
    with DB_LOCK:
        rows = DB.execute("SELECT id, ts, zone, severity FROM breakouts WHERE user_id = ? ORDER BY ts", (uid,)).fetchall()
    return [dict(r) for r in rows]


def user_diary(uid):
    with DB_LOCK:
        rows = DB.execute("SELECT day, data FROM diary WHERE user_id = ? ORDER BY day", (uid,)).fetchall()
    out = []
    for r in rows:
        d = json.loads(r["data"])
        d["day"] = r["day"]
        out.append(d)
    return out


def user_reviews(uid):
    with DB_LOCK:
        rows = DB.execute("SELECT detected, false_pos, missed FROM reviews WHERE user_id = ?", (uid,)).fetchall()
    return [dict(r) for r in rows]


def current_view(scans):
    """Scans comparable with the newest one (same detector version and sensitivity)."""
    cur = [r for r in scans if r.get("v") == ALGO_VERSION]
    if not cur:
        return []
    newest = max(cur, key=lambda r: r["id"])
    return [r for r in cur if r.get("sens", 1) == newest.get("sens", 1)
            and r.get("det", "rules") == newest.get("det", "rules")]


# ----------------------------------------------------------------------------
# Personal-baseline zone risk + self-validation
# ----------------------------------------------------------------------------
def _stats(vals, floor):
    v = np.asarray(vals, dtype=float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) * 1.4826)
    return med, max(mad, floor)


def _clip(v):
    return min(max(v, 0.0), 6.0)


def zone_score(zone, past, recent):
    comps = {}
    for key in ("rel", "spots", "shine"):
        base = [r[f"{zone}_{key}"] for r in past]
        med, spread = _stats(base, FLOORS[key])
        cur = float(np.mean([r[f"{zone}_{key}"] for r in recent]))
        comps[key] = {"cur": cur, "med": med, "z": (cur - med) / spread}
    score = (W_RED * _clip(comps["rel"]["z"]) + W_SPOTS * _clip(comps["spots"]["z"])
             + W_SHINE * _clip(comps["shine"]["z"]))
    return score, comps


def level_for(score):
    return "low" if score < LOW_MAX else "watch" if score < WATCH_MAX else "high"


def backtest(ok_r, breakouts):
    n = len(ok_r)
    scored = {z: [] for z in ZONES}
    for i in range(BASE_MIN + 1, n):
        past, recent = ok_r[: i - 1], ok_r[i - 1: i + 1]
        for z in ZONES:
            sc, _ = zone_score(z, past, recent)
            scored[z].append((ok_r[i]["ts"], sc))
    bo = {z: sorted(b["ts"] for b in breakouts if b["zone"] == z) for z in ZONES}

    def positive(z, t):
        return any(t + LEAD <= b <= t + HORIZON for b in bo[z])

    def recent_breakout(z, t):
        return any(t - 14 * 86400 < b <= t for b in bo[z])

    events = [(b["zone"], b["ts"]) for b in breakouts]
    testable = [(z, b) for z, b in events if any(b - HORIZON <= t <= b - LEAD for t, _ in scored[z])]
    pairs = sum(len(v) for v in scored.values())
    pos_pairs = sum(1 for z in ZONES for t, _ in scored[z] if positive(z, t))
    base_rate = pos_pairs / pairs if pairs else None

    def evaluate(flag):
        flagged = tp = 0
        for z in ZONES:
            for t, sc in scored[z]:
                if flag(z, t, sc):
                    flagged += 1
                    if positive(z, t):
                        tp += 1
        caught = 0
        for z, b in testable:
            if any(flag(z, t, sc) for t, sc in scored[z] if b - HORIZON <= t <= b - LEAD):
                caught += 1
        return {"flagged": flagged, "tp": tp, "precision": (tp / flagged) if flagged else None,
                "caught": caught, "catch_rate": (caught / len(testable)) if testable else None}

    rows = []
    for thr in THRESHOLDS:
        row = evaluate(lambda z, t, sc, thr=thr: sc >= thr)
        row["threshold"] = thr
        rows.append(row)
    recurrence = evaluate(lambda z, t, sc: recent_breakout(z, t))
    enough = len(testable) >= 5
    d = rows[0]
    lead_h, hor_d = LEAD // 3600, HORIZON // 86400
    if not enough:
        verdict = (f"Too early to judge. {len(events)} breakout(s) logged, {len(testable)} testable "
                   f"(a good scan is needed {lead_h} hours to {hor_d} days before each one). "
                   f"Aim for at least 5, ideally 15 or more. Scanning every 2 to 3 days speeds this up.")
    elif d["precision"] is None:
        verdict = "The score never flagged a zone, so there is nothing to compare yet."
    else:
        lift = (d["precision"] / base_rate) if base_rate else None
        verdict = (f"At the 'watch' level the score caught {d['caught']} of {len(testable)} testable breakouts. "
                   f"When it flagged a zone, a breakout followed {d['precision']:.0%} of the time")
        verdict += (f" (a random zone-week: {base_rate:.0%}, so about x{lift:.1f})." if base_rate and lift else ".")
        if recurrence["precision"] is not None:
            verdict += (f" The simple rule 'it broke out here in the last 2 weeks' was right "
                        f"{recurrence['precision']:.0%} of the time. The photos only add value if the score beats that.")
        if len(testable) < 15:
            verdict += " Few events so far: treat this as a hint, not proof."
    return {"n_scans": n, "pairs": pairs, "events": len(events), "testable": len(testable), "enough": enough,
            "base_rate": base_rate, "rows": rows, "recurrence": recurrence, "verdict": verdict,
            "lead_hours": lead_h, "horizon_days": hor_d}


FACTORS = [
    ("sleep_low", "Slept under 6 hours", lambda c: c.get("sleep") is not None and c["sleep"] < 6),
    ("stress_high", "High stress (4-5)", lambda c: (c.get("stress") or 0) >= 4),
    ("sugar", "Lots of sugar / sweets", lambda c: bool(c.get("sugar"))),
    ("dairy", "Dairy", lambda c: bool(c.get("dairy"))),
    ("new_product", "Tried a new product", lambda c: bool(c.get("new_product"))),
    ("sweat", "Heavy sweat / exercise", lambda c: bool(c.get("sweat"))),
]


def lifestyle_insights(diary, breakouts):
    bdays = {datetime.fromtimestamp(b["ts"]).date() for b in breakouts}
    today = date.today()
    rows = []
    for key, label, fn in FACTORS:
        w, wo = [0, 0], [0, 0]
        for c in diary:
            try:
                d = datetime.strptime(c["day"], "%Y-%m-%d").date()
            except ValueError:
                continue
            if d + timedelta(days=2) > today:
                continue
            hit = (d + timedelta(days=1) in bdays) or (d + timedelta(days=2) in bdays)
            tgt = w if fn(c) else wo
            tgt[0] += 1
            tgt[1] += int(hit)
        rows.append({"key": key, "label": label, "n_with": w[0], "rate_with": (w[1] / w[0]) if w[0] else None,
                     "n_without": wo[0], "rate_without": (wo[1] / wo[0]) if wo[0] else None,
                     "enough": w[0] >= 5 and wo[0] >= 5})
    return rows


def review_summary(rows):
    n = len(rows)
    det = sum(r["detected"] for r in rows)
    fp = sum(r["false_pos"] for r in rows)
    miss = sum(r["missed"] for r in rows)
    tp = det - fp
    return {"n": n, "detected": det, "false_pos": fp, "missed": miss,
            "precision": (tp / det) if det else None, "recall": (tp / (tp + miss)) if (tp + miss) else None}


# ----------------------------------------------------------------------------
# Questionnaire, nutrition watch, plans
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Trigger Lab: personal experiments. Change ONE thing for weeks, keep scanning, and compare with your own
# baseline, with honest uncertainty (bootstrap range), a lead-in period and an adherence check.
# ----------------------------------------------------------------------------
EXPERIMENTS = {
    "dairy": {"label": "Cut dairy", "evidence": "Limited to moderate",
              "detail": "Avoid milk, yoghurt, cheese, paneer and whey protein. Keep calcium and B12 from other foods.",
              "ask": "Did you avoid dairy today?"},
    "sugar": {"label": "Cut sweets and sugary drinks", "evidence": "Moderate",
              "detail": "Keep sweets, desserts and sugary drinks to a minimum. Fruit is fine.",
              "ask": "Did you avoid sweets and sugary drinks today?"},
    "fried": {"label": "Skip fried and fast food", "evidence": "Weak",
              "detail": "Avoid deep-fried and fast food. Cooking the same food another way is fine.",
              "ask": "Did you avoid fried and fast food today?"},
    "sleep": {"label": "Sleep 7+ hours", "evidence": "Limited (observational)",
              "detail": "Aim to be asleep early enough for 7 to 8 hours every night.",
              "ask": "Did you sleep 7 hours or more last night?"},
}
LEAD_IN_MAX = 14          # days at the start ignored, because skin takes weeks to respond
BASELINE_DAYS = 42        # how far back "before" scans are taken from
MIN_SCANS_EACH = 3
EXP_CAVEATS = [
    "This is a test on one person with no control group. Weather, stress, periods, new products or a change in "
    "photo lighting can look like an effect.",
    "People often start when their skin is at its worst, so some improvement happens on its own "
    "(regression to the mean).",
    "Change only one thing at a time and keep everything else the same.",
]


def _day(ts):
    return datetime.fromtimestamp(ts).date()


def change_note(prev, cur):
    """Is a change in spot count bigger than ordinary scan-to-scan wobble? Poisson-style rule of thumb."""
    noise = max(2.0, math.sqrt(2.0 * max((prev + cur) / 2.0, 1.0)))
    return {"delta": cur - prev, "noise": round(noise, 1), "within_noise": abs(cur - prev) <= noise}


def user_experiments(uid):
    with DB_LOCK:
        rows = DB.execute("SELECT id, factor, start_ts, days, status, stopped_ts, adherence FROM experiments "
                          "WHERE user_id = ? ORDER BY start_ts DESC, id DESC", (uid,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["adherence"] = json.loads(d["adherence"] or "{}")
        except ValueError:
            d["adherence"] = {}
        out.append(d)
    return out


def exp_state(exp, now):
    if exp["status"] == "stopped":
        return "stopped"
    return "done" if (_day(now) - _day(exp["start_ts"])).days + 1 > exp["days"] else "active"


def adherence_stats(exp, now):
    start = _day(exp["start_ts"])
    end_day = start + timedelta(days=exp["days"] - 1)
    if exp["status"] == "stopped" and exp.get("stopped_ts"):
        last = min(_day(exp["stopped_ts"]), end_day)
    else:
        last = min(_day(now), end_day)
    elapsed = max(1, (last - start).days + 1)
    logged = kept = 0
    for k, v in exp["adherence"].items():
        try:
            d = datetime.strptime(k, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start <= d <= last:
            logged += 1
            kept += int(bool(v))
    kept_pct = (kept / logged) if logged else None
    return {"elapsed": elapsed, "logged": logged, "kept": kept, "coverage": logged / elapsed, "kept_pct": kept_pct,
            "reliable": bool(logged >= 3 and logged / elapsed >= 0.6 and kept_pct is not None and kept_pct >= 0.7)}


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def _t975(df):
    df = max(1, int(math.floor(df)))
    return T975.get(df, 2.042 if df < 30 else 2.0)


def welch_range(base, dur):
    """95% interval for (mean during - mean before). Small samples, so Welch t with a Poisson variance floor
    (counts are at least as variable as their mean)."""
    b, d = np.asarray(base, float), np.asarray(dur, float)
    nb, nd = len(b), len(d)
    vb = max(float(b.var(ddof=1)), float(b.mean()), 0.25)
    vd = max(float(d.var(ddof=1)), float(d.mean()), 0.25)
    se2 = vb / nb + vd / nd
    df = se2 ** 2 / ((vb / nb) ** 2 / (nb - 1) + (vd / nd) ** 2 / (nd - 1))
    half = _t975(df) * math.sqrt(se2)
    diff = float(d.mean() - b.mean())
    return diff - half, diff + half


def base_trend_t(points):
    """t-statistic of the slope of the 'before' scans over time (needs 5+ scans). Negative = already improving."""
    if len(points) < 5:
        return None
    x = np.array([p[0] / 86400.0 for p in points])
    y = np.array([p[1] for p in points], float)
    x = x - x.mean()
    sxx = float((x ** 2).sum())
    if sxx == 0:
        return None
    slope = float((x * (y - y.mean())).sum() / sxx)
    resid = y - y.mean() - slope * x
    se = math.sqrt(max(float((resid ** 2).sum()) / (len(y) - 2), 1e-9) / sxx)
    return slope / se


def analyze_experiment(exp, ok_scans, breakouts, now):
    start, days = exp["start_ts"], exp["days"]
    end = start + days * 86400
    stop = exp.get("stopped_ts") if exp["status"] == "stopped" and exp.get("stopped_ts") else None
    upto = min(now, (stop or end) + 86400)
    lead = min(LEAD_IN_MAX, days // 2) * 86400
    base_pts = [(r["ts"], r["spots_total"]) for r in ok_scans if start - BASELINE_DAYS * 86400 <= r["ts"] < start]
    base = [v for _, v in base_pts]
    dur = [r["spots_total"] for r in ok_scans if start + lead <= r["ts"] <= upto]
    ad = adherence_stats(exp, now)
    out = {"n_base": len(base), "n_during": len(dur), "lead_in_days": lead // 86400, "adherence": ad,
           "mean_base": None, "mean_during": None, "diff": None, "low": None, "high": None, "pct": None,
           "verdict": "need_more", "text": "", "breakouts": None, "caveats": EXP_CAVEATS}
    if len(base) < MIN_SCANS_EACH or len(dur) < MIN_SCANS_EACH:
        out["text"] = (f"Not enough scans yet. A fair comparison needs at least {MIN_SCANS_EACH} good scans in the "
                       f"{BASELINE_DAYS} days before you started (you have {len(base)}) and {MIN_SCANS_EACH} after the "
                       f"{lead // 86400}-day lead-in (you have {len(dur)}). Scanning every 2 days makes the test far more sensitive.")
        return out
    mb, md = float(np.mean(base)), float(np.mean(dur))
    lo, hi = welch_range(base, dur)
    diff = md - mb
    out.update({"mean_base": round(mb, 1), "mean_during": round(md, 1), "diff": round(diff, 1),
                "low": round(lo, 1), "high": round(hi, 1), "pct": (round(100.0 * diff / mb) if mb > 0 else None)})
    dsp = max(1, (upto - (start + lead)) / 86400.0)
    nb = sum(1 for b in breakouts if start - BASELINE_DAYS * 86400 <= b["ts"] < start)
    nd = sum(1 for b in breakouts if start + lead <= b["ts"] <= upto)
    if dsp >= 7 and nb + nd >= 3:
        out["breakouts"] = {"before_per_week": round(nb / (BASELINE_DAYS / 7.0), 1), "during_per_week": round(nd / (dsp / 7.0), 1)}
    label = EXPERIMENTS.get(exp["factor"], {}).get("label", exp["factor"]).lower()
    if not ad["reliable"]:
        out["verdict"] = "unreliable"
        got = ("you have not logged any days" if ad["logged"] == 0
               else f"you kept to it on {ad['kept']} of the {ad['logged']} days you logged")
        out["text"] = (f"This is not a fair test yet: {got} ({ad['elapsed']} days so far). "
                       f"Answer Yes or No every day and keep to '{label}' on at least 70% of them.")
        return out
    need = max(1.0, 0.2 * mb)
    if hi < 0 and abs(diff) >= need:
        out["verdict"] = "better"
        out["text"] = (f"Active spots averaged {mb:.1f} before and {md:.1f} during ({diff:+.1f}). The 95% range of "
                       f"change is {lo:+.1f} to {hi:+.1f}, so this looks like a real improvement for you. To be surer it "
                       "was this change, stop it for a few weeks and see whether spots come back.")
    elif lo > 0 and abs(diff) >= need:
        out["verdict"] = "worse"
        out["text"] = (f"Active spots averaged {mb:.1f} before and {md:.1f} during ({diff:+.1f}), range {lo:+.1f} to "
                       f"{hi:+.1f}. Spots were higher, which is unlikely to be caused by '{label}'. Something else changed.")
    else:
        out["verdict"] = "unclear"
        out["text"] = (f"No clear difference: {mb:.1f} spots before and {md:.1f} during ({diff:+.1f}, 95% range "
                       f"{lo:+.1f} to {hi:+.1f}). Either '{label}' is not a trigger for you, the test was too short, or "
                       "scans were too noisy to see a small effect.")
    tt = base_trend_t(base_pts)
    if tt is not None and ((out["verdict"] == "better" and tt <= -1.5) or (out["verdict"] == "worse" and tt >= 1.5)):
        way = "falling" if tt < 0 else "rising"
        out["verdict"] = "unclear"
        out["text"] = (f"Spots averaged {mb:.1f} before and {md:.1f} during ({diff:+.1f}), but they were already {way} "
                       f"before you started, so this may have happened anyway. Treat it as unproven.")
    return out


def suggest_factor(user, breakouts):
    rows = lifestyle_insights(user_diary(user["id"]), breakouts)
    keymap = {"dairy": "dairy", "sugar": "sugar", "sleep_low": "sleep"}
    best = None
    for r in rows:
        if r["key"] in keymap and r["enough"] and r["rate_with"] is not None and r["rate_without"] is not None:
            gap = r["rate_with"] - r["rate_without"]
            if gap >= 0.15 and (best is None or gap > best[0]):
                best = (gap, keymap[r["key"]])
    if best:
        return {"key": best[1], "why": "Your diary shows more breakouts after days with this."}
    q = user["profile"].get("last_q", {}) or {}
    if q.get("dairy", 0) >= 3:
        return {"key": "dairy", "why": "You said you have dairy every day."}
    if q.get("sweets", 0) >= 2:
        return {"key": "sugar", "why": "You said you have sweets or sugary drinks 3+ days a week."}
    if q.get("fried", 0) >= 2:
        return {"key": "fried", "why": "You said you have fried or fast food 3+ days a week."}
    if q.get("sleep") is not None and q["sleep"] < 6.5:
        return {"key": "sleep", "why": "You said you sleep under 6.5 hours."}
    return None


def _exp_item(e, ok_scans, breakouts, now):
    st = exp_state(e, now)
    ad = adherence_stats(e, now)
    cfg = EXPERIMENTS.get(e["factor"], {"label": e["factor"], "ask": "Did you stick to it today?", "detail": "", "evidence": ""})
    today = date.today().isoformat()
    return {"id": e["id"], "factor": e["factor"], "label": cfg["label"], "ask": cfg["ask"], "days": e["days"],
            "start_ts": e["start_ts"], "state": st, "day": min(ad["elapsed"], e["days"]), "today": e["adherence"].get(today),
            "analysis": analyze_experiment(e, ok_scans, breakouts, now)}


def build_lab(user):
    now = time.time()
    ok_scans = [r for r in current_view(user_scans(user["id"])) if r["ok"]]
    bo = user_breakouts(user["id"])
    items = [_exp_item(e, ok_scans, bo, now) for e in user_experiments(user["id"])]
    active = next((i for i in items if i["state"] == "active"), None)
    return clean({"active": active, "past": [i for i in items if i["state"] != "active"],
                  "factors": [{"key": k, **{f: v[f] for f in ("label", "detail", "evidence")}} for k, v in EXPERIMENTS.items()],
                  "baseline_scans": sum(1 for r in ok_scans if now - BASELINE_DAYS * 86400 <= r["ts"] <= now),
                  "suggested": suggest_factor(user, bo), "caveats": EXP_CAVEATS,
                  "lead_in": min(LEAD_IN_MAX, 28 // 2), "min_scans": MIN_SCANS_EACH})


def lab_home(user):
    now = time.time()
    for e in user_experiments(user["id"]):
        if exp_state(e, now) == "active":
            i = _exp_item(e, [], [], now)
            return {"id": i["id"], "label": i["label"], "ask": i["ask"], "day": i["day"], "days": i["days"], "today": i["today"]}
    return None


def api_lab_start(user, payload):
    factor, days = payload.get("factor"), int(payload.get("days", 28))
    if factor not in EXPERIMENTS:
        raise ValueError("Unknown test.")
    if days not in (28, 42):
        raise ValueError("Choose 28 or 42 days.")
    now = time.time()
    if any(exp_state(e, now) == "active" for e in user_experiments(user["id"])):
        raise ValueError("Finish or stop your current test first. Changing one thing at a time keeps the result meaningful.")
    with DB_LOCK:
        DB.execute("INSERT INTO experiments (user_id, factor, start_ts, days) VALUES (?, ?, ?, ?)",
                   (user["id"], factor, now, days))
        DB.commit()
    return {"ok": True}


def api_lab_checkin(user, payload):
    now = time.time()
    act = next((e for e in user_experiments(user["id"]) if exp_state(e, now) == "active"), None)
    if not act:
        raise ValueError("No test is running.")
    day = str(payload.get("day") or date.today().isoformat())
    d = datetime.strptime(day, "%Y-%m-%d").date()
    if d > date.today() or d < _day(act["start_ts"]):
        raise ValueError("That day is outside your test.")
    adh = dict(act["adherence"])
    adh[day] = bool(payload.get("ok"))
    with DB_LOCK:
        DB.execute("UPDATE experiments SET adherence = ? WHERE id = ? AND user_id = ?", (json.dumps(adh), act["id"], user["id"]))
        DB.commit()
    return {"ok": True}


def api_lab_stop(user, payload):
    with DB_LOCK:
        DB.execute("UPDATE experiments SET status = 'stopped', stopped_ts = ? WHERE id = ? AND user_id = ? AND status = 'active'",
                   (time.time(), int(payload.get("id", 0)), user["id"]))
        DB.commit()
    return {"ok": True}


def build_summary(user):
    uid = user["id"]
    now = time.time()
    ok_scans = [r for r in current_view(user_scans(uid)) if r["ok"]]
    latest = ok_scans[-1] if ok_scans else None
    bo = user_breakouts(uid)
    cutoff = now - 90 * 86400
    diary = user_diary(uid)
    since = (date.today() - timedelta(days=29)).isoformat()
    routine_days = sum(1 for d in diary if d["day"] >= since and any((d.get("routine") or {}).values()))
    prof = user["profile"]
    q = (latest or {}).get("q") or prof.get("last_q", {}) or {}
    exps = [_exp_item(e, ok_scans, bo, now) for e in user_experiments(uid)]
    return clean({
        "name": user["name"], "generated": date.today().isoformat(),
        "profile": {"goal": prof.get("goal", ""), "diet": prof.get("diet", "omni"), "skin_type": prof.get("self_skin_type", ""),
                    "avoid": prof.get("avoid", "")},
        "scans": [{"ts": r["ts"], "score": r["score"], "grade": r["grade"], "spots": r["spots_total"],
                   "pustules": r["pustules_total"], "marks": r["marks_total"]} for r in ok_scans[-12:]],
        "latest_zones": ({z: {"spots": latest[f"{z}_spots"], "marks": latest[f"{z}_marks"]} for z in ZONES} if latest else None),
        "breakouts_90d": {z: sum(1 for b in bo if b["zone"] == z and b["ts"] >= cutoff) for z in ZONES},
        "routine_days_30": routine_days,
        "self_report": {k: q.get(k) for k in ("sleep", "stress", "water", "sunlight", "picking", "supps") if k in q},
        "triggers": [r for r in lifestyle_insights(diary, bo) if r["enough"]],
        "experiments": [{"label": i["label"], "days": i["days"], "state": i["state"], "start_ts": i["start_ts"],
                         "verdict": i["analysis"]["verdict"], "text": i["analysis"]["text"]} for i in exps if i["state"] != "active"],
        "labels": LABELS,
    })


FREQ = [[0, "Rarely"], [1, "1-2 days"], [2, "3-5 days"], [3, "Daily"]]
QUESTIONS = [
    {"group": "Lifestyle", "key": "sleep", "label": "Average sleep per night (hours)", "type": "number",
     "min": 3, "max": 12, "step": 0.5, "default": 7},
    {"group": "Lifestyle", "key": "stress", "label": "Stress level lately", "type": "choice",
     "options": [[1, "Very low"], [2, "Low"], [3, "Medium"], [4, "High"], [5, "Very high"]], "default": 2},
    {"group": "Lifestyle", "key": "water", "label": "Glasses of water per day", "type": "number",
     "min": 0, "max": 15, "step": 1, "default": 6},
    {"group": "Lifestyle", "key": "sunlight", "label": "Time outdoors in daylight (20+ minutes)", "type": "choice",
     "options": [[0, "Rarely"], [1, "Some days"], [2, "Most days"]], "default": 1},
    {"group": "Lifestyle", "key": "picking", "label": "Do you pick or squeeze pimples?", "type": "choice",
     "options": [[0, "Never"], [1, "Sometimes"], [2, "Often"]], "default": 0},
    {"group": "Lifestyle", "key": "new_product", "label": "Started a new skin or hair product in the last 2 weeks",
     "type": "bool", "default": False},
    {"group": "Food (days per week)", "key": "fish", "label": "Fish or seafood", "type": "freq"},
    {"group": "Food (days per week)", "key": "meat", "label": "Meat or poultry", "type": "freq"},
    {"group": "Food (days per week)", "key": "eggs", "label": "Eggs", "type": "freq"},
    {"group": "Food (days per week)", "key": "dairy", "label": "Milk, yoghurt, cheese, paneer", "type": "freq"},
    {"group": "Food (days per week)", "key": "nuts_seeds", "label": "Nuts and seeds", "type": "freq"},
    {"group": "Food (days per week)", "key": "legumes_grains", "label": "Beans, lentils, chickpeas, whole grains", "type": "freq"},
    {"group": "Food (days per week)", "key": "greens", "label": "Leafy or colourful vegetables", "type": "freq"},
    {"group": "Food (days per week)", "key": "fruit", "label": "Fresh fruit", "type": "freq"},
    {"group": "Food (days per week)", "key": "sweets", "label": "Sweets, desserts or sugary drinks", "type": "freq"},
    {"group": "Food (days per week)", "key": "fried", "label": "Fried or fast food", "type": "freq"},
    {"group": "Supplements", "key": "supps", "label": "Supplements you take", "type": "multi",
     "options": [["multivitamin", "Multivitamin"], ["zinc", "Zinc"], ["vitd", "Vitamin D"],
                 ["omega3", "Omega-3 / fish oil"], ["b12", "Vitamin B12"], ["iron", "Iron"], ["vitc", "Vitamin C"]],
     "default": []},
]
FREQ_KEYS = [q["key"] for q in QUESTIONS if q["type"] == "freq"]


def clean_q(q):
    out = {}
    if not isinstance(q, dict):
        return out
    for spec in QUESTIONS:
        k = spec["key"]
        if k not in q or q[k] is None:
            continue
        v, t = q[k], spec["type"]
        try:
            if t == "number":
                out[k] = min(max(float(v), spec["min"]), spec["max"])
            elif t == "choice":
                vals = [o[0] for o in spec["options"]]
                if int(v) in vals:
                    out[k] = int(v)
            elif t == "freq":
                if int(v) in (0, 1, 2, 3):
                    out[k] = int(v)
            elif t == "bool":
                out[k] = bool(v)
            elif t == "multi":
                allowed = {o[0] for o in spec["options"]}
                out[k] = [s for s in v if s in allowed] if isinstance(v, list) else []
        except (TypeError, ValueError):
            continue
    return out


def food_answers(q):
    return sum(1 for k in FREQ_KEYS if k in q)


GROUP_LABEL = {"meat": "meat or poultry", "fish": "fish", "eggs": "eggs", "dairy": "dairy",
               "nuts_seeds": "nuts and seeds", "legumes_grains": "beans, lentils and whole grains",
               "greens": "leafy or colourful vegetables", "fruit": "fresh fruit"}

# foods: (name, tags) - tags say which diets exclude it
NUTRIENTS = [
    {"key": "zinc", "name": "Zinc", "w": {"meat": 1.0, "nuts_seeds": 0.9, "legumes_grains": 0.7, "fish": 0.6,
                                          "eggs": 0.5, "dairy": 0.4}, "target": 5.0, "supp": ("zinc",),
     "multi": 0.8, "plant": {"veg": 0.9, "vegan": 0.8}, "rank": 1, "evidence": "Limited to moderate",
     "skin": "A 2020 review found people with acne tend to have lower zinc, and zinc by mouth reduced inflamed "
             "spots in trials. Most trials are small, it works less well than standard treatment and does not replace it.",
     "foods": [("Pumpkin seeds", ()), ("Chickpeas or lentils", ()), ("Cashews or almonds", ()),
               ("Tofu", ()), ("Beef or chicken", ("meat",)), ("Oysters or crab", ("fish",)),
               ("Eggs", ("egg",)), ("Yoghurt", ("dairy",))],
     "amount": "Adults need roughly 8 to 11 mg a day. The safe upper limit is about 40 mg a day.",
     "caution": "High-dose zinc supplements can cause nausea and, over time, low copper. Food first."},
    {"key": "omega3", "name": "Omega-3 fats (EPA and DHA)",
     "w": {"fish": 2.0, "nuts_seeds": 0.6, "eggs": 0.2}, "target": 3.0, "supp": ("omega3",), "multi": 0.0,
     "plant": {}, "rank": 2, "evidence": "Limited",
     "skin": "Omega-3 fats are anti-inflammatory and a few small trials show fewer inflamed spots, "
             "but the evidence is limited.",
     "foods": [("Salmon, sardines or mackerel (2 servings a week)", ("fish",)), ("Walnuts", ()),
               ("Chia or ground flaxseed", ()), ("Algae-based omega-3 supplement", ())],
     "amount": "Health bodies commonly suggest 2 portions of fish a week, one of them oily.",
     "caution": "Fish oil can interact with blood thinners. Ask a doctor first if you take any."},
    {"key": "vitamin_d", "name": "Vitamin D", "w": {"fish": 0.6, "eggs": 0.3, "dairy": 0.3}, "target": 3.0,
     "supp": ("vitd",), "multi": 0.7, "plant": {}, "rank": 3, "evidence": "Limited",
     "skin": "Reviews consistently find lower vitamin D in people with acne, more so in severe acne. A few small "
             "trials suggest supplements help people who are low, but it is not proven to treat acne on its own.",
     "foods": [("Safe daylight on face and arms most days", ()), ("Oily fish", ("fish",)), ("Egg yolks", ("egg",)),
               ("Fortified milk or plant milk", ()), ("UV-exposed mushrooms", ())],
     "amount": "Adults need roughly 10 to 15 micrograms (400 to 600 IU) a day. The upper limit is 100 micrograms.",
     "caution": "In many countries advice is to consider a daily supplement in autumn and winter. Check your local guidance."},
    {"key": "vitamin_a", "name": "Vitamin A (carotenoids)", "w": {"greens": 1.0, "eggs": 0.5, "dairy": 0.3,
                                                                  "fruit": 0.3}, "target": 3.5,
     "supp": (), "multi": 0.8, "plant": {}, "rank": 4, "evidence": "Weak for food; strong for prescription retinoids",
     "skin": "Vitamin A supports normal skin renewal. Retinoid medicines work for acne, but eating more vitamin A "
             "has not been shown to treat it.",
     "foods": [("Sweet potato", ()), ("Carrots", ()), ("Spinach or kale", ()), ("Mango or papaya", ()),
               ("Eggs", ("egg",))],
     "amount": "Adults need roughly 700 to 900 micrograms RAE a day.",
     "caution": "Do not take high-dose vitamin A supplements. Too much is harmful, especially in pregnancy."},
    {"key": "vitamin_c", "name": "Vitamin C", "w": {"fruit": 1.2, "greens": 0.8}, "target": 3.0,
     "supp": ("vitc",), "multi": 0.7, "plant": {}, "rank": 5, "evidence": "Weak",
     "skin": "Vitamin C helps skin repair and may help marks fade. Eating more of it has not been shown to "
             "affect acne, so it is not an acne treatment.",
     "foods": [("Oranges, kiwi or guava", ()), ("Strawberries", ()), ("Bell peppers", ()),
               ("Amla (Indian gooseberry)", ()), ("Broccoli", ())],
     "amount": "Adults need roughly 75 to 90 mg a day. The upper limit is 2,000 mg.",
     "caution": "Food is enough for most people. Very high supplement doses can upset the stomach."},
    {"key": "b12", "name": "Vitamin B12", "w": {"meat": 0.9, "fish": 0.9, "eggs": 0.7, "dairy": 0.8},
     "target": 3.0, "supp": ("b12",), "multi": 0.8, "plant": {}, "rank": 6, "evidence": "Not an acne factor",
     "skin": "Low B12 is not a known cause of acne. It matters for energy and blood, and vegans are at higher risk. "
             "Very high B12 doses can trigger breakouts in some people.",
     "foods": [("Eggs", ("egg",)), ("Milk, yoghurt or paneer", ("dairy",)), ("Fish", ("fish",)),
               ("Meat or poultry", ("meat",)), ("Fortified cereal, plant milk or nutritional yeast", ())],
     "amount": "Adults need about 2.4 micrograms a day.",
     "caution": "Only supplement if a blood test or your doctor says you need it. Vegans should ask about a B12 supplement."},
    {"key": "iron", "name": "Iron", "w": {"meat": 1.0, "legumes_grains": 0.7, "greens": 0.6, "eggs": 0.3,
                                          "fish": 0.3}, "target": 4.0, "supp": ("iron",), "multi": 0.6,
     "plant": {"veg": 0.85, "vegan": 0.75}, "rank": 7, "evidence": "Not an acne factor",
     "skin": "Small studies on iron and acne disagree, and none show a clear link. It is included because low "
             "iron is common and causes tiredness, paleness and hair shedding.",
     "foods": [("Lentils or chickpeas", ()), ("Spinach with a squeeze of lemon", ()), ("Tofu", ()),
               ("Beef or chicken", ("meat",)), ("Iron-fortified cereal", ())],
     "amount": "Adults need roughly 8 to 18 mg a day depending on age and sex.",
     "caution": "Do not take iron supplements unless a blood test shows you need them. Too much is harmful."},
]

DIET_EXCLUDES = {"omni": (), "veg": ("meat", "fish"), "vegan": ("meat", "fish", "egg", "dairy")}


def _diet_foods(n, diet):
    ex = DIET_EXCLUDES.get(diet, ())
    return [name for name, tags in n["foods"] if not any(t in ex for t in tags)]


def nutrient_index(n, q, diet):
    f = lambda k: q.get(k, 0)
    if n["key"] == "vitamin_d":
        sun = {0: 0.0, 1: 1.0, 2: 2.0}.get(q.get("sunlight", 1), 1.0)
        idx = (sun + 0.6 * f("fish") + 0.3 * f("eggs") + 0.3 * f("dairy")) / n["target"]
    else:
        idx = sum(w * f(k) for k, w in n["w"].items()) / n["target"]
    idx *= n["plant"].get(diet, 1.0)
    supps = set(q.get("supps", []))
    if n["key"] == "omega3" and diet in ("veg", "vegan") and "omega3" not in supps:
        idx = min(idx, 0.5 if diet == "vegan" else 0.6)
    if n["key"] == "b12" and diet == "vegan" and not ({"b12", "multivitamin"} & supps):
        idx = 0.1
    if set(n["supp"]) & supps:
        idx = max(idx, 1.0)
    elif "multivitamin" in supps and n["multi"]:
        idx = max(idx, n["multi"])
    return min(idx, 1.0)


def nutrition_watch(q, diet):
    have = food_answers(q) >= 6
    out = []
    for n in NUTRIENTS:
        item = {"key": n["key"], "name": n["name"], "evidence": n["evidence"], "skin": n["skin"],
                "amount": n["amount"], "caution": n["caution"], "foods": _diet_foods(n, diet), "rank": n["rank"]}
        if not have:
            item.update({"likelihood": "unknown", "why": "Answer the food questions on the Scan page to see this."})
        else:
            idx = nutrient_index(n, q, diet)
            if idx < 0.35:
                lk = "likely_low"
            elif idx < 0.65:
                lk = "possibly_low"
            else:
                lk = "likely_fine"
            rare = [GROUP_LABEL[k] for k, w in sorted(n["w"].items(), key=lambda kv: -kv[1])
                    if k in GROUP_LABEL and q.get(k, 0) <= 1
                    and not (diet in ("veg", "vegan") and k in ("meat", "fish"))
                    and not (diet == "vegan" and k in ("eggs", "dairy"))][:3]
            if lk == "likely_fine":
                why = "Your answers suggest you get enough from food or supplements."
            elif lk == "possibly_low":
                why = "You get some, but not consistently."
            else:
                why = "Your answers suggest few of the main sources."
            if lk != "likely_fine" and rare:
                why += " Foods you eat rarely: " + ", ".join(rare) + "."
            if n["key"] == "b12" and diet == "vegan" and lk != "likely_fine":
                why = "Plant foods do not contain B12 naturally, so vegans usually need fortified foods or a supplement."
            item.update({"likelihood": lk, "why": why, "score": round(idx, 2)})
        out.append(item)
    order = {"likely_low": 0, "possibly_low": 1, "unknown": 2, "likely_fine": 3}
    out.sort(key=lambda i: (order[i["likelihood"]], i["rank"]))
    return out


SAMPLE_DAY = {
    "omni": ["Breakfast: eggs or overnight oats with berries and a handful of walnuts",
             "Lunch: lentil or chicken and vegetable bowl with brown rice and a squeeze of lemon",
             "Snack: an orange or guava with pumpkin seeds",
             "Dinner: baked salmon or sardines with sweet potato and steamed greens"],
    "veg": ["Breakfast: oats with chia seeds, berries and walnuts",
            "Lunch: chickpea and spinach curry with brown rice and lemon",
            "Snack: guava or an orange with pumpkin seeds and yoghurt",
            "Dinner: paneer or tofu stir-fry with peppers and broccoli"],
    "vegan": ["Breakfast: tofu scramble with spinach, and fortified plant milk",
              "Lunch: lentil dal with vegetables and brown rice, with lemon",
              "Snack: an orange with pumpkin seeds and a few walnuts",
              "Dinner: chickpea and sweet potato bowl with tahini. Ask a doctor about B12 and algae omega-3."],
}

TIPS = [
    "Change your pillowcase at least once a week and wipe your phone screen daily.",
    "Wash your face after a sweaty workout. Sweat and friction can trigger spots.",
    "Hands off: picking spots raises the chance of marks and scars.",
    "Sunscreen every morning helps dark marks fade faster.",
    "Introduce one new skincare product at a time, two weeks apart.",
    "Give a new treatment 8 to 12 weeks before judging it.",
    "Keep hair products and oils off your forehead and hairline.",
    "Aim for 7 to 9 hours of sleep. Poor sleep tends to go with more breakouts.",
    "Use a gentle, fragrance-free cleanser twice a day at most. Over-washing irritates skin.",
    "Take your scan photos in the same daylight spot each time for fair comparisons.",
    "Drink water through the day, and keep sugary drinks occasional.",
    "Check ingredient lists for anything you know you react to.",
    "Do not use toothpaste, lemon juice or baking soda on spots. They irritate skin.",
    "Log every new breakout in the Diary. It teaches the app what triggers yours.",
]


def routine_plan(m, q, profile, grade, skin_type):
    preg = profile.get("pregnancy", "unknown")
    oily = skin_type in ("Oily", "Combination") or profile.get("self_skin_type") in ("oily", "combination")
    dry = profile.get("self_skin_type") == "dry"
    marks = m["marks_total"]
    cleanser = ("Gel or foaming gentle cleanser" if oily else "Gentle cream or lotion cleanser" if dry
                else "Gentle, fragrance-free cleanser")
    am = [{"step": cleanser, "why": "Wash once in the morning. Use lukewarm water and pat dry."},
          {"step": "Niacinamide 2-5% serum (optional)",
           "why": "Can help oil, redness and marks. Introduce it on its own first."},
          {"step": "Oil-free moisturiser", "why": "Even oily skin needs moisture. Look for 'non-comedogenic'."},
          {"step": "Sunscreen SPF 30+ (non-comedogenic)",
           "why": "Sun makes dark marks last longer, and some acne treatments make skin more sun-sensitive."}]
    if marks >= 3:
        am.insert(2, {"step": "Vitamin C serum (optional)", "why": "May help dark marks fade. Start slowly."})
    pm = [{"step": cleanser, "why": "Wash away sunscreen, oil and dirt. Do not scrub."}]
    treat = []
    if grade == "Clear / almost clear":
        treat.append({"step": "Maintenance: no strong active needed",
                      "why": "Keep the basics steady. Use a hydrocolloid spot patch on any new pimple."})
    else:
        treat.append({"step": "Benzoyl peroxide 2.5-5% (wash or spot)",
                      "why": "Kills acne bacteria and helps inflamed spots. Can bleach fabric and dry skin."})
        treat.append({"step": "Salicylic acid 0.5-2%",
                      "why": "Helps clogged pores and blackheads. A gentler option if benzoyl peroxide is too drying."})
        if preg == "no":
            treat.append({"step": "Adapalene 0.1% gel (start every 2-3 nights)",
                          "why": "A retinoid that helps prevent new spots. Expect dryness and possibly more spots for the first weeks. Whether it is sold without a prescription depends on your country, so ask a pharmacist."})
        else:
            treat.append({"step": "Retinoids: ask your doctor first",
                          "why": "Skip retinoids if you are pregnant, planning pregnancy or breastfeeding, unless a doctor advises otherwise."})
        treat.append({"step": "Start with ONE of these, not all",
                      "why": "Layering several strong actives is the fastest way to irritate skin."})
    if marks >= 3:
        treat.append({"step": "Azelaic acid (availability varies by country)",
                      "why": "Can help both spots and dark marks and is usually gentle. Ask a pharmacist."})
    pm.extend(treat)
    pm.append({"step": "Moisturiser", "why": "Helps skin cope with treatments."})
    avoid = ["Picking or squeezing spots", "Harsh scrubs and hot water",
             "Washing more than twice a day", "Toothpaste, lemon juice or baking soda on spots",
             "Heavy oils and pomades on the forehead or hairline",
             "Adding several new actives at once"]
    safety = ["Patch test any new product on a small area for 2 to 3 days first.",
              "Introduce one new active at a time, about two weeks apart.",
              "Expect some dryness in the first 2 to 4 weeks. Reduce how often you use it if skin stings or peels.",
              "If you are under 18, talk to a parent or doctor before starting actives like benzoyl peroxide or adapalene."]
    if preg != "no":
        safety.insert(0, "If you are pregnant, planning a pregnancy or breastfeeding, avoid retinoids and check every "
                          "acne product with a doctor or pharmacist first.")
    if profile.get("avoid"):
        safety.append("You told us you react to: " + profile["avoid"][:120] + ". Check labels for these.")
    return {"am": am, "pm": pm, "avoid": avoid, "safety": safety}


def doctor_flags(m, grade, history_trend):
    reasons = []
    if grade in ("Moderate", "Severe"):
        reasons.append(f"Your scan shows about {m['spots_total']} active spots ({grade.lower()}). "
                       "A doctor or dermatologist can offer stronger treatments than shop-bought ones.")
    if m["pustules_total"] >= 5:
        reasons.append("Several pustule-like spots can mean active inflammation that is worth getting checked.")
    if history_trend is not None and history_trend >= 4:
        reasons.append("Your active spots have risen across recent scans.")
    general = ["Painful, deep or cyst-like lumps",
               "Acne that leaves scars or dark marks that are not fading",
               "No improvement after 8 to 12 weeks of consistent treatment",
               "Sudden severe acne as an adult, or acne with other symptoms such as very irregular periods",
               "Acne that is affecting your mood or confidence"]
    return {"needed": bool(reasons), "reasons": reasons, "general": general}


def lifestyle_plan(q, m):
    out = []
    if q.get("sleep") is not None and q["sleep"] < 7:
        out.append({"title": "Sleep more", "text": "You sleep about %.1f hours. Aim for 7 to 9. Poor sleep tends to go with more breakouts." % q["sleep"]})
    if q.get("stress", 0) >= 4:
        out.append({"title": "Lower stress", "text": "Stress raises oil-producing hormones. Try daily walks, breathing exercises or a fixed wind-down time."})
    if q.get("water") is not None and q["water"] < 6:
        out.append({"title": "Drink more water", "text": "You drink about %d glasses. Aim for 6 to 8 spread through the day." % q["water"]})
    if q.get("picking", 0) >= 1:
        out.append({"title": "Hands off", "text": "Picking spreads inflammation and causes marks and scars. Use a hydrocolloid patch instead."})
    if q.get("sweets", 0) >= 2:
        out.append({"title": "Cut back on sugar", "text": "You have sweets or sugary drinks 3 or more days a week. Low-glycaemic eating has some evidence for fewer spots. Try 1 to 2 days a week for 8 weeks."})
    if q.get("dairy", 0) >= 3:
        out.append({"title": "Test your dairy", "text": "Some people break out more with a lot of milk or whey protein. Try cutting it for 4 to 8 weeks and see. Keep calcium and B12 from other foods."})
    if q.get("fried", 0) >= 2:
        out.append({"title": "Ease off fried food", "text": "Evidence is weak, but fried and fast food usually crowds out vegetables and fish."})
    if q.get("new_product"):
        out.append({"title": "Suspect the new product", "text": "If spots increased after starting something new, pause it and see whether things settle."})
    out.append({"title": "Hygiene basics", "text": "Change your pillowcase weekly, wipe your phone daily, wash after sweating, and keep hair products off your forehead."})
    return out


def build_concerns(m, skin_type):
    sc = skin_score(m)

    def lvl(v):
        return "good" if v >= 80 else "fair" if v >= 55 else "care"

    spots, pust, marks = m["spots_total"], m["pustules_total"], m["marks_total"]
    rels = [m[f"{z}_rel"] for z in ZONES]
    sd = float(np.std(rels))
    even_txt = "Very even" if sd < 0.5 else "Slightly uneven" if sd < 1.0 else "Uneven"
    t = (m["forehead_shine"] + m["nose_shine"]) / 2.0
    c = (m["left_cheek_shine"] + m["right_cheek_shine"]) / 2.0
    return [
        {"key": "acne", "label": "Acne", "score": sc["s_acne"], "level": lvl(sc["s_acne"]),
         "detail": f"{spots} active spots ({spots - pust} inflamed, {pust} pustule-like)"},
        {"key": "marks", "label": "Dark marks", "score": sc["s_marks"], "level": lvl(sc["s_marks"]),
         "detail": f"{marks} dark marks (can include freckles or moles)"},
        {"key": "oil", "label": "Oiliness", "score": sc["s_oil"], "level": lvl(sc["s_oil"]),
         "detail": f"Shine: T-zone {t:.1f}%, cheeks {c:.1f}%. Skin type estimate: {skin_type}"},
        {"key": "even", "label": "Redness evenness", "score": sc["s_even"], "level": lvl(sc["s_even"]),
         "detail": f"{even_txt} redness across zones"},
    ]


def compute_outlook(m, q, history):
    drivers = []
    score = [0.0]

    def add(pts, text):
        drivers.append({"text": text, "pts": pts, "dir": "up" if pts > 0 else "down"})
        score[0] += pts

    spots, pust, marks = m["spots_total"], m["pustules_total"], m["marks_total"]
    acne_pts = spots + 0.5 * pust
    if acne_pts >= 25:
        add(55, f"Many active spots right now ({spots})")
    elif acne_pts >= 10:
        add(45, f"Several active spots right now ({spots})")
    elif acne_pts >= 3:
        add(30, f"A few active spots right now ({spots})")
    elif acne_pts >= 1:
        add(15, f"{spots} active spot(s) right now")
    if pust >= 2:
        add(8, f"{pust} pustule-like spots (active inflammation)")
    oily = (m["forehead_shine"] + m["nose_shine"]) / 2.0
    if oily >= 6:
        add(8, "Oily shine in the T-zone")
    if float(np.std([m[f"{z}_rel"] for z in ZONES])) >= 1.0:
        add(5, "Uneven redness across the face")
    trend = None
    if len(history) >= 3:
        prev = float(np.median([r["spots_total"] + 0.5 * r.get("pustules_total", 0) for r in history[-3:]]))
        trend = acne_pts - prev
        if trend >= 3:
            add(12, "Active spots have increased since your recent scans")
        elif trend <= -3:
            add(-8, "Active spots have dropped since your recent scans")
    if q:
        if q.get("sleep") is not None:
            if q["sleep"] < 6:
                add(6, "You sleep under 6 hours")
            elif q["sleep"] >= 7.5:
                add(-3, "Good sleep")
        if q.get("stress", 0) >= 4:
            add(6, "High stress")
        if q.get("sweets", 0) >= 2:
            add(6, "Sweets or sugary drinks 3+ days a week")
        if q.get("dairy", 0) >= 3:
            add(4, "Dairy every day")
        if q.get("fried", 0) >= 2:
            add(4, "Fried or fast food 3+ days a week")
        if q.get("picking", 0) >= 2:
            add(6, "You often pick spots")
        elif q.get("picking", 0) == 1:
            add(3, "You sometimes pick spots")
        if q.get("new_product"):
            add(4, "A new product started recently")
        if q.get("fish", 0) >= 2:
            add(-3, "Fish 3+ days a week (omega-3)")
        if q.get("greens", 0) >= 2 and q.get("fruit", 0) >= 2:
            add(-3, "Plenty of vegetables and fruit")
    val = int(min(100, max(0, round(score[0]))))
    level = "low" if val < 30 else "moderate" if val < 55 else "high"
    have_q = bool(q) and food_answers(q) >= 6
    if len(history) >= BASE_MIN + 1 and have_q:
        conf = "Higher: uses your own scan history and lifestyle answers"
    elif have_q or len(history) >= 3:
        conf = "Medium: based on this scan plus " + ("your lifestyle answers" if have_q else "your recent scans")
    else:
        conf = "Low: first scan with no lifestyle answers. Answer the quick questions and scan regularly to improve it"
    return {"level": level, "score": val, "drivers": drivers, "confidence": conf, "trend": trend,
            "note": "This is an estimate from published links between lifestyle and acne plus what your photo shows. "
                    "It is not a validated predictor. The Insights page tests it against your own logged breakouts."}


def zone_outlook(m, history, cur_scan):
    out = {}
    if len(history) >= BASE_MIN + 1:
        past, recent = history[:-1], [history[-1], cur_scan]
        for z in ZONES:
            sc, comps = zone_score(z, past, recent)
            reasons = []
            if comps["rel"]["z"] >= 1.0:
                reasons.append(f"redness {comps['rel']['cur'] - comps['rel']['med']:+.1f} vs your usual")
            ds = comps["spots"]["cur"] - comps["spots"]["med"]
            if comps["spots"]["z"] >= 1.0 and ds >= 0.5:
                reasons.append(f"{ds:+.1f} more spots than usual")
            if comps["shine"]["z"] >= 1.0:
                reasons.append("oilier than usual")
            out[z] = {"level": level_for(sc), "score": round(sc, 2), "reasons": reasons, "basis": "your history"}
    else:
        for z in ZONES:
            n = m[f"{z}_spots"]
            reasons = [f"{n} active spot(s) here"] if n else []
            if m[f"{z}_shine"] >= 6:
                reasons.append("oily shine")
            out[z] = {"level": "low" if n == 0 else "watch" if n <= 2 else "high", "score": None,
                      "reasons": reasons, "basis": "this scan only"}
    return out


def build_result(uid, scan_id):
    user = get_user(uid)
    scans = user_scans(uid)
    scan = next((s for s in scans if s["id"] == scan_id), None)
    if not scan:
        raise KeyError("scan not found")
    profile = user["profile"]
    hist = [r for r in scans if r.get("v") == ALGO_VERSION and r.get("sens", 1) == scan.get("sens", 1)
            and r.get("det", "rules") == scan.get("det", "rules") and r["ok"] and (r["ts"], r["id"]) < (scan["ts"], scan["id"])]
    q = scan.get("q", {}) or {}
    diet = profile.get("diet", "omni")
    skin_type = estimate_skin_type(scan)
    concerns = build_concerns(scan, skin_type)
    outlook = compute_outlook(scan, q, hist)
    outlook["zones"] = zone_outlook(scan, hist, scan)
    nutrition = nutrition_watch(q, diet)
    flagged = [n for n in nutrition if n["likelihood"] in ("likely_low", "possibly_low")]
    eat_more, grocery, seen = [], [], set()
    for n in flagged:
        foods = n["foods"][:3]
        eat_more.append({"nutrient": n["name"], "foods": foods})
        for f in foods:
            if f not in seen:
                seen.add(f)
                grocery.append({"key": f, "label": f, "for": n["name"]})
    limit = []
    if q.get("sweets", 0) >= 1 or not q:
        limit.append({"food": "Sweets and sugary drinks", "evidence": "Moderate",
                      "text": "Low-glycaemic eating has been linked to fewer spots in trials."})
    if q.get("dairy", 0) >= 2 or not q:
        limit.append({"food": "Lots of milk or whey protein", "evidence": "Limited to moderate",
                      "text": "Some studies link them to more acne. Try cutting back for 4 to 8 weeks to see if you notice a difference."})
    if q.get("fried", 0) >= 2 or not q:
        limit.append({"food": "Fried and ultra-processed food", "evidence": "Weak",
                      "text": "The link is weak, but these often crowd out fish, vegetables and fruit."})
    grade = scan["grade"]
    prev_ok = hist[-1] if hist else None
    trend = outlook.get("trend")
    result = {
        "scan": {"id": scan["id"], "ts": scan["ts"], "ok": scan["ok"], "views": scan.get("views", {}),
                 "detector": scan.get("det", "rules"),
                 "notes": scan.get("quality_notes", []), "lesions": scan.get("lesions", []),
                 "have_answers": food_answers(q) >= 6},
        "score": {"overall": scan["score"], "grade": grade, "skin_type": skin_type,
                  "parts": {"acne": scan["s_acne"], "marks": scan["s_marks"], "oil": scan["s_oil"], "even": scan["s_even"]}},
        "counts": {"spots": scan["spots_total"], "pustules": scan["pustules_total"], "marks": scan["marks_total"],
                   "inflamed": scan["spots_total"] - scan["pustules_total"],
                   "by_zone": {z: {"spots": scan[f"{z}_spots"], "marks": scan[f"{z}_marks"],
                                   "rel": scan[f"{z}_rel"], "shine": scan[f"{z}_shine"]} for z in ZONES}},
        "concerns": concerns,
        "outlook": outlook,
        "nutrition": nutrition,
        "plan": {"eat_more": eat_more, "limit": limit, "sample_day": SAMPLE_DAY.get(diet, SAMPLE_DAY["omni"]),
                 "grocery": grocery, "lifestyle": lifestyle_plan(q, scan),
                 "routine": routine_plan(scan, q, profile, grade, skin_type),
                 "doctor": doctor_flags(scan, grade, trend)},
        "progress": {"change": (change_note(prev_ok["spots_total"], scan["spots_total"]) if prev_ok else None),
                     "previous": ({"ts": prev_ok["ts"], "score": prev_ok["score"], "spots": prev_ok["spots_total"]}
                                  if prev_ok else None),
                     "series": [{"ts": r["ts"], "score": r["score"], "spots": r["spots_total"]}
                                for r in (hist + [scan])[-12:]]},
        "grocery_state": profile.get("grocery", {}),
        "diet": diet,
    }
    return clean(result)


# ----------------------------------------------------------------------------
# Home, history, diary, insights
# ----------------------------------------------------------------------------
def weekly_report(ok_r, breakouts):
    now = time.time()
    wk = [r for r in ok_r if r["ts"] >= now - 7 * 86400]
    pv = [r for r in ok_r if now - 14 * 86400 <= r["ts"] < now - 7 * 86400]
    if not wk:
        return {"ready": False, "text": "Scan this week to get a weekly summary."}

    def av(rs, k):
        return float(np.mean([r[k] for r in rs])) if rs else None

    def cmp(cur, prev, digits=1):
        if prev is None:
            return ""
        return f" (last week {prev:.{digits}f}, {cur - prev:+.{digits}f})"

    sc, sp = av(wk, "score"), av(pv, "score")
    ls, lp = av(wk, "spots_total"), av(pv, "spots_total")
    lines = [f"Scans this week: {len(wk)}", f"Skin score: {sc:.0f}" + cmp(sc, sp, 0),
             f"Active spots per scan: {ls:.1f}" + cmp(ls, lp)]
    nb = sum(1 for b in breakouts if b["ts"] >= now - 7 * 86400)
    lines.append(f"Breakouts you logged: {nb}")
    return {"ready": True, "text": "\n".join(lines), "score": sc, "score_prev": sp}


def routine_streak(diary):
    days = {d["day"] for d in diary if any((d.get("routine") or {}).values())}
    d = date.today()
    if d.isoformat() not in days:
        d -= timedelta(days=1)
    n = 0
    while d.isoformat() in days:
        n += 1
        d -= timedelta(days=1)
    return n


def build_home(user):
    uid = user["id"]
    scans = user_scans(uid)
    cur = current_view(scans)
    ok_r = [r for r in cur if r["ok"]]
    diary = user_diary(uid)
    breakouts = user_breakouts(uid)
    latest = ok_r[-1] if ok_r else None
    today = date.today().isoformat()
    today_d = next((d for d in diary if d["day"] == today), {})
    days_since = int((time.time() - latest["ts"]) // 86400) if latest else None
    profile = user["profile"]
    result = None
    if latest:
        result = {"id": latest["id"], "ts": latest["ts"], "score": latest["score"], "grade": latest["grade"],
                  "spots": latest["spots_total"], "marks": latest["marks_total"],
                  "outlook": build_result(uid, latest["id"])["outlook"]["level"]}
    return clean({
        "name": user["name"], "latest": result, "days_since": days_since,
        "next_due": (max(0, 7 - days_since) if days_since is not None else 0),
        "routine_streak": routine_streak(diary), "today_routine": today_d.get("routine") or {},
        "weekly": weekly_report(ok_r, breakouts), "tip": TIPS[date.today().toordinal() % len(TIPS)],
        "recent": [{"id": r["id"], "ts": r["ts"], "score": r["score"], "grade": r["grade"],
                    "spots": r["spots_total"], "ok": r["ok"]} for r in cur[-4:][::-1]],
        "scan_count": len(scans), "lab": lab_home(user),
        "profile_done": bool(profile.get("last_q")) or bool(profile.get("goal")),
    })


def build_history(uid):
    scans = user_scans(uid)
    cur = current_view(scans)
    return clean({
        "scans": [{"id": r["id"], "ts": r["ts"], "ok": r["ok"], "score": r["score"], "grade": r["grade"],
                   "spots": r["spots_total"], "pustules": r["pustules_total"], "marks": r["marks_total"],
                   "lesions": r.get("lesions", []), "by_zone": {z: r[f"{z}_spots"] for z in ZONES}}
                  for r in cur][-60:],
        "hidden": len(scans) - len(cur), "labels": LABELS, "zones": ZONES,
    })


def build_diary(uid):
    diary = user_diary(uid)
    return clean({"entries": diary[-60:], "breakouts": user_breakouts(uid)[-100:], "streak": routine_streak(diary),
                  "labels": LABELS, "zones": ZONES})


def build_insights(uid):
    scans = user_scans(uid)
    ok_r = [r for r in current_view(scans) if r["ok"]]
    breakouts = user_breakouts(uid)
    return clean({"validation": backtest(ok_r, breakouts),
                  "triggers": lifestyle_insights(user_diary(uid), breakouts),
                  "detector": review_summary(user_reviews(uid)), "scans": len(ok_r)})


# ----------------------------------------------------------------------------
# API actions
# ----------------------------------------------------------------------------
def api_signup(payload):
    uid = create_user(payload.get("name"), payload.get("email"), payload.get("password"))
    return uid


def api_check(user, payload):
    img = decode_image(payload.get("image"))
    sens = int(user["profile"].get("sens", 1))
    res = analyze_photo(img, sens if sens in SENS_TABLE else 1)
    return {"found": res["found"], "notes": res.get("notes", []), "faces": res.get("faces", 0)}


def api_scan(user, payload):
    photos = payload.get("photos") or {}
    if "front" not in photos:
        raise ValueError("A front photo is required.")
    profile = user["profile"]
    sens = int(payload.get("sens", profile.get("sens", 1)))
    if sens not in SENS_TABLE:
        sens = 1
    results, view_out = {}, {}
    for view in VIEWS:
        if view in photos and photos[view]:
            res = analyze_photo(decode_image(photos[view]), sens)
            results[view] = res
            view_out[view] = {"found": res["found"], "notes": res.get("notes", []), "w": res["w"], "h": res["h"],
                              "box": res.get("box"), "lesions": [{"x": l["x"], "y": l["y"], "r": l["r"], "t": l["t"],
                                                                  "z": l["z"]} for l in res.get("lesions", [])],
                              "zones": {z: q["box"] for z, q in res.get("zones", {}).items()}}
    front = results.get("front")
    if not front or not front["found"]:
        raise ValueError("No face found in the front photo. Use a straight-on, well-lit photo with your whole face "
                         "in the frame.")
    metrics, packed = combine_views(results)
    q = clean_q(payload.get("q"))
    notes = quality_notes(metrics)
    ok = not notes
    data = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}
    data.update(skin_score(data))
    data.update({"v": ALGO_VERSION, "sens": sens, "det": front.get("detector", "rules"), "lesions": packed, "q": q, "quality_notes": notes,
                 "views": {v: {"found": r["found"], "notes": r.get("notes", [])} for v, r in results.items()}})
    ts = time.time()
    day = payload.get("day")
    if day and day != date.today().isoformat():
        ts = datetime.strptime(str(day), "%Y-%m-%d").replace(hour=12).timestamp()
        ts = min(ts, time.time())
    with DB_LOCK:
        cur = DB.execute("INSERT INTO scans (user_id, ts, ok, data) VALUES (?, ?, ?, ?)",
                         (user["id"], ts, int(ok), json.dumps(data)))
        if q:
            prof = {**profile, "last_q": {**profile.get("last_q", {}), **q}}
            DB.execute("UPDATE users SET profile = ? WHERE id = ?", (json.dumps(prof), user["id"]))
        DB.commit()
        sid = cur.lastrowid
    return {"id": sid, "views": view_out, "result": build_result(user["id"], sid)}


def api_profile(user, payload):
    prof = dict(user["profile"])
    if "diet" in payload and payload["diet"] in ("omni", "veg", "vegan"):
        prof["diet"] = payload["diet"]
    if "pregnancy" in payload and payload["pregnancy"] in ("no", "yes", "unknown"):
        prof["pregnancy"] = payload["pregnancy"]
    if "self_skin_type" in payload and payload["self_skin_type"] in ("", "oily", "dry", "combination", "normal", "sensitive"):
        prof["self_skin_type"] = payload["self_skin_type"]
    if "goal" in payload:
        prof["goal"] = str(payload["goal"])[:60]
    if "avoid" in payload:
        prof["avoid"] = str(payload["avoid"])[:120]
    if "sens" in payload and int(payload["sens"]) in SENS_TABLE:
        prof["sens"] = int(payload["sens"])
    if isinstance(payload.get("grocery"), dict):
        g = dict(prof.get("grocery", {}))
        for k, v in list(payload["grocery"].items())[:80]:
            g[str(k)[:80]] = bool(v)
        prof["grocery"] = g
    name = user["name"]
    if payload.get("name"):
        name = str(payload["name"]).strip()[:60] or name
    with DB_LOCK:
        DB.execute("UPDATE users SET profile = ?, name = ? WHERE id = ?", (json.dumps(prof), name, user["id"]))
        DB.commit()
    return {"ok": True}


ROUTINE_KEYS = ("am_cleanse", "spf", "pm_cleanse", "treatment", "moisturiser")


def api_diary(user, payload):
    day = str(payload.get("day", ""))
    datetime.strptime(day, "%Y-%m-%d")
    patch = payload.get("patch") or {}
    with DB_LOCK:
        row = DB.execute("SELECT data FROM diary WHERE user_id = ? AND day = ?", (user["id"], day)).fetchone()
        data = json.loads(row["data"]) if row else {}
        if "sleep" in patch:
            data["sleep"] = min(max(float(patch["sleep"]), 0), 14)
        if "stress" in patch:
            data["stress"] = min(max(int(patch["stress"]), 1), 5)
        if "water" in patch:
            data["water"] = min(max(int(patch["water"]), 0), 30)
        for k in ("sugar", "dairy", "new_product", "sweat"):
            if k in patch:
                data[k] = bool(patch[k])
        if "note" in patch:
            data["note"] = str(patch["note"])[:200]
        if isinstance(patch.get("routine"), dict):
            r = dict(data.get("routine", {}))
            for k in ROUTINE_KEYS:
                if k in patch["routine"]:
                    r[k] = bool(patch["routine"][k])
            data["routine"] = r
        DB.execute("INSERT OR REPLACE INTO diary (user_id, day, data) VALUES (?, ?, ?)",
                   (user["id"], day, json.dumps(data)))
        DB.commit()
    return {"ok": True}


def api_breakout(user, payload):
    zone = payload.get("zone")
    if zone not in ZONES:
        raise ValueError("Unknown zone.")
    sev = min(max(int(payload.get("severity", 1)), 1), 3)
    ts = time.time()
    day = payload.get("day")
    if day and day != date.today().isoformat():
        ts = min(datetime.strptime(str(day), "%Y-%m-%d").replace(hour=12).timestamp(), time.time())
    with DB_LOCK:
        DB.execute("INSERT INTO breakouts (user_id, ts, zone, severity) VALUES (?, ?, ?, ?)",
                   (user["id"], ts, zone, sev))
        DB.commit()
    return {"ok": True}


def api_delete(user, payload):
    kind, rid = payload.get("kind"), int(payload.get("id", 0))
    table = {"scan": "scans", "breakout": "breakouts"}.get(kind)
    if not table:
        raise ValueError("bad kind")
    with DB_LOCK:
        DB.execute(f"DELETE FROM {table} WHERE id = ? AND user_id = ?", (rid, user["id"]))
        DB.commit()
    return {"ok": True}


def api_review(user, payload):
    det = min(max(int(payload.get("detected", 0)), 0), 300)
    fp = min(max(int(payload.get("false_pos", 0)), 0), det)
    miss = min(max(int(payload.get("missed", 0)), 0), 300)
    with DB_LOCK:
        DB.execute("INSERT INTO reviews (user_id, ts, detected, false_pos, missed) VALUES (?, ?, ?, ?, ?)",
                   (user["id"], time.time(), det, fp, miss))
        DB.commit()
    return {"ok": True}


def api_delete_account(user, payload):
    uid = user["id"]
    with DB_LOCK:
        row = DB.execute("SELECT salt, pw_hash FROM users WHERE id = ?", (uid,)).fetchone()
    if not row or not hmac.compare_digest(_hash(payload.get("password") or "", bytes(row["salt"])), bytes(row["pw_hash"])):
        raise ValueError("Incorrect password.")
    with DB_LOCK:
        for t in ("scans", "diary", "breakouts", "reviews", "sessions", "experiments"):
            DB.execute(f"DELETE FROM {t} WHERE user_id = ?", (uid,))
        DB.execute("DELETE FROM users WHERE id = ?", (uid,))
        DB.commit()
    return {"ok": True}


def export_csv(uid, kind):
    buf = io.StringIO()
    w = csv.writer(buf)
    if kind == "scans":
        cols = ["score", "grade", "spots_total", "pustules_total", "marks_total"] + [
            f"{z}_{f}" for z in ZONES for f in FIELDS]
        w.writerow(["time", "quality_ok"] + cols)
        for r in user_scans(uid):
            w.writerow([datetime.fromtimestamp(r["ts"]).isoformat(timespec="seconds"), r["ok"]]
                       + [r.get(c) for c in cols])
    elif kind == "breakouts":
        w.writerow(["time", "zone", "severity"])
        for b in user_breakouts(uid):
            w.writerow([datetime.fromtimestamp(b["ts"]).isoformat(timespec="seconds"), b["zone"], b["severity"]])
    else:
        w.writerow(["day", "sleep", "stress", "water", "sugar", "dairy", "new_product", "sweat", "note"])
        for c in user_diary(uid):
            w.writerow([c["day"], c.get("sleep"), c.get("stress"), c.get("water"), c.get("sugar"), c.get("dairy"),
                        c.get("new_product"), c.get("sweat"), c.get("note", "")])
    return buf.getvalue().encode("utf-8")


# ----------------------------------------------------------------------------
# HTTP server (127.0.0.1 only; cookie sessions; custom header required for writes)
# ----------------------------------------------------------------------------
COOKIE = "sk_session"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost")

    def _token(self):
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        try:
            c = SimpleCookie(raw)
        except Exception:
            return None
        return c[COOKIE].value if COOKIE in c else None

    def _send(self, code, body, ctype="application/json", cookie=None, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200, cookie=None):
        self._send(code, json.dumps(clean(obj)).encode("utf-8"), cookie=cookie)

    def _set_cookie(self, token, max_age=SESSION_DAYS * 86400):
        return f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={max_age}"

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, b"forbidden", "text/plain")
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            return self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        user = user_for_token(self._token())
        if url.path == "/api/me":
            if not user:
                return self._json({"user": None})
            return self._json({"user": public_user(user), "questions": QUESTIONS, "freq": FREQ,
                               "detector": "model" if get_model() else "rules",
                               "labels": LABELS, "zones": ZONES})
        if not user:
            return self._json({"error": "Please sign in."}, 401)
        try:
            if url.path == "/api/home":
                return self._json(build_home(user))
            if url.path == "/api/history":
                return self._json(build_history(user["id"]))
            if url.path == "/api/diary":
                return self._json(build_diary(user["id"]))
            if url.path == "/api/insights":
                return self._json(build_insights(user["id"]))
            if url.path == "/api/lab":
                return self._json(build_lab(user))
            if url.path == "/api/summary":
                return self._json(build_summary(user))
            m = re.match(r"^/api/scan/(\d+)$", url.path)
            if m:
                return self._json(build_result(user["id"], int(m.group(1))))
            if url.path == "/api/export.csv":
                kind = query.get("kind", ["scans"])[0]
                return self._send(200, export_csv(user["id"], kind), "text/csv",
                                  extra={"Content-Disposition": f'attachment; filename="skinscope_{kind}.csv"'})
        except KeyError:
            return self._json({"error": "Not found."}, 404)
        except Exception as e:
            print("Server error:", repr(e))
            return self._json({"error": "Server error."}, 500)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._host_ok() or self.headers.get("X-SK") != "1":
            return self._json({"error": "forbidden"}, 403)
        length = int(self.headers.get("Content-Length") or 0)
        if length > 14_000_000:
            return self._json({"error": "Upload too large."}, 413)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            return self._json({"error": "Bad request."}, 400)
        path = urlparse(self.path).path
        try:
            if path == "/api/signup":
                uid = api_signup(payload)
                return self._json({"user": public_user(get_user(uid))}, cookie=self._set_cookie(start_session(uid)))
            if path == "/api/login":
                uid = authenticate(payload.get("email"), payload.get("password"))
                return self._json({"user": public_user(get_user(uid))}, cookie=self._set_cookie(start_session(uid)))
            if path == "/api/logout":
                end_session(self._token())
                return self._json({"ok": True}, cookie=self._set_cookie("", 0))
            user = user_for_token(self._token())
            if not user:
                return self._json({"error": "Please sign in."}, 401)
            routes = {"/api/check": api_check, "/api/scan": api_scan, "/api/profile": api_profile,
                      "/api/diary": api_diary, "/api/breakout": api_breakout, "/api/delete": api_delete,
                      "/api/review": api_review, "/api/account/delete": api_delete_account,
                      "/api/lab/start": api_lab_start, "/api/lab/checkin": api_lab_checkin, "/api/lab/stop": api_lab_stop}
            fn = routes.get(path)
            if fn is None:
                return self._json({"error": "Not found."}, 404)
            out = fn(user, payload)
            if path == "/api/account/delete":
                return self._json(out, cookie=self._set_cookie("", 0))
            return self._json(out)
        except Throttled as e:
            return self._json({"error": str(e)}, 429)
        except (ValueError, KeyError, TypeError) as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            print("Server error:", repr(e))
            return self._json({"error": "Server error."}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reset-password", metavar="EMAIL")
    parser.add_argument("--detector", choices=["auto", "rules", "model"], default="auto",
                        help="auto uses the trained model if one is installed, else the rule-based detector")
    args = parser.parse_args()
    global DET_MODE
    DET_MODE = args.detector
    init_db()
    if args.reset_password:
        return reset_password_cli(args.reset_password)
    server, port = None, args.port
    for p in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        sys.exit("Could not open a local port. Close other copies of the app and try again.")
    url = f"http://127.0.0.1:{port}"
    print(f"\nSkinScope is running at {url}")
    print(f"Your data lives in: {DATA_DIR}")
    try:
        print("Spot detector: " + ("trained model (" + MODEL_PATH.name + ")" if get_model()
                                   else "rule-based (no trained model installed)"))
    except RuntimeError as e:
        sys.exit(str(e))
    print("Press Ctrl+C here to stop.\n")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


# ----------------------------------------------------------------------------
# The web app
# ----------------------------------------------------------------------------
HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkinScope - skin analysis</title>
<style>
:root{color-scheme:dark;
 --bg:#07060f;--surface:rgba(22,19,44,.62);--surface-solid:#14112a;--surface2:rgba(255,255,255,.035);--topbar:rgba(9,8,20,.72);
 --ink:#f4f1ff;--muted:#a19dbf;--line:rgba(255,255,255,.08);--line2:rgba(255,255,255,.16);
 --brand:#8b6cff;--brand2:#ff5fa8;--brand3:#22d3ee;--brand-ink:#c7b8ff;
 --good:#34d399;--warn:#fbbf24;--bad:#f87171;--soft:rgba(139,108,255,.16);--track:rgba(255,255,255,.08);--track2:rgba(255,255,255,.14);
 --input:rgba(255,255,255,.04);--hi:rgba(255,255,255,.07);
 --shadow:0 18px 50px -18px rgba(0,0,0,.75);--glow:0 10px 34px -8px rgba(139,108,255,.65);
 --aur1:rgba(124,77,255,.45);--aur2:rgba(255,79,163,.32);--aur3:rgba(34,211,238,.22);--face:#1b1736;--face2:#0f0c22}
:root[data-theme=light]{color-scheme:light;
 --bg:#f4f2fb;--surface:rgba(255,255,255,.74);--surface-solid:#fff;--surface2:rgba(31,27,46,.03);--topbar:rgba(255,255,255,.75);
 --ink:#17132b;--muted:#5f5a7c;--line:rgba(31,27,46,.09);--line2:rgba(31,27,46,.17);
 --brand:#6d4aff;--brand2:#e8478f;--brand3:#0891b2;--brand-ink:#5534f0;
 --good:#059669;--warn:#d97706;--bad:#dc2626;--soft:rgba(109,74,255,.10);--track:rgba(31,27,46,.08);--track2:rgba(31,27,46,.14);
 --input:#fff;--hi:rgba(255,255,255,.9);
 --shadow:0 14px 40px -20px rgba(45,30,110,.35);--glow:0 10px 30px -10px rgba(109,74,255,.55);
 --aur1:rgba(124,77,255,.20);--aur2:rgba(255,79,163,.16);--aur3:rgba(34,211,238,.14);--face:#f1edff;--face2:#e4dcff}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,Roboto,sans-serif;-webkit-font-smoothing:antialiased;min-height:100vh;overflow-x:hidden}
::selection{background:var(--brand);color:#fff}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--track2);border-radius:10px;border:2px solid var(--bg)}::-webkit-scrollbar-track{background:transparent}
a{color:var(--brand-ink);text-decoration:none;transition:color .2s}a:hover{color:var(--brand2)}
h1,h2,h3{margin:0 0 8px;line-height:1.2;letter-spacing:-.02em}
h2{font-size:22px;font-weight:750}h3{font-size:16px;font-weight:700}
main>h2:first-child,#view>h2:first-child{font-size:30px;font-weight:800;letter-spacing:-.03em;background:linear-gradient(100deg,var(--ink) 30%,var(--brand-ink) 70%,var(--brand2));-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:6px}
ul{padding-left:20px}li{margin:3px 0}li::marker{color:var(--brand)}

/* ---------- ambient background ---------- */
.aurora{position:fixed;inset:0;z-index:-1;overflow:hidden;pointer-events:none;background:var(--bg)}
.aurora i{position:absolute;border-radius:50%;filter:blur(90px);opacity:1;animation:drift 26s ease-in-out infinite alternate}
.aurora i:nth-child(1){width:60vmax;height:60vmax;left:-18vmax;top:-22vmax;background:var(--aur1)}
.aurora i:nth-child(2){width:50vmax;height:50vmax;right:-16vmax;top:8vmax;background:var(--aur2);animation-duration:32s;animation-delay:-8s}
.aurora i:nth-child(3){width:46vmax;height:46vmax;left:22vmax;bottom:-26vmax;background:var(--aur3);animation-duration:38s;animation-delay:-16s}
.aurora::after{content:"";position:absolute;inset:0;background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);background-size:56px 56px;-webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 0%,transparent 70%);mask-image:radial-gradient(ellipse at 50% 0%,#000 0%,transparent 70%);opacity:.55}
@keyframes drift{0%{transform:translate(0,0) scale(1)}50%{transform:translate(6vmax,4vmax) scale(1.12)}100%{transform:translate(-4vmax,7vmax) scale(.95)}}

/* ---------- top bar ---------- */
.top{position:sticky;top:0;z-index:20;background:var(--topbar);-webkit-backdrop-filter:blur(18px) saturate(160%);backdrop-filter:blur(18px) saturate(160%);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:16px;padding:11px 24px;flex-wrap:wrap}
.logo{display:inline-flex;align-items:center;gap:9px;font-weight:850;font-size:19px;letter-spacing:-.03em;cursor:pointer;user-select:none}
.logo svg{width:30px;height:30px;flex:none;filter:drop-shadow(0 4px 12px rgba(139,108,255,.55));transition:transform .5s cubic-bezier(.2,.9,.3,1.4)}
.logo:hover svg{transform:rotate(-12deg) scale(1.08)}
.logo span,.logo.txt{background:linear-gradient(95deg,var(--ink),var(--brand-ink) 55%,var(--brand2));-webkit-background-clip:text;background-clip:text;color:transparent}
.nav{display:flex;gap:4px;flex:1;flex-wrap:wrap}
.nav a{display:inline-flex;align-items:center;gap:7px;padding:8px 14px;border-radius:99px;color:var(--muted);font-weight:650;font-size:14px;border:1px solid transparent;transition:all .25s}
.nav a svg{width:16px;height:16px;opacity:.8}
.nav a:hover{color:var(--ink);background:var(--surface2);border-color:var(--line)}
.nav a.on{color:#fff;background:linear-gradient(120deg,var(--brand),var(--brand2));box-shadow:var(--glow)}
.nav a.on svg{opacity:1}
@media(max-width:1000px){.nav a span{display:none}.nav a{padding:9px 11px}}
.me{display:flex;align-items:center;gap:10px;font-size:14px;color:var(--muted)}
.me>a{display:inline-flex;align-items:center;gap:8px;color:var(--ink);font-weight:600}
.avatar{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:13px;color:#fff;background:conic-gradient(from 200deg,var(--brand),var(--brand2),var(--brand3),var(--brand));box-shadow:0 0 0 2px var(--bg),0 0 0 3px var(--line2)}
main{max-width:1180px;margin:0 auto;padding:28px 24px 60px}
@media(max-width:600px){main{padding:18px 14px 50px}.top{padding:10px 14px}}
@media(max-width:700px){.top{gap:10px}.nav{order:3;flex:1 1 100%;flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none;margin:0 -14px;padding:2px 14px}.nav::-webkit-scrollbar{display:none}.nav a{flex:none}.nav a span{display:inline}.me{margin-left:auto}.pills{position:static}}

/* ---------- surfaces ---------- */
.card{position:relative;background:var(--surface);-webkit-backdrop-filter:blur(20px) saturate(140%);backdrop-filter:blur(20px) saturate(140%);border:1px solid var(--line);border-radius:22px;padding:22px;margin-bottom:20px;box-shadow:inset 0 1px 0 var(--hi),var(--shadow);transition:border-color .3s,transform .3s}
.card:hover{border-color:var(--line2)}
.card::before{content:"";position:absolute;inset:0 22px auto;height:1px;background:linear-gradient(90deg,transparent,var(--brand),var(--brand2),transparent);opacity:.45;border-radius:1px}
#view>*{animation:rise .6s cubic-bezier(.2,.8,.2,1) both}
#view>*:nth-child(2){animation-delay:.05s}#view>*:nth-child(3){animation-delay:.1s}#view>*:nth-child(4){animation-delay:.15s}#view>*:nth-child(5){animation-delay:.2s}#view>*:nth-child(6){animation-delay:.25s}#view>*:nth-child(n+7){animation-delay:.3s}
@keyframes rise{from{opacity:0;transform:translateY(14px) scale(.99)}to{opacity:1;transform:none}}
.grid{display:grid;gap:20px}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
.g21{grid-template-columns:minmax(0,1.5fr) minmax(0,1fr)}
.grid>.card{margin-bottom:0}
@media(max-width:900px){.g2,.g3,.g4,.g21{grid-template-columns:1fr}}
.muted{color:var(--muted)}.small{font-size:13.5px}.tiny{font-size:12px}

/* ---------- controls ---------- */
.btn{position:relative;overflow:hidden;display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line2);background:var(--surface2);color:var(--ink);padding:9px 17px;border-radius:13px;font:inherit;font-weight:650;cursor:pointer;transition:transform .2s,box-shadow .25s,border-color .25s,background .25s;text-decoration:none}
.btn:hover{border-color:var(--brand);color:var(--ink);transform:translateY(-1px);box-shadow:0 8px 22px -12px var(--brand)}
.btn:active{transform:translateY(0) scale(.98)}
.btn.p{background:linear-gradient(120deg,var(--brand),#a46bff 45%,var(--brand2));background-size:180% 100%;border-color:transparent;color:#fff;box-shadow:var(--glow);text-shadow:0 1px 1px rgba(0,0,0,.15)}
.btn.p:hover{background-position:100% 0;color:#fff;box-shadow:0 14px 40px -8px rgba(255,95,168,.6)}
.btn::after{content:"";position:absolute;top:0;left:-60%;width:40%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.35),transparent);transform:skewX(-20deg);transition:left .6s}
.btn.p:hover::after{left:130%}
.btn.big{padding:14px 28px;font-size:16px;border-radius:15px}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none;filter:grayscale(.5)}
.btn.danger{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 35%,transparent)}
.btn.danger:hover{background:color-mix(in srgb,var(--bad) 12%,transparent);border-color:var(--bad);box-shadow:none}
.btn.sm{padding:6px 12px;font-size:13px;border-radius:10px}
.btn.icon{padding:7px;width:36px;height:36px;justify-content:center;border-radius:11px}
.btn.icon svg{width:18px;height:18px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 11px;border-radius:99px;font-size:12px;font-weight:700;background:var(--soft);color:var(--brand-ink);border:1px solid color-mix(in srgb,var(--brand) 25%,transparent);white-space:nowrap}
.chip.good{background:color-mix(in srgb,var(--good) 14%,transparent);color:var(--good);border-color:color-mix(in srgb,var(--good) 30%,transparent)}
.chip.warn{background:color-mix(in srgb,var(--warn) 14%,transparent);color:var(--warn);border-color:color-mix(in srgb,var(--warn) 30%,transparent)}
.chip.bad{background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad);border-color:color-mix(in srgb,var(--bad) 30%,transparent)}
.chip.grey{background:var(--surface2);color:var(--muted);border-color:var(--line)}
input[type=text],input[type=email],input[type=password],input[type=number],input[type=date],select,textarea{width:100%;padding:11px 13px;border:1px solid var(--line2);border-radius:12px;font:inherit;background:var(--input);color:var(--ink);transition:border-color .2s,box-shadow .2s}
input:disabled{opacity:.6}
select option{background:var(--surface-solid);color:var(--ink)}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--brand);box-shadow:0 0 0 4px color-mix(in srgb,var(--brand) 22%,transparent)}
input[type=checkbox],input[type=radio]{accent-color:var(--brand);width:17px;height:17px}
label.f{display:block;font-weight:700;font-size:12px;letter-spacing:.06em;text-transform:uppercase;margin:16px 0 6px;color:var(--muted)}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;display:grid;grid-template-columns:minmax(0,1fr) 230px;gap:24px;align-items:center;color:#fff;border-radius:28px;padding:34px 36px;background:radial-gradient(120% 140% at 0% 0%,#7c4dff 0%,transparent 55%),radial-gradient(90% 120% at 100% 100%,#ff4fa3 0%,transparent 55%),radial-gradient(70% 90% at 80% 0%,#22d3ee 0%,transparent 55%),linear-gradient(135deg,#2b1a74,#5a1a5a);background-size:160% 160%;animation:mesh 18s ease-in-out infinite alternate;box-shadow:0 30px 70px -30px rgba(124,77,255,.8),inset 0 1px 0 rgba(255,255,255,.25)}
.hero::after{content:"";position:absolute;inset:0;background-image:radial-gradient(rgba(255,255,255,.18) 1px,transparent 1px);background-size:22px 22px;-webkit-mask-image:linear-gradient(90deg,transparent,#000);mask-image:linear-gradient(90deg,transparent,#000);pointer-events:none}
.hero>*{position:relative;z-index:1}
.hero h2{font-size:32px;font-weight:850;letter-spacing:-.03em;color:#fff}
.hero .btn.p{background:#fff;color:#4b2bd6;text-shadow:none;box-shadow:0 12px 30px -10px rgba(0,0,0,.5)}
.hero .btn.p:hover{color:#4b2bd6}
.hero-art svg{width:100%;height:auto;filter:drop-shadow(0 16px 30px rgba(0,0,0,.35))}
@media(max-width:760px){.hero{grid-template-columns:1fr;padding:26px}.hero-art{display:none}.hero h2{font-size:26px}}
@keyframes mesh{0%{background-position:0% 0%}100%{background-position:100% 100%}}

/* ---------- scan illustration ---------- */
.scanart .sline{animation:sweep 3.2s cubic-bezier(.45,0,.55,1) infinite}
.scanart .lm{animation:twinkle 2.4s ease-in-out infinite}
.scanart .lm:nth-of-type(3n){animation-delay:.6s}.scanart .lm:nth-of-type(3n+1){animation-delay:1.2s}
.scanart .hit{transform-box:fill-box;transform-origin:center;animation:ping 2.4s ease-out infinite}
.scanart .hit.b{animation-delay:.8s}.scanart .hit.c{animation-delay:1.6s}
@keyframes sweep{0%,100%{transform:translateY(0)}50%{transform:translateY(176px)}}
@keyframes twinkle{0%,100%{opacity:.25}50%{opacity:1}}
@keyframes ping{0%{transform:scale(.6);opacity:1}80%,100%{transform:scale(2.4);opacity:0}}

/* ---------- auth ---------- */
.auth{position:relative;min-height:100vh;display:grid;grid-template-columns:1.15fr 1fr}
@media(max-width:900px){.auth{grid-template-columns:1fr}}
.auth .l{padding:64px 56px;display:flex;flex-direction:column;justify-content:center;animation:rise .8s both}
.auth .l h1{font-size:clamp(36px,5vw,58px);font-weight:880;letter-spacing:-.03em;line-height:1.02;margin-bottom:18px}
.auth .l h1 em{font-style:normal;background:linear-gradient(100deg,var(--brand-ink),var(--brand2) 60%,var(--brand3));-webkit-background-clip:text;background-clip:text;color:transparent}
.auth .r{display:flex;align-items:center;justify-content:center;padding:28px;animation:rise .8s .15s both}
.auth .box{width:100%;max-width:430px;padding:30px}
.auth .art{width:150px;margin-bottom:22px}
.auth .themebtn{position:absolute;top:18px;right:18px;z-index:5}
@media(max-width:600px){.auth .l{padding:40px 20px 10px}}
.tabs{display:flex;gap:4px;background:var(--surface2);border:1px solid var(--line);padding:5px;border-radius:14px;margin-bottom:16px}
.tabs button{flex:1;border:0;background:none;padding:10px;border-radius:10px;font:inherit;font-weight:750;color:var(--muted);cursor:pointer;transition:all .25s}
.tabs button.on{background:linear-gradient(120deg,var(--brand),var(--brand2));color:#fff;box-shadow:var(--glow)}
.feat{display:flex;gap:14px;align-items:flex-start;margin:12px 0;padding:12px 14px;border-radius:16px;background:var(--surface2);border:1px solid var(--line);max-width:540px;transition:transform .3s,border-color .3s}
.feat:hover{transform:translateX(4px);border-color:var(--line2)}
.feat b{display:block}
.feat .chip{width:30px;height:30px;padding:0;justify-content:center;border-radius:10px;font-size:14px;background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;border:0;flex:none}
.err{color:var(--bad);font-size:13px;min-height:20px;margin-top:8px}

/* ---------- stats ---------- */
.stat{text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;min-height:190px}
.stat .n{font-size:40px;font-weight:850;letter-spacing:-.04em;line-height:1.1;background:linear-gradient(135deg,var(--ink),var(--brand-ink));-webkit-background-clip:text;background-clip:text;color:transparent}
.stat .ic{width:46px;height:46px;border-radius:15px;display:grid;place-items:center;margin-bottom:6px;background:linear-gradient(135deg,color-mix(in srgb,var(--brand) 30%,transparent),color-mix(in srgb,var(--brand2) 22%,transparent));border:1px solid var(--line2)}
.stat .ic svg{width:22px;height:22px;color:var(--ink)}

/* ---------- scan page ---------- */
.slots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
@media(max-width:700px){.slots{grid-template-columns:1fr}}
.slot{border:1.5px dashed var(--line2);border-radius:20px;background:radial-gradient(circle at 50% 30%,var(--soft),transparent 70%),var(--surface2);min-height:240px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;text-align:center;padding:14px;cursor:pointer;position:relative;overflow:hidden;transition:all .3s}
.slot:hover{border-color:var(--brand);box-shadow:var(--glow);transform:translateY(-2px)}
.slot .cam{width:58px;height:58px;border-radius:18px;display:grid;place-items:center;margin-bottom:8px;background:linear-gradient(135deg,var(--brand),var(--brand2));box-shadow:var(--glow);transition:transform .4s cubic-bezier(.2,.9,.3,1.4)}
.slot .cam svg{width:28px;height:28px;color:#fff}
.slot:hover .cam{transform:scale(1.08) rotate(-6deg)}
.slot.has{border-style:solid;border-color:var(--line2);background:#000;padding:0}
.slot img{width:100%;height:240px;object-fit:cover;display:block}
.slot.checking::after{content:"";position:absolute;left:0;right:0;top:0;height:40%;background:linear-gradient(180deg,transparent,color-mix(in srgb,var(--brand3) 45%,transparent) 90%,var(--brand3));border-bottom:2px solid var(--brand3);animation:slotscan 1.6s ease-in-out infinite;pointer-events:none}
@keyframes slotscan{0%{transform:translateY(-100%)}100%{transform:translateY(250%)}}
.slot .cap{position:absolute;left:10px;right:10px;bottom:10px;background:rgba(10,8,24,.65);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);color:#fff;font-size:12.5px;padding:8px 11px;text-align:left;border-radius:12px;border:1px solid rgba(255,255,255,.15)}
.slot .x{position:absolute;top:10px;right:10px;background:rgba(10,8,24,.65);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);color:#fff;border:1px solid rgba(255,255,255,.2);border-radius:50%;width:30px;height:30px;cursor:pointer;font-size:16px;z-index:2;transition:transform .2s}
.slot .x:hover{transform:rotate(90deg);background:var(--bad)}
.qgrp{margin-top:18px}.qgrp h4{margin:0 0 6px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--brand-ink)}
.qrow{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;padding:10px 0;border-bottom:1px solid var(--line)}
.qrow .ql{flex:1 1 260px;font-weight:550}
.opt{display:inline-flex;align-items:center;gap:5px;padding:6px 13px;border:1px solid var(--line2);border-radius:99px;font-size:13px;cursor:pointer;background:var(--surface2);transition:all .2s;user-select:none}
.opt:hover{border-color:var(--brand)}
.opt:has(input:checked){background:linear-gradient(120deg,var(--brand),var(--brand2));border-color:transparent;color:#fff;font-weight:650;box-shadow:0 6px 18px -8px var(--brand)}
.opt input{display:none}
.qrow input[type=number]{width:100px}

/* ---------- data viz ---------- */
.ring{position:relative;display:inline-block}
.ring .arc{animation:arc 1.4s cubic-bezier(.2,.8,.2,1) both}
@keyframes arc{from{stroke-dashoffset:var(--c)}to{stroke-dashoffset:var(--off)}}
.bars{display:grid;gap:12px}.bar{display:grid;grid-template-columns:130px 1fr 40px;gap:12px;align-items:center;font-size:14px;font-weight:550}
.bar b{text-align:right;font-variant-numeric:tabular-nums}
.bar .t{height:10px;background:var(--track);border-radius:99px;overflow:hidden}
.bar .t i{display:block;height:100%;border-radius:99px;transform-origin:left;animation:grow 1.2s cubic-bezier(.2,.8,.2,1) both}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px;position:sticky;top:66px;z-index:10;padding:10px 0}
.pills a{padding:8px 15px;background:var(--topbar);-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);border:1px solid var(--line2);border-radius:99px;font-weight:650;font-size:13px;color:var(--ink);transition:all .2s}
.pills a:hover{border-color:var(--brand);color:var(--ink);box-shadow:var(--glow)}
.gauge{display:flex;gap:6px;margin:12px 0}.gauge i{flex:1;height:12px;border-radius:99px;background:var(--track);transition:background .4s}
.nut{position:relative;border:1px solid var(--line);border-radius:18px;padding:16px;background:var(--surface2);transition:transform .3s,border-color .3s,box-shadow .3s;overflow:hidden}
.nut::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,var(--brand),var(--brand2))}
.nut:hover{transform:translateY(-3px);border-color:var(--line2);box-shadow:var(--shadow)}
.nut h4{margin:0;font-size:15.5px}.nut ul{margin:6px 0 0;padding-left:18px}
.step{display:flex;gap:14px;padding:12px 0;border-bottom:1px solid var(--line)}
.step:last-child{border-bottom:0}
.step .n{width:30px;height:30px;border-radius:10px;background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;font-weight:800;font-size:14px;display:flex;align-items:center;justify-content:center;flex:none;box-shadow:0 6px 16px -6px var(--brand)}
.note{position:relative;border:1px solid color-mix(in srgb,var(--warn) 30%,transparent);background:color-mix(in srgb,var(--warn) 9%,transparent);padding:12px 16px 12px 18px;border-radius:14px;font-size:14px;overflow:hidden}
.note::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--warn);box-shadow:0 0 14px var(--warn)}
.note.info{border-color:color-mix(in srgb,var(--brand) 30%,transparent);background:color-mix(in srgb,var(--brand) 10%,transparent)}
.note.info::before{background:linear-gradient(180deg,var(--brand),var(--brand2));box-shadow:0 0 14px var(--brand)}
.note.bad{border-color:color-mix(in srgb,var(--bad) 30%,transparent);background:color-mix(in srgb,var(--bad) 10%,transparent)}
.note.bad::before{background:var(--bad);box-shadow:0 0 14px var(--bad)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;box-shadow:0 0 8px currentColor;vertical-align:-1px}
.facemap{width:100%;max-width:240px;display:block;margin:0 auto;overflow:visible}
.faceo{fill:url(#faceFill);stroke:url(#faceStroke);stroke-width:2}
.zone rect{transition:fill-opacity .25s,transform .25s;transform-box:fill-box;transform-origin:center}
.zone text{font-size:8px;font-weight:700;fill:var(--ink);pointer-events:none}
.clickable .zone{cursor:pointer}
.clickable .zone:hover rect{fill-opacity:.85;transform:scale(1.04)}
.lesion{animation:pop .5s cubic-bezier(.2,.9,.3,1.5) both;transform-box:fill-box;transform-origin:center}
@keyframes pop{from{transform:scale(0);opacity:0}to{transform:scale(1);opacity:1}}
table{width:100%;border-collapse:separate;border-spacing:0}
th,td{padding:10px 10px;text-align:left;border-bottom:1px solid var(--line);font-size:14px}
th{color:var(--muted);font-weight:700;font-size:12px;letter-spacing:.05em;text-transform:uppercase}
tbody tr{transition:background .2s}tbody tr:hover{background:var(--surface2)}
td{font-variant-numeric:tabular-nums}
.angle{text-align:center;padding:14px;border-radius:18px;background:var(--surface2);border:1px solid var(--line)}
.angle svg{width:100%;max-width:150px;display:block;margin:8px auto}
.checkrow{display:flex;gap:10px;align-items:flex-start;padding:7px 0}
label.checkrow{cursor:pointer;border-radius:10px;padding:8px 10px;margin:0 -10px;transition:background .2s}
label.checkrow:hover{background:var(--surface2)}
.chartwrap{position:relative}
canvas.chart{width:100%;height:240px;display:block;cursor:crosshair}
#ovcanvas{width:100%;border-radius:16px;background:#000;display:block;box-shadow:var(--shadow)}

/* ---------- overlays ---------- */
#toast{position:fixed;bottom:26px;left:50%;transform:translateX(-50%);background:var(--surface-solid);color:var(--ink);border:1px solid var(--line2);box-shadow:var(--shadow),var(--glow);padding:12px 22px;border-radius:99px;display:none;z-index:99;max-width:90vw;text-align:center;font-weight:600;animation:toastin .35s cubic-bezier(.2,.9,.3,1.3)}
@keyframes toastin{from{opacity:0;transform:translate(-50%,16px)}to{opacity:1;transform:translate(-50%,0)}}
#rvmodal{position:fixed;inset:0;background:rgba(5,4,14,.75);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);z-index:60;display:none;overflow:auto;padding:20px}
#rvbox{max-width:960px;margin:20px auto;background:var(--surface-solid);border:1px solid var(--line2);border-radius:24px;padding:22px;box-shadow:var(--shadow);animation:rise .4s both}
#rvcanvas{width:100%;display:block;border-radius:14px;cursor:crosshair}
#busy{position:fixed;inset:0;z-index:80;display:none;place-items:center;background:rgba(5,4,14,.78);-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px)}
#busy.on{display:grid;animation:fadein .3s both}
#busy .in{text-align:center;color:#fff;max-width:340px;padding:20px}
#busy svg{width:190px;margin:0 auto 18px;display:block;filter:drop-shadow(0 0 30px rgba(139,108,255,.7))}
#busy b{font-size:20px;display:block;letter-spacing:-.02em}
#busy .sub{color:#c9c4e6;font-size:13.5px;margin-top:6px;min-height:22px}
#busy .prog{margin-top:16px}
#busy .prog i{width:40%;animation:indet 1.4s ease-in-out infinite}
@keyframes indet{0%{transform:translateX(-110%)}100%{transform:translateX(260%)}}
@keyframes fadein{from{opacity:0}to{opacity:1}}
.prog{height:10px;background:var(--track);border-radius:99px;overflow:hidden}
.prog i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--brand),var(--brand2),var(--brand3));background-size:200% 100%;animation:flow 3s linear infinite;box-shadow:0 0 14px var(--brand)}
@keyframes flow{to{background-position:200% 0}}
.loading{display:flex;align-items:center;gap:12px}
.spinner{width:22px;height:22px;border-radius:50%;border:3px solid var(--track2);border-top-color:var(--brand);animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.print-only{display:none}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.001s!important;animation-iteration-count:1!important;transition:none!important}}
@media print{:root{--bg:#fff;--surface:#fff;--surface2:#f6f5fa;--ink:#111;--muted:#555;--line:#ddd;--line2:#ccc;--track:#eee;--brand-ink:#5534f0;--face:#f1edff;--face2:#e4dcff;color-scheme:light}
 .aurora,.top,.pills,.btn,#toast,#busy,.noprint{display:none!important}body{background:#fff}
 .card{box-shadow:none;break-inside:avoid;-webkit-backdrop-filter:none;backdrop-filter:none;animation:none!important}
 .card::before,.nut::before{display:none}#view>*{animation:none!important}
 main>h2:first-child,#view>h2:first-child,.stat .n{color:#111;background:none}.print-only{display:block}}
.auth .art{background:linear-gradient(135deg,#2b1a74,#5a1a5a);border-radius:30px;padding:14px;box-shadow:0 20px 50px -20px rgba(124,77,255,.8)}
.me .nm{max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@media(max-width:600px){.me .nm{display:none}}
.btn svg{width:16px;height:16px;flex:none}
.slot-acts{margin-top:12px;justify-content:center}
.slot .retake{position:absolute;top:10px;left:10px;background:rgba(10,8,24,.65);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);color:#fff;border:1px solid rgba(255,255,255,.2);border-radius:50%;width:30px;height:30px;cursor:pointer;z-index:2;display:grid;place-items:center;padding:0;transition:transform .2s}
.slot .retake svg{width:16px;height:16px}.slot .retake:hover{transform:scale(1.1);background:var(--brand)}
#cam{position:fixed;inset:0;z-index:70;display:none;place-items:center;background:rgba(5,4,14,.88);-webkit-backdrop-filter:blur(16px);backdrop-filter:blur(16px);padding:16px;overflow:auto}
#cam.on{display:grid;animation:fadein .25s both}
.cam-in{width:min(100%,480px);display:flex;flex-direction:column;gap:12px;color:#fff}
.cam-top{display:flex;justify-content:space-between;align-items:center;gap:12px}
.cam-top b{font-size:18px;letter-spacing:-.02em}.cam-top .tiny{color:#c9c4e6}
#cam .btn{color:#fff;border-color:rgba(255,255,255,.25);background:rgba(255,255,255,.07)}
#cam .opt{color:#fff;border-color:rgba(255,255,255,.25);background:rgba(255,255,255,.07)}
#cam .opt:has(input:checked){background:linear-gradient(120deg,#8b6cff,#ff5fa8);border-color:transparent;box-shadow:0 6px 18px -8px #8b6cff}
#cam .opt:has(input:checked)::before{content:"\2713";font-weight:900}
.cam-stage{position:relative;width:min(100%,calc(62vh*.8));aspect-ratio:4/5;margin:0 auto;border-radius:28px;overflow:hidden;background:#000;box-shadow:0 30px 80px -20px rgba(124,77,255,.6),0 0 0 1px rgba(255,255,255,.12)}
.cam-stage video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.cam-stage video.mirror{transform:scaleX(-1)}
.cam-guide{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.cam-guide .oval{stroke:rgba(255,255,255,.75);transition:stroke .3s}
.cam-guide .oval-dash{stroke:rgba(255,255,255,.35);transform-box:fill-box;transform-origin:center;animation:spin 20s linear infinite}
.cam-guide .brk{stroke:url(#camG)}
#cam.warn .cam-guide .oval{stroke:#fbbf24}
#cam.ok .cam-guide .oval,#cam.ok .cam-guide .oval-dash{stroke:#34d399}
.cam-scan{position:absolute;left:12%;right:12%;top:8%;height:3px;border-radius:3px;background:linear-gradient(90deg,transparent,#67e8f9,transparent);box-shadow:0 0 18px #67e8f9;animation:camscan 2.6s ease-in-out infinite alternate;pointer-events:none}
#cam.ok .cam-scan{background:linear-gradient(90deg,transparent,#34d399,transparent);box-shadow:0 0 18px #34d399}
@keyframes camscan{from{top:8%}to{top:90%}}
.cam-msg{position:absolute;left:50%;top:14px;transform:translateX(-50%);max-width:88%;padding:8px 16px;border-radius:99px;background:rgba(10,8,24,.72);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,.18);font-weight:650;font-size:13.5px;text-align:center;transition:background .3s}
#cam.warn .cam-msg{background:rgba(180,120,10,.82)}
#cam.ok .cam-msg{background:rgba(5,150,105,.88)}
.cam-count{position:absolute;inset:0;display:none;place-items:center;pointer-events:none}
.cam-count.on{display:grid}
.cam-count span{font-size:130px;font-weight:900;color:#fff;text-shadow:0 0 50px #8b6cff,0 0 20px #ff5fa8;animation:countpop 1s ease-out both}
@keyframes countpop{0%{transform:scale(1.8);opacity:0}25%{transform:scale(1);opacity:1}100%{transform:scale(.85);opacity:.15}}
.cam-flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none}
.cam-flash.go{animation:camflash .55s ease-out}
@keyframes camflash{0%{opacity:.95}100%{opacity:0}}
.cam-err{position:absolute;inset:0;display:none;flex-direction:column;gap:14px;align-items:center;justify-content:center;text-align:center;padding:30px;background:rgba(10,8,24,.92);font-size:14px}
.cam-err.on{display:flex}
.cam-checks{display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
.cam-checks span{display:inline-flex;align-items:center;gap:5px;padding:4px 11px;border-radius:99px;font-size:12px;font-weight:700;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);color:#c9c4e6;transition:all .3s}
.cam-checks span::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.cam-checks span.y{color:#34d399;border-color:rgba(52,211,153,.45);background:rgba(52,211,153,.12)}
.cam-checks span.n{color:#fbbf24;border-color:rgba(251,191,36,.45);background:rgba(251,191,36,.12)}
.cam-ctl{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px}
.cam-ctl .opt{justify-self:start}.cam-ctl .btn{justify-self:end}
.shutter{width:74px;height:74px;border-radius:50%;border:0;cursor:pointer;background:radial-gradient(circle,#fff 0 50%,transparent 53%),conic-gradient(#8b6cff,#ff5fa8,#22d3ee,#8b6cff);box-shadow:0 0 0 4px rgba(255,255,255,.14),0 10px 34px -6px rgba(139,108,255,.8);transition:transform .15s}
.shutter:hover{transform:scale(1.06)}.shutter:active{transform:scale(.9)}
.cam-foot{text-align:center;color:#a19dbf}
</style></head>
<body>
<div class="aurora" aria-hidden="true"><i></i><i></i><i></i></div>
<div id="app"></div>
<div id="rvmodal"><div id="rvbox">
  <h3>Check the detections</h3>
  <p class="muted small">This picture stays in your browser and is not saved. <b>Tap a circle</b> that is not really a pimple (it turns grey). <b>Tap an empty spot</b> where a pimple was missed (tap again to remove).</p>
  <canvas id="rvcanvas"></canvas>
  <div class="row" style="margin-top:10px"><button class="btn p" data-act="rvSubmit">Submit review</button><button class="btn" data-act="rvCancel">Cancel</button><span id="rvstat" class="muted small"></span></div>
</div></div>
<input type="file" id="fileIn" accept="image/*" hidden>
<div id="toast"></div>
<div id="busy" role="status" aria-live="polite"><div class="in"><div id="busyArt"></div><b>Analysing your skin</b><div class="sub" id="busySub"></div><div class="prog"><i></i></div></div></div>
<div id="cam" role="dialog" aria-modal="true" aria-label="Camera">
 <div class="cam-in">
  <div class="cam-top"><div><b id="camTitle">Front photo</b><div class="tiny" id="camSub"></div></div><button class="btn icon" data-act="camClose" title="Close camera">&times;</button></div>
  <div class="cam-stage">
   <video id="camVideo" playsinline muted autoplay></video>
   <svg class="cam-guide" viewBox="0 0 400 500" preserveAspectRatio="none" aria-hidden="true">
    <defs><mask id="camMask"><rect width="400" height="500" fill="#fff"/><ellipse cx="200" cy="245" rx="132" ry="178" fill="#000"/></mask>
     <linearGradient id="camG" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#c4b5fd"/><stop offset=".5" stop-color="#ff8cc6"/><stop offset="1" stop-color="#67e8f9"/></linearGradient></defs>
    <rect width="400" height="500" fill="rgba(5,4,14,.55)" mask="url(#camMask)"/>
    <ellipse class="oval-dash" cx="200" cy="245" rx="146" ry="192" fill="none" stroke-width="1.5" stroke-dasharray="3 10"/>
    <ellipse class="oval" cx="200" cy="245" rx="132" ry="178" fill="none" stroke-width="3.5"/>
    <g class="brk" fill="none" stroke-width="4" stroke-linecap="round"><path d="M24 70V24h46M376 70V24h-46M24 430v46h46M376 430v46h-46"/></g>
   </svg>
   <div class="cam-scan"></div>
   <div class="cam-msg" id="camMsg">Starting camera...</div>
   <div class="cam-count" id="camCount"></div>
   <div class="cam-flash" id="camFlash"></div>
   <div class="cam-err" id="camErr"><div id="camErrMsg"></div><button class="btn p" data-act="camUpload">Upload a photo instead</button></div>
  </div>
  <div class="cam-checks" id="camChecks"></div>
  <div class="cam-ctl"><label class="opt"><input type="checkbox" id="camAuto" checked> Auto-capture</label><button class="shutter" data-act="camShoot" title="Take photo now" aria-label="Take photo now"></button><button class="btn sm" data-act="camFlip">Switch camera</button></div>
  <div class="tiny cam-foot">Live checks run on this computer. Nothing is saved until you press Analyse.</div>
 </div>
</div>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sum=a=>a.reduce((x,y)=>x+y,0), mean=a=>a.length?sum(a)/a.length:0;
let me=null, cfg=null, authMode='login', authErr='';
const mem={}; let sc={photos:{},canvases:{},checks:{},q:{}}; let pendingView=null; let rv=null;
const VIEW_INFO={front:{name:'Front',sub:'Look straight at the camera',required:true},
  right:{name:'Right cheek view',sub:'Turn your head slightly to your LEFT',required:false},
  left:{name:'Left cheek view',sub:'Turn your head slightly to your RIGHT',required:false}};
const LT={i:'#e5484d',p:'#f5b700',m:'#a3672d'};
const ZC={forehead:'#f59e0b',left_cheek:'#10b981',right_cheek:'#3b82f6',nose:'#a855f7',chin:'#ec4899'};

function toast(msg){const t=$('#toast');t.textContent=msg;t.style.display='block';clearTimeout(toast.h);toast.h=setTimeout(()=>t.style.display='none',3800);}
async function api(path,body){
  const o={method:body===undefined?'GET':'POST',headers:{'X-SK':'1'}};
  if(body!==undefined){o.headers['Content-Type']='application/json';o.body=JSON.stringify(body);}
  const r=await fetch(path,o); let j={}; try{j=await r.json();}catch(e){}
  if(!r.ok){const e=new Error(j.error||('Error '+r.status));e.status=r.status;throw e;}
  return j;
}
const fmtDate=ts=>new Date(ts*1000).toLocaleDateString(undefined,{day:'numeric',month:'short'});
const fmtDT=ts=>new Date(ts*1000).toLocaleString(undefined,{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'});
const todayStr=()=>{const d=new Date();d.setMinutes(d.getMinutes()-d.getTimezoneOffset());return d.toISOString().slice(0,10);};
const pct=v=>v==null?'n/a':Math.round(v*100)+'%';
const lvlColor=v=>v>=80?'#10b981':v>=55?'#f59e0b':'#ef4444';
const lvlGrad=v=>v>=80?['#34d399','#22d3ee']:v>=55?['#fbbf24','#fb7185']:['#f87171','#ec4899'];
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduceMotion=()=>window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- theme ---------- */
let THEME=(()=>{try{return localStorage.getItem('skinscope-theme')||'dark';}catch(e){return 'dark';}})();
function applyTheme(){document.documentElement.dataset.theme=THEME;try{localStorage.setItem('skinscope-theme',THEME);}catch(e){}}
applyTheme();

/* ---------- icons and artwork ---------- */
const ICONS={
  home:'<path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/>',
  scan:'<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3.5"/>',
  results:'<path d="M3 20h18M6 16v-5M11 16V6M16 16v-8"/>',
  history:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  diary:'<path d="M5 4h11a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3z"/><path d="M5 17a3 3 0 0 1 3-3h11"/>',
  lab:'<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3"/><path d="M7.5 15h9"/>',
  insights:'<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 16.5l.6 1.9 1.9.6-1.9.6-.6 1.9-.6-1.9-1.9-.6 1.9-.6z"/>',
  camera:'<path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13.5" r="3.5"/>',
  moon:'<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  flame:'<path d="M12 3c1 4 5 5.5 5 10a5 5 0 0 1-10 0c0-2.5 1.5-3.5 2-5 1 1.5 2 2 3 2-1-2-1-4 0-7z"/>',
  shield:'<path d="M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
  dots:'<circle cx="8" cy="9" r="2.2"/><circle cx="15.5" cy="7.5" r="1.6"/><circle cx="14" cy="15" r="2.6"/><circle cx="7.5" cy="16" r="1.4"/>',
};
const ic=n=>`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[n]||''}</svg>`;
const LOGO=`<svg viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="lgG" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#8b6cff"/><stop offset=".6" stop-color="#ff5fa8"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs><rect x="1" y="1" width="30" height="30" rx="10" fill="url(#lgG)"/><circle cx="14" cy="14" r="6.5" fill="none" stroke="#fff" stroke-width="2.4"/><path d="M19 19l5.5 5.5" stroke="#fff" stroke-width="2.6" stroke-linecap="round"/><circle cx="12" cy="12" r="1.6" fill="#fff"/></svg>`;
const themeBtn=()=>`<button class="btn icon" data-act="theme" title="Switch light / dark theme">${ic(THEME==='dark'?'sun':'moon')}</button>`;
function scanArt(){
  const lm=[[78,104],[122,104],[100,128],[86,160],[114,160],[100,168],[64,90],[136,90],[100,58],[70,138],[130,138],[100,196]];
  return `<svg class="scanart" viewBox="0 0 200 240" aria-hidden="true"><defs><linearGradient id="saG" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#c4b5fd"/><stop offset=".5" stop-color="#ff8cc6"/><stop offset="1" stop-color="#67e8f9"/></linearGradient><linearGradient id="saL" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#67e8f9" stop-opacity="0"/><stop offset="1" stop-color="#67e8f9" stop-opacity=".55"/></linearGradient><clipPath id="saC"><ellipse cx="100" cy="122" rx="62" ry="86"/></clipPath></defs>
   <g fill="none" stroke="url(#saG)" stroke-width="3" stroke-linecap="round"><path d="M18 46V22h24M182 46V22h-24M18 194v24h24M182 194v24h-24"/></g>
   <ellipse cx="100" cy="122" rx="62" ry="86" fill="rgba(255,255,255,.07)" stroke="url(#saG)" stroke-width="2.5"/>
   <g stroke="rgba(255,255,255,.2)" stroke-width="1" fill="none"><path d="M100 36v172M38 122h124M50 82q50 18 100 0M50 164q50-18 100 0M72 46q-12 76 0 152M128 46q12 76 0 152"/></g>
   ${lm.map(p=>`<circle class="lm" cx="${p[0]}" cy="${p[1]}" r="2.3" fill="#fff"/>`).join('')}
   <circle cx="76" cy="148" r="3" fill="#ff8cc6"/><circle class="hit" cx="76" cy="148" r="5" fill="none" stroke="#ff8cc6" stroke-width="2"/>
   <circle cx="130" cy="80" r="3" fill="#fbbf24"/><circle class="hit b" cx="130" cy="80" r="5" fill="none" stroke="#fbbf24" stroke-width="2"/>
   <circle cx="118" cy="178" r="3" fill="#ff8cc6"/><circle class="hit c" cx="118" cy="178" r="5" fill="none" stroke="#ff8cc6" stroke-width="2"/>
   <g clip-path="url(#saC)"><rect class="sline" x="30" y="10" width="140" height="26" fill="url(#saL)"/><rect class="sline" x="30" y="35" width="140" height="2" fill="#67e8f9"/></g></svg>`;
}
function fx(root){
  if(reduceMotion()) return;
  (root||document).querySelectorAll('[data-count]').forEach(el=>{
    const to=+el.dataset.count; if(!isFinite(to)) return; const t0=performance.now(),D=1100;
    const st=n=>{const p=Math.min(1,(n-t0)/D);el.textContent=Math.round(to*(1-Math.pow(1-p,3)));if(p<1)requestAnimationFrame(st);else el.textContent=el.dataset.count;};
    requestAnimationFrame(st);
  });
}
let BUSY_T=null;
function busy(on){
  const b=$('#busy'); if(!b) return; clearInterval(BUSY_T);
  if(!on){b.classList.remove('on');return;}
  const msgs=['Finding your face...','Mapping five skin zones...','Counting spots and dark marks...','Checking shine and redness...','Building your plan...'];
  let i=0; $('#busyArt').innerHTML=scanArt(); $('#busySub').textContent=msgs[0]; b.classList.add('on');
  BUSY_T=setInterval(()=>{i=(i+1)%msgs.length;$('#busySub').textContent=msgs[i];},1300);
}

/* ---------- small visual helpers ---------- */
let _gid=0;
function ring(score,size=130,label=''){
  const sw=Math.max(8,Math.round(size*0.085)),h=size/2,r=h-sw/2-6,c=2*Math.PI*r,v=Math.max(0,Math.min(100,score)),off=c*(1-v/100);
  const [g1,g2]=lvlGrad(score),id='rg'+(++_gid),arc=`cx="${h}" cy="${h}" r="${r}" fill="none" stroke="url(#${id})" stroke-width="${sw}" stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${off}" style="--c:${c}px;--off:${off}px" transform="rotate(-90 ${h} ${h})"`;
  return `<div class="ring"><svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" style="overflow:visible"><defs><linearGradient id="${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${g1}"/><stop offset="1" stop-color="${g2}"/></linearGradient><filter id="${id}f" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="${sw*0.7}"/></filter></defs>
   <circle cx="${h}" cy="${h}" r="${r}" fill="none" style="stroke:var(--track)" stroke-width="${sw}"/>
   <circle class="arc" ${arc} opacity=".6" filter="url(#${id}f)"/><circle class="arc" ${arc}/>
   <text x="50%" y="${size>=120?h-size*0.04:h}" text-anchor="middle" dominant-baseline="central" font-size="${size*0.3}" font-weight="850" letter-spacing="-1" style="fill:var(--ink)" data-count="${score}">${score}</text>
   ${size>=120?`<text x="50%" y="${h+size*0.17}" text-anchor="middle" dominant-baseline="central" font-size="${size*0.085}" font-weight="600" style="fill:var(--muted)">out of 100</text>`:''}</svg>${label?`<div class="tiny muted" style="text-align:center">${label}</div>`:''}</div>`;
}
function bar(label,v){const [a,b]=lvlGrad(v);return `<div class="bar"><span>${esc(label)}</span><div class="t"><i style="width:${v}%;background:linear-gradient(90deg,${a},${b});box-shadow:0 0 14px ${a}99"></i></div><b data-count="${v}">${v}</b></div>`;}
const FX=22,FY=18,FW=156,FH=224;
const FACE_DEFS=`<defs><radialGradient id="faceFill" cx="50%" cy="38%" r="70%"><stop offset="0" style="stop-color:var(--face)"/><stop offset="1" style="stop-color:var(--face2)"/></radialGradient><linearGradient id="faceStroke" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#8b6cff"/><stop offset=".5" stop-color="#ff5fa8"/><stop offset="1" stop-color="#22d3ee"/></linearGradient><filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
const FACE_BASE=`<ellipse cx="24" cy="132" rx="9" ry="20" class="faceo"/><ellipse cx="176" cy="132" rx="9" ry="20" class="faceo"/><ellipse cx="100" cy="130" rx="80" ry="114" class="faceo"/><g fill="none" style="stroke:var(--line2)" stroke-width="2" stroke-linecap="round"><path d="M58 100q14-8 28 0M114 100q14-8 28 0"/><path d="M62 116q10-7 20 0q-10 6-20 0zM118 116q10-7 20 0q-10 6-20 0"/><path d="M100 122v24q-6 6 0 8"/><path d="M80 186q20 12 40 0"/></g>`;
function faceMap(colorFn,clickable){
  let s=`<svg viewBox="0 0 200 260" class="facemap">${FACE_DEFS}${FACE_BASE}`;
  for(const [z,f] of Object.entries(cfg.zones)){
    const x=FX+FW*f[0],w=FW*(f[1]-f[0]),y=FY+FH*f[2],h=FH*(f[3]-f[2]),c=colorFn(z);
    s+=`<g class="zone" ${clickable?`data-act="pickZone" data-z="${z}"`:''}><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" style="fill:${c};stroke:${c}" fill-opacity="0.28" stroke-width="1.8" filter="url(#softGlow)"/><text x="${x+w/2}" y="${y+h/2+3}" text-anchor="middle">${esc(cfg.labels[z])}</text></g>`;
  }
  return s+'</svg>';
}
function lesionMap(les){
  let s=`<svg viewBox="0 0 200 260" class="facemap">${FACE_DEFS}${FACE_BASE}`;
  for(const f of Object.values(cfg.zones)) s+=`<rect x="${FX+FW*f[0]}" y="${FY+FH*f[2]}" width="${FW*(f[1]-f[0])}" height="${FH*(f[3]-f[2])}" rx="10" fill="none" style="stroke:var(--line2)" stroke-dasharray="3 4"/>`;
  les.forEach((l,i)=>{const c=LT[l[3]]||'#000';s+=`<circle class="lesion" style="animation-delay:${Math.min(i*30,900)}ms" cx="${FX+FW*l[0]}" cy="${FY+FH*l[1]}" r="${Math.max(2.6,FW*l[2])}" fill="${c}" fill-opacity="0.85" stroke="${c}" filter="url(#softGlow)"/>`;});
  return s+'</svg><div class="legend" style="justify-content:center;margin-top:8px"><span><span class="dot" style="background:'+LT.i+';color:'+LT.i+'"></span>inflamed</span><span><span class="dot" style="background:'+LT.p+';color:'+LT.p+'"></span>pustule-like</span><span><span class="dot" style="background:'+LT.m+';color:'+LT.m+'"></span>dark mark</span></div>';
}
function angleSvg(deg,label){
  const rad=deg*Math.PI/180,S=Math.sin(rad),C=Math.cos(rad),nx=60+S*46,ny=60-C*46;
  return `<svg viewBox="0 0 120 150" aria-hidden="true"><defs><linearGradient id="agr" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#8b6cff"/><stop offset="1" stop-color="#ff5fa8"/></linearGradient></defs>
   <circle cx="60" cy="60" r="50" fill="none" style="stroke:var(--line2)" stroke-dasharray="2 5"/>
   <line x1="60" y1="60" x2="60" y2="118" style="stroke:var(--line2)" stroke-dasharray="3 4"/>
   <circle cx="60" cy="60" r="34" style="fill:var(--soft)" stroke="url(#agr)" stroke-width="2.5"/>
   <circle cx="${60+S*14-C*11}" cy="${56-C*14-S*11}" r="2.4" style="fill:var(--ink)"/><circle cx="${60+S*14+C*11}" cy="${56-C*14+S*11}" r="2.4" style="fill:var(--ink)"/>
   <polygon points="${nx},${ny} ${60+S*30+C*7},${60-C*30+S*7} ${60+S*30-C*7},${60-C*30-S*7}" fill="url(#agr)"/>
   <rect x="40" y="120" width="40" height="22" rx="7" style="fill:var(--ink)"/><circle cx="60" cy="131" r="6.5" style="fill:var(--bg)"/><circle cx="60" cy="131" r="3" fill="url(#agr)"/>
   <text x="60" y="113" text-anchor="middle" font-size="9" font-weight="600" style="fill:var(--muted)">camera</text></svg>`;
}

/* ---------- trend chart: animated draw-in, smooth line, crosshair + tooltip ---------- */
function hexA(hex,a){const h=hex.replace('#','');if(h.length!==6)return hex;return `rgba(${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)},${a})`;}
function drawChart(canvas,series,marks){
  if(!canvas) return; canvas._cfg={series,marks:marks||[]}; canvas._mx=null;
  if(!canvas._hov){canvas._hov=1;
    canvas.addEventListener('mousemove',e=>{const r=canvas.getBoundingClientRect();canvas._mx=e.clientX-r.left;paintChart(canvas,1);});
    canvas.addEventListener('mouseleave',()=>{canvas._mx=null;paintChart(canvas,1);});}
  if(reduceMotion()){paintChart(canvas,1);return;}
  const t0=performance.now(),D=1000,tok=canvas._tok=(canvas._tok||0)+1;
  const step=n=>{if(canvas._tok!==tok)return;const p=Math.min(1,(n-t0)/D);paintChart(canvas,1-Math.pow(1-p,3));if(p<1)requestAnimationFrame(step);};
  requestAnimationFrame(step);
}
function paintChart(canvas,prog){
  const {series,marks}=canvas._cfg||{series:[],marks:[]};
  const dpr=window.devicePixelRatio||1,W=canvas.clientWidth,H=canvas.clientHeight; if(!W) return;
  canvas.width=W*dpr;canvas.height=H*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,W,H);
  const ink=cssv('--ink'),muted=cssv('--muted'),line=cssv('--line'),line2=cssv('--line2'),surf=cssv('--surface-solid');
  const all=series.flatMap(s=>s.pts);c.font='600 12px system-ui,sans-serif';c.fillStyle=muted;
  if(all.length<2){c.fillText('Save at least two scans to see a trend',12,28);return;}
  let x0=Math.min(...all.map(p=>p[0])),x1=Math.max(...all.map(p=>p[0]));if(x1===x0)x1=x0+86400000;
  let y0=Math.min(...all.map(p=>p[1])),y1=Math.max(...all.map(p=>p[1]));if(y1===y0){y0-=1;y1+=1;}
  const pad=(y1-y0)*0.15;y0-=pad;y1+=pad;const L=40,R=18,T=18,B=28,px=t=>L+(t-x0)/(x1-x0)*(W-L-R),py=v=>H-B-(v-y0)/(y1-y0)*(H-T-B);
  c.lineWidth=1;c.strokeStyle=line;c.setLineDash([]);
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4,y=Math.round(py(v))+.5;c.beginPath();c.moveTo(L,y);c.lineTo(W-R,y);c.stroke();c.fillText(v.toFixed(0),6,y+4);}
  const fd=t=>new Date(t).toLocaleDateString(undefined,{month:'short',day:'numeric'});
  c.fillText(fd(x0),L,H-8);const e=fd(x1);c.fillText(e,W-R-c.measureText(e).width,H-8);
  c.setLineDash([4,4]);c.strokeStyle='#fb923c';for(const m of marks){if(m<x0||m>x1)continue;c.beginPath();c.moveTo(px(m),T);c.lineTo(px(m),H-B);c.stroke();}
  c.setLineDash([]);
  c.save();c.beginPath();c.rect(0,0,L+(W-L-R)*prog+8,H);c.clip();
  for(const s of series){
    if(!s.pts.length)continue;const col=s.color||cssv('--brand'),P=s.pts.map(p=>[px(p[0]),py(p[1])]);
    /* monotone cubic (Fritsch-Carlson): smooth, never overshoots the data */
    const n=P.length,m=[],tg=new Array(n).fill(0);
    for(let i=0;i<n-1;i++){const dx=P[i+1][0]-P[i][0];m.push(dx>0.5?(P[i+1][1]-P[i][1])/dx:0);}
    if(n>1){tg[0]=m[0];tg[n-1]=m[n-2];}
    for(let i=1;i<n-1;i++)tg[i]=m[i-1]*m[i]<=0?0:(m[i-1]+m[i])/2;
    for(let i=0;i<n-1;i++){if(m[i]===0){tg[i]=tg[i+1]=0;continue;}const a=tg[i]/m[i],b=tg[i+1]/m[i],h=a*a+b*b;if(h>9){const k=3/Math.sqrt(h);tg[i]=k*a*m[i];tg[i+1]=k*b*m[i];}}
    const path=()=>{c.moveTo(P[0][0],P[0][1]);for(let i=0;i<n-1;i++){const dx=P[i+1][0]-P[i][0];
      if(dx<=0.5){c.lineTo(P[i+1][0],P[i+1][1]);continue;}
      c.bezierCurveTo(P[i][0]+dx/3,P[i][1]+tg[i]*dx/3,P[i+1][0]-dx/3,P[i+1][1]-tg[i+1]*dx/3,P[i+1][0],P[i+1][1]);}};
    const g=c.createLinearGradient(0,T,0,H-B);g.addColorStop(0,hexA(col,.32));g.addColorStop(1,hexA(col,0));
    c.beginPath();path();c.lineTo(P[P.length-1][0],H-B);c.lineTo(P[0][0],H-B);c.closePath();c.fillStyle=g;c.fill();
    c.beginPath();path();c.strokeStyle=col;c.lineWidth=2;c.lineJoin='round';c.shadowColor=hexA(col,.7);c.shadowBlur=14;c.stroke();c.shadowBlur=0;
    P.forEach((p,i)=>{const last=i===P.length-1;c.beginPath();c.arc(p[0],p[1],last?5.5:4,0,6.283);c.fillStyle=col;c.fill();c.lineWidth=2;c.strokeStyle=surf;c.stroke();});
    if(prog===1&&canvas._mx==null){const p=P[P.length-1],v=String(s.pts[s.pts.length-1][1]);c.font='800 13px system-ui,sans-serif';c.fillStyle=ink;const w=c.measureText(v).width;c.fillText(v,Math.min(p[0]-w/2,W-w-2),Math.max(14,p[1]-12));}
  }
  c.restore();
  if(prog===1&&canvas._mx!=null&&series[0]&&series[0].pts.length){
    const s=series[0],col=s.color||cssv('--brand');let bi=0,bd=1e9;s.pts.forEach((p,i)=>{const d=Math.abs(px(p[0])-canvas._mx);if(d<bd){bd=d;bi=i;}});
    const p=s.pts[bi],X=px(p[0]),Y=py(p[1]);
    c.strokeStyle=line2;c.lineWidth=1;c.setLineDash([3,3]);c.beginPath();c.moveTo(X,T);c.lineTo(X,H-B);c.stroke();c.setLineDash([]);
    c.beginPath();c.arc(X,Y,9,0,6.283);c.fillStyle=hexA(col,.25);c.fill();c.beginPath();c.arc(X,Y,5.5,0,6.283);c.fillStyle=col;c.fill();c.lineWidth=2;c.strokeStyle=surf;c.stroke();
    const t1=`${s.name||'Value'}: ${p[1]}`,t2=new Date(p[0]).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'});
    c.font='700 12.5px system-ui,sans-serif';const w1=c.measureText(t1).width;c.font='500 11.5px system-ui,sans-serif';const w2=c.measureText(t2).width;
    const bw=Math.max(w1,w2)+22,bh=44;let bx=X+12;if(bx+bw>W-4)bx=X-12-bw;let by=Math.max(4,Math.min(Y-bh/2,H-B-bh));
    c.fillStyle=surf;c.strokeStyle=line2;c.lineWidth=1;c.beginPath();if(c.roundRect)c.roundRect(bx,by,bw,bh,10);else c.rect(bx,by,bw,bh);c.fill();c.stroke();
    c.fillStyle=ink;c.font='700 12.5px system-ui,sans-serif';c.fillText(t1,bx+11,by+19);c.fillStyle=muted;c.font='500 11.5px system-ui,sans-serif';c.fillText(t2,bx+11,by+35);
  }
}

/* ---------- auth screen ---------- */
function tplAuth(){
  const su=authMode==='signup';
  return `<div class="auth"><div class="themebtn">${themeBtn()}</div><div class="l"><div class="logo" style="font-size:24px;margin-bottom:26px">${LOGO}<span>SkinScope</span></div>
   <div class="art">${scanArt()}</div>
   <h1>Know your skin.<br><em>Get ahead of breakouts.</em></h1>
   <p class="muted" style="font-size:17px;max-width:520px">Upload a selfie, get a skin report in seconds, see what may be behind your breakouts, and follow a plan you can act on.</p>
   <div class="feat"><span class="chip">1</span><div><b>Upload a selfie</b><span class="muted small">Front and side photos, taken in daylight.</span></div></div>
   <div class="feat"><span class="chip">2</span><div><b>Get your report</b><span class="muted small">Acne, dark marks, oiliness, breakout outlook and a nutrition watch-list.</span></div></div>
   <div class="feat"><span class="chip">3</span><div><b>Follow your plan</b><span class="muted small">Foods, habits and a simple skincare routine, with progress tracking.</span></div></div>
   <p class="tiny muted" style="margin-top:20px">Your account and photos stay on this computer. Photos are analysed and never saved.</p></div>
   <div class="r"><div class="box card">
    <div class="tabs"><button class="${su?'':'on'}" data-act="authMode" data-m="login">Sign in</button><button class="${su?'on':''}" data-act="authMode" data-m="signup">Create account</button></div>
    ${su?`<label class="f">Your name</label><input type="text" id="a_name" maxlength="60" autocomplete="name">`:''}
    <label class="f">Email</label><input type="email" id="a_email" autocomplete="email">
    <label class="f">Password ${su?'(at least 8 characters)':''}</label><input type="password" id="a_pw" autocomplete="${su?'new-password':'current-password'}">
    <div class="err" id="a_err">${esc(authErr)}</div>
    <button class="btn p big" style="width:100%;justify-content:center" data-act="authGo">${su?'Create my account':'Sign in'}</button>
    <p class="tiny muted" style="margin-top:12px">${su?'By creating an account you confirm this is not medical advice.':'Forgot your password? In the terminal run: python skinscope_app.py --reset-password YOUR_EMAIL'}</p>
   </div></div></div>`;
}

/* ---------- shell ---------- */
function shell(active,inner){
  const nav=[['home','Home'],['scan','Scan'],['results','Results'],['history','History'],['diary','Diary'],['lab','Lab'],['insights','Insights']];
  return `<div class="top"><span class="logo" data-act="go" data-to="home">${LOGO}<span>SkinScope</span></span>
    <div class="nav">${nav.map(([k,l])=>`<a href="#/${k}" class="${active===k?'on':''}" title="${l}">${ic(k)}<span>${l}</span></a>`).join('')}</div>
    <div class="me">${themeBtn()}<a href="#/profile" title="Profile"><span class="avatar">${esc((me.name||'?').trim().charAt(0).toUpperCase()||'?')}</span><span class="nm">${esc(me.name)}</span></a><button class="btn sm" data-act="logout">Sign out</button></div></div>
    <main id="view">${inner}</main>`;
}

/* ---------- home ---------- */
function tplHome(d){
  const L=d.latest, r=d.today_routine||{};
  const steps=[['am_cleanse','Morning cleanse'],['spf','Sunscreen'],['pm_cleanse','Evening cleanse'],['treatment','Treatment / spot care'],['moisturiser','Moisturiser']];
  const due=d.scan_count===0?'Take your first scan':(d.next_due<=0?'Time for your weekly scan':`Next scan in ${d.next_due} day${d.next_due===1?'':'s'}`);
  return `<div class="hero"><div><h2>Hi ${esc(d.name)}, ready for your skin check?</h2>
    <p style="opacity:.92;max-width:560px">Upload a front photo (and two side photos if you like) to see your skin score, breakout outlook, nutrition watch-list and a plan.</p>
    <div class="row"><a class="btn p big" href="#/scan">Start skin scan</a><span style="opacity:.9">${esc(due)}</span></div></div><div class="hero-art">${scanArt()}</div></div>
   <div class="grid g4" style="margin:20px 0">
    <div class="card stat">${L?ring(L.score,110):`<div class="ic">${ic('scan')}</div><div class="n">--</div>`}<div class="small muted">Skin score</div>${L?`<div class="chip">${esc(L.grade)}</div>`:''}</div>
    <div class="card stat"><div class="ic">${ic('shield')}</div><div class="n">${L?esc(L.outlook.charAt(0).toUpperCase()+L.outlook.slice(1)):'--'}</div><div class="small muted">Breakout outlook</div>${L?`<a class="small" href="#/results/${L.id}">See why</a>`:''}</div>
    <div class="card stat"><div class="ic">${ic('dots')}</div><div class="n" ${L?`data-count="${L.spots}"`:''}>${L?L.spots:'--'}</div><div class="small muted">Active spots (latest)</div></div>
    <div class="card stat"><div class="ic">${ic('flame')}</div><div class="n" data-count="${d.routine_streak}">${d.routine_streak}</div><div class="small muted">Day routine streak</div></div>
   </div>
   ${d.profile_done?'':`<div class="note info" style="margin-bottom:18px"><b>Finish your profile</b> for safer, more accurate advice (diet, pregnancy status, ingredients you react to). <a href="#/profile">Open profile</a></div>`}
   ${d.lab?`<div class="card"><h3>Trigger Lab &middot; day ${d.lab.day} of ${d.lab.days}: ${esc(d.lab.label)}</h3><p class="small muted" style="margin:4px 0 10px">${esc(d.lab.ask)}</p><div class="row"><button class="btn ${d.lab.today===true?'p':''}" data-act="labCheckin" data-ok="1">Yes</button><button class="btn ${d.lab.today===false?'p':''}" data-act="labCheckin" data-ok="0">No</button><a class="small" href="#/lab">Open Trigger Lab</a></div></div>`:''}
   <div class="grid g21">
    <div class="card"><h3>Today's routine</h3><p class="small muted">Tick what you have done. It takes two seconds and builds your streak.</p>
      ${steps.map(([k,l])=>`<label class="checkrow"><input type="checkbox" data-act="routineTick" data-k="${k}" ${r[k]?'checked':''}> ${l}</label>`).join('')}
      <div class="note info" style="margin-top:12px"><b>Tip of the day.</b> ${esc(d.tip)}</div></div>
    <div class="card"><h3>This week</h3><pre style="white-space:pre-wrap;font:inherit;margin:0" class="small">${esc(d.weekly.text)}</pre>
      <h3 style="margin-top:16px">Recent scans</h3>
      ${d.recent.length?d.recent.map(s=>`<div class="row" style="justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--line)"><a href="#/results/${s.id}">${fmtDT(s.ts)}</a><span>${s.ok?'':'<span class="chip warn">low quality</span> '}score <b>${s.score}</b> &middot; ${s.spots} spots</span></div>`).join(''):'<p class="muted small">No scans yet.</p>'}</div>
   </div>
   <div class="card"><h3>What SkinScope looks at</h3><div class="grid g4">
    ${[['Acne','Counts inflamed and pustule-like spots in 5 face zones.'],['Dark marks','Finds darker spots that may be post-acne marks.'],['Oiliness & skin type','Compares shine in the T-zone and cheeks.'],['Redness evenness','Checks how uneven redness is across zones.'],['Breakout outlook','Estimates the chance of new spots from your skin and lifestyle.'],['Nutrition watch','Flags nutrients you may be low on from your diet answers.'],['Action plan','Foods, habits and a simple routine to follow.'],['Progress','Tracks score, spots and triggers over time.']].map(([a,b])=>`<div><b>${a}</b><div class="small muted">${b}</div></div>`).join('')}</div></div>`;
}

/* ---------- scan ---------- */
function tplQuestions(){
  const groups={};
  for(const q of cfg.questions){(groups[q.group]=groups[q.group]||[]).push(q);}
  const val=k=>sc.q[k];
  return Object.entries(groups).map(([g,qs])=>`<div class="qgrp"><h4>${esc(g)}</h4>${qs.map(q=>{
    let inp='';
    if(q.type==='number') inp=`<input type="number" data-q="${q.key}" data-kind="num" min="${q.min}" max="${q.max}" step="${q.step}" value="${val(q.key)!=null?val(q.key):(q.default!=null?q.default:'')}">`;
    else if(q.type==='bool') inp=`<label class="opt"><input type="checkbox" data-q="${q.key}" data-kind="bool" ${val(q.key)?'checked':''}> Yes</label>`;
    else if(q.type==='multi') inp=q.options.map(o=>`<label class="opt"><input type="checkbox" data-qm="${q.key}" value="${o[0]}" ${(val(q.key)||[]).includes(o[0])?'checked':''}> ${esc(o[1])}</label>`).join('');
    else { const opts=q.type==='freq'?cfg.freq:q.options; const cur=val(q.key)!=null?val(q.key):(q.type==='choice'?q.default:null);
      inp=opts.map(o=>`<label class="opt"><input type="radio" name="q_${q.key}" data-q="${q.key}" data-kind="num" value="${o[0]}" ${cur===o[0]?'checked':''}> ${esc(o[1])}</label>`).join(''); }
    return `<div class="qrow"><span class="ql">${esc(q.label)}</span><span class="row" style="gap:6px">${inp}</span></div>`;}).join('')}</div>`).join('');
}
function slotHtml(v){
  const info=VIEW_INFO[v], p=sc.photos[v], ck=sc.checks[v];
  if(p) return `<div class="slot has${ck?'':' checking'}" data-act="pickPhoto" data-v="${v}"><img src="${p}" alt="">${HAS_CAM?`<button class="retake" data-act="openCam" data-v="${v}" title="Retake with camera">${ic('camera')}</button>`:''}<button class="x" data-act="removePhoto" data-v="${v}" title="Remove">&times;</button><div class="cap">${esc(info.name)} &middot; ${ck?(ck.found?(ck.notes.length?'<span style="color:#fcd34d">'+esc(ck.notes[0])+'</span>':'&#10003; good photo'):'<span style="color:#fca5a5">no face found</span>'):'checking...'}</div></div>`;
  return `<div class="slot" data-act="pickPhoto" data-v="${v}"><div class="cam">${ic('camera')}</div><b>${esc(info.name)}</b><div class="small muted">${esc(info.sub)}</div><div class="tiny muted" style="margin-top:6px">${info.required?'Required':'Optional'} &middot; ${HAS_CAM?'take or upload a photo':'click to upload'}</div>${HAS_CAM?`<div class="row slot-acts"><button class="btn sm p" data-act="openCam" data-v="${v}">${ic('scan')} Use camera</button><button class="btn sm" data-act="pickPhoto" data-v="${v}">Upload</button></div>`:''}</div>`;
}
function tplScan(){
  const ready=!!sc.photos.front&&sc.checks.front&&sc.checks.front.found;
  return `<h2>Skin scan</h2><p class="muted">Step 1: add your photos. Step 2: answer a few quick questions. Step 3: analyse.</p>
  <div class="card"><h3>1. Add your photos</h3><div class="slots" id="slots">${VIEWS_ORDER.map(slotHtml).join('')}</div>
    <p class="small muted" style="margin-top:10px">The front photo is required. The side photos are optional and help count spots on each cheek. Photos are analysed on this computer and never saved.</p></div>
  <div class="card"><h3>2. Quick questions <span class="chip grey">recommended</span></h3><p class="small muted">These power your breakout outlook and nutrition watch-list. Without the food answers, the nutrition section stays locked.</p><div id="qform">${tplQuestions()}</div></div>
  <div class="card"><div class="row"><button class="btn p big" id="analyzeBtn" data-act="analyze" ${ready?'':'disabled'}>3. Analyse my skin</button><span class="small muted" id="analyzeHint">${ready?'':'Add a clear front photo to continue.'}</span></div></div>
  ${tplInstructions()}`;
}
const VIEWS_ORDER=['front','right','left'];
function tplInstructions(){
  return `<div class="card" id="howto"><h2>How to take your photos</h2>
   <div class="grid g3" style="margin:14px 0">
    <div class="angle"><b>1. Front</b>${angleSvg(0)}<p class="small muted">Face the camera straight on. Both ears visible, chin level, neutral expression. Your whole face from hairline to chin fills most of the frame.</p></div>
    <div class="angle"><b>2. Right cheek view</b>${angleSvg(-30)}<p class="small muted">Turn your head about 30 degrees to your <b>left</b> so the camera sees more of your <b>right</b> cheek. Keep your eyes on the camera.</p></div>
    <div class="angle"><b>3. Left cheek view</b>${angleSvg(30)}<p class="small muted">Turn your head about 30 degrees to your <b>right</b> so the camera sees more of your <b>left</b> cheek. Do not turn further than 30 degrees or the face may not be detected.</p></div></div>
   <div class="grid g2">
    <div><h3>Lighting and setup</h3><ul class="small">
      <li>Stand facing a window in daylight. Light should fall evenly on both cheeks.</li>
      <li>No flash, no overhead yellow light, no strong shadows from one side.</li>
      <li>Phone at eye level, about an arm's length away. Hold it steady.</li>
      <li>Clean face, no makeup or heavy moisturiser. Pull hair off the forehead.</li>
      <li>Remove glasses. Use the back camera if you can, or the front camera with beauty filters turned off.</li>
      <li>Take every photo in the same place and at about the same time of day.</li></ul></div>
    <div><h3>Common mistakes</h3><ul class="small">
      <li>Portrait mode, beauty mode or filters smooth out spots.</li>
      <li>Photos that are too dark, too bright or blurry are flagged and not counted in your trends.</li>
      <li>Wide-angle selfies close to the face distort the cheeks.</li>
      <li>On iPhone, turn off Settings, Camera, Mirror Front Camera so left and right stay correct.</li>
      <li>If Chrome cannot open an iPhone HEIC photo, use Safari or set Camera, Formats to Most Compatible.</li></ul></div></div>
   <div class="note info small">Privacy: photos are processed in memory on this computer and are not stored. Only numbers and spot positions are kept.</div></div>`;
}
async function loadPhotoFile(view,file){
  if(!file||!file.type.startsWith('image/')){toast('Please choose an image file');return;}
  const url=URL.createObjectURL(file);
  try{
    const img=new Image();img.src=url;await img.decode();
    let w=img.naturalWidth,h=img.naturalHeight;const maxW=1280;if(w>maxW){h=Math.round(h*maxW/w);w=maxW;}
    const c=document.createElement('canvas');c.width=w;c.height=h;const g=c.getContext('2d');
    g.translate(w,0);g.scale(-1,1);g.drawImage(img,0,0,w,h);  /* mirrored so left/right match a mirror */
    sc.canvases[view]=c;sc.photos[view]=c.toDataURL('image/jpeg',0.92);sc.checks[view]=null;
  }catch(e){toast('Could not open that image (iPhone HEIC photos need Safari, or export as JPEG).');URL.revokeObjectURL(url);return;}
  URL.revokeObjectURL(url);refreshScan();
  try{sc.checks[view]=await api('/api/check',{image:sc.photos[view]});}catch(e){sc.checks[view]={found:false,notes:[e.message]};}
  refreshScan();
}
function refreshScan(){const s=$('#slots');if(!s)return;s.innerHTML=VIEWS_ORDER.map(slotHtml).join('');
  const ready=!!sc.photos.front&&sc.checks.front&&sc.checks.front.found;const b=$('#analyzeBtn');if(b)b.disabled=!ready;
  const h=$('#analyzeHint');if(h)h.textContent=ready?'':(sc.photos.front?'Front photo: '+((sc.checks.front&&sc.checks.front.notes[0])||'checking...'):'Add a clear front photo to continue.');}
async function doAnalyze(){
  const b=$('#analyzeBtn');b.disabled=true;b.textContent='Analysing...';busy(true);
  try{
    const q={};for(const k in sc.q){if(sc.q[k]!==undefined&&sc.q[k]!==null)q[k]=sc.q[k];}
    for(const s of cfg.questions){if(q[s.key]===undefined&&s.default!==undefined&&s.type!=='freq')q[s.key]=s.default;}
    const photos={};for(const v of VIEWS_ORDER){if(sc.photos[v])photos[v]=sc.photos[v];}
    const out=await api('/api/scan',{photos,q});
    mem[out.id]={views:out.views,canvases:{...sc.canvases}};
    me.profile.last_q={...me.profile.last_q,...q};
    sc={photos:{},canvases:{},checks:{},q:{...me.profile.last_q}};
    location.hash='#/results/'+out.id;
  }catch(e){busy(false);toast(e.message);b.disabled=false;b.textContent='3. Analyse my skin';}
}

/* ---------- live camera capture with real-time photo checks ---------- */
const HAS_CAM=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia);
const CAM_CHECKS=[['face','Face found'],['light','Brightness'],['even','Even light'],['sharp','Sharp'],['dist','Close enough']];
let cam=null;
function camSet(state,msg){const el=$('#cam');el.classList.toggle('ok',state==='ok');el.classList.toggle('warn',state==='warn');if(msg!=null)$('#camMsg').textContent=msg;}
function camChips(st){$('#camChecks').innerHTML=CAM_CHECKS.map(([k,l])=>{const v=st?st[k]:null;return `<span class="${v===true?'y':v===false?'n':''}">${l}</span>`;}).join('');}
async function openCam(view){
  closeCam();
  cam={view,facing:'user',good:0,alive:true,stream:null,timer:null,cd:null};
  const info=VIEW_INFO[view];$('#camTitle').textContent=info.name;$('#camSub').textContent=info.sub+'. Fit your face inside the oval.';
  $('#camErr').classList.remove('on');$('#camCount').classList.remove('on');camChips(null);camSet('','Starting camera...');
  $('#cam').classList.add('on');await camStart();
}
async function camStart(){
  camStop();
  try{
    const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:cam.facing,width:{ideal:1280},height:{ideal:960}},audio:false});
    if(!cam||!cam.alive){s.getTracks().forEach(t=>t.stop());return;}
    cam.stream=s;const v=$('#camVideo');v.srcObject=s;v.classList.toggle('mirror',cam.facing==='user');
    try{await v.play();}catch(e){}
    camSet('','Looking for your face...');clearTimeout(cam.timer);cam.timer=setTimeout(camLoop,400);
  }catch(e){
    if(!cam)return;
    $('#camErrMsg').textContent=e&&e.name==='NotAllowedError'?'Camera access was blocked. Allow the camera for this page in your browser settings, or upload a photo instead.':'No camera could be opened on this device. You can upload a photo instead.';
    $('#camErr').classList.add('on');
  }
}
function camStop(){if(cam&&cam.stream){cam.stream.getTracks().forEach(t=>t.stop());cam.stream=null;}}
function closeCam(){if(!cam)return;cam.alive=false;clearTimeout(cam.timer);clearInterval(cam.cd);camStop();const v=$('#camVideo');if(v)v.srcObject=null;cam=null;$('#cam').classList.remove('on');}
function camFrame(maxW){const v=$('#camVideo');let w=v.videoWidth,h=v.videoHeight;if(!w||!h)return null;if(w>maxW){h=Math.round(h*maxW/w);w=maxW;}const c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d').drawImage(v,0,0,w,h);return c;}
async function camLoop(){
  if(!cam||!cam.alive)return;
  const c=cam.cd?null:camFrame(640);
  if(c){
    try{const r=await api('/api/check',{image:c.toDataURL('image/jpeg',0.8)});if(cam&&cam.alive&&!cam.cd)camJudge(r);}
    catch(e){if(e.status===401){closeCam();me=null;route();return;}}
  }
  if(cam&&cam.alive)cam.timer=setTimeout(camLoop,600);
}
function camJudge(r){
  const notes=r.notes||[],n=notes.join(' ').toLowerCase(),f=!!r.found;
  const st=f?{face:true,light:!/too dark|too bright/.test(n),even:!/uneven/.test(n),sharp:!/blurry/.test(n),dist:!/too small/.test(n)}:{face:false};
  camChips(st);
  const ok=f&&!notes.length;cam.good=ok?cam.good+1:0;
  if(!f)camSet('warn','Looking for your face... centre it in the oval');
  else if(!ok)camSet('warn',notes[0]);
  else camSet('ok',cam.good>=2?'Perfect, hold still':'Looks good, hold still');
  if(ok&&cam.good>=2&&$('#camAuto').checked)camCountdown();
}
function camCountdown(){
  if(!cam||cam.cd)return;let k=3;const el=$('#camCount');el.innerHTML=`<span>${k}</span>`;el.classList.add('on');
  cam.cd=setInterval(()=>{k--;if(!cam)return;if(k<=0){clearInterval(cam.cd);cam.cd=null;el.classList.remove('on');camShoot();}else el.innerHTML=`<span>${k}</span>`;},1000);
}
async function camShoot(){
  if(!cam)return;clearInterval(cam.cd);cam.cd=null;$('#camCount').classList.remove('on');
  const c=camFrame(1280);if(!c){toast('The camera is not ready yet');return;}
  const fl=$('#camFlash');fl.classList.remove('go');void fl.offsetWidth;fl.classList.add('go');
  const view=cam.view,blob=await new Promise(res=>c.toBlob(res,'image/jpeg',0.92));
  setTimeout(()=>{closeCam();if(blob){loadPhotoFile(view,blob);toast('Photo captured. Checking it now.');}},350);
}
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&cam)closeCam();});

/* ---------- results ---------- */
function likeChip(l){return {likely_low:'<span class="chip bad">Likely low</span>',possibly_low:'<span class="chip warn">Possibly low</span>',likely_fine:'<span class="chip good">Likely fine</span>',unknown:'<span class="chip grey">Answer food questions</span>'}[l];}
function tplResults(r){
  const s=r.score, o=r.outlook, P=r.plan, id=r.scan.id;
  const lv={low:['Low','#16a34a',1],moderate:['Moderate','#d97706',2],high:['High','#dc2626',3]}[o.level];
  const zl={low:'Low',watch:'Watch',high:'High'}, zc={low:'good',watch:'warn',high:'bad'};
  const m=mem[id];
  const views=m?Object.keys(m.views).filter(v=>m.views[v].found):[];
  const drivers=(o.drivers||[]);
  const seriesN=r.progress.series.length;
  return `<div class="row noprint" style="justify-content:space-between"><h2 style="margin:0">Your skin report</h2><div class="row"><button class="btn" data-act="print">Download / print report</button><a class="btn" href="#/summary">Doctor summary</a><a class="btn p" href="#/scan">New scan</a></div></div>
   <p class="muted small">${fmtDT(r.scan.ts)}${r.scan.ok?'':' &middot; <span class="chip warn">low photo quality: not counted in trends</span>'}</p>
   ${r.scan.notes.length?`<div class="note" style="margin-bottom:14px"><b>Photo quality:</b> ${r.scan.notes.map(esc).join('. ')}. Retake in daylight for a more reliable result.</div>`:''}
   <div class="pills"><a href="#/results/${id}" data-act="jump" data-to="sec-overview">Overview</a><a href="#/results/${id}" data-act="jump" data-to="sec-outlook">Breakout outlook</a><a href="#/results/${id}" data-act="jump" data-to="sec-nutrition">Nutrition</a><a href="#/results/${id}" data-act="jump" data-to="sec-plan">How to fix it</a><a href="#/results/${id}" data-act="jump" data-to="sec-skincare">Skincare</a><a href="#/results/${id}" data-act="jump" data-to="sec-progress">Progress</a></div>
   <div class="card" id="sec-overview"><div class="grid g21"><div>
     <div class="row" style="gap:24px">${ring(s.overall,150)}<div><h2 style="margin:0">${esc(s.grade)}</h2><p class="muted small" style="margin:4px 0">Skin type estimate: <b>${esc(s.skin_type)}</b></p>
      <div class="row"><span class="chip">${r.counts.spots} active spots</span><span class="chip grey">${r.counts.marks} dark marks</span></div></div></div>
     <div class="bars" style="margin-top:18px">${r.concerns.map(c=>`<div>${bar(c.label,c.score)}<div class="tiny muted" style="margin:-4px 0 0 130px">${esc(c.detail)}</div></div>`).join('')}</div></div>
    <div>${views.length?`<div class="row" style="margin-bottom:8px">${views.map((v,i)=>`<button class="btn sm ${i?'':'p'}" data-act="showView" data-v="${v}" data-id="${id}">${esc(VIEW_INFO[v].name)}</button>`).join('')}</div><canvas id="ovcanvas"></canvas><div class="legend" style="margin-top:6px"><span><span class="dot" style="background:${LT.i}"></span>inflamed</span><span><span class="dot" style="background:${LT.p}"></span>pustule-like</span><span><span class="dot" style="background:${LT.m}"></span>dark mark</span></div><div class="row noprint" style="margin-top:8px"><button class="btn sm" data-act="openReview" data-id="${id}">Check detections</button></div>`:lesionMap(r.scan.lesions)}
     ${m?Object.entries(m.views).filter(([v,x])=>!x.found).map(([v])=>`<p class="tiny muted">${esc(VIEW_INFO[v].name)}: no face found, not used.</p>`).join(''):''}</div></div>
    <table style="margin-top:14px"><thead><tr><th>Zone</th><th>Active spots</th><th>Dark marks</th><th>Shine</th></tr></thead><tbody>${Object.entries(r.counts.by_zone).map(([z,v])=>`<tr><td><span class="dot" style="background:${ZC[z]}"></span>${esc(cfg.labels[z])}</td><td>${v.spots}</td><td>${v.marks}</td><td>${v.shine.toFixed(1)}%</td></tr>`).join('')}</tbody></table>
    <p class="tiny muted">${r.scan.detector==='model'?'Active spots are found by a trained detection model and dark marks by colour rules.':'Detection is rule-based.'} Either can miss faint spots or mistake freckles, moles or irritation for lesions. Use Check detections to see how well it does on your face.</p></div>

   <div class="card" id="sec-outlook"><h2>Breakout outlook (next 7 days)</h2>
    <div class="grid g2"><div><div class="row"><span class="chip ${o.level==='low'?'good':o.level==='moderate'?'warn':'bad'}" style="font-size:15px;padding:4px 14px">${lv[0]} chance of new breakouts</span><span class="muted small">score ${o.score}/100</span></div>
      <div class="gauge">${[1,2,3].map(n=>`<i style="background:${n<=lv[2]?lv[1]:'var(--track)'};${n<=lv[2]?`box-shadow:0 0 14px ${lv[1]}`:''}"></i>`).join('')}</div>
      <p class="small muted">Confidence: ${esc(o.confidence)}</p>
      <h3 style="margin-top:14px">What is driving it</h3>
      ${drivers.length?drivers.map(d=>`<div class="checkrow"><span class="chip ${d.dir==='up'?'bad':'good'}">${d.dir==='up'?'raises':'lowers'}</span><span>${esc(d.text)}</span></div>`).join(''):'<p class="muted small">No strong drivers found.</p>'}
      <div class="note info small" style="margin-top:12px">${esc(o.note)}</div></div>
     <div><h3>By zone <span class="tiny muted">(${esc(Object.values(o.zones)[0].basis)})</span></h3>${Object.entries(o.zones).map(([z,v])=>`<div class="checkrow"><span class="chip ${zc[v.level]}" style="min-width:56px;text-align:center">${zl[v.level]}</span><span><b>${esc(cfg.labels[z])}</b> <span class="small muted">${v.reasons.length?esc(v.reasons.join(', ')):'nothing unusual'}</span></span></div>`).join('')}
       <div style="margin-top:10px">${faceMap(z=>({low:'#86efac',watch:'#fcd34d',high:'#fca5a5'}[o.zones[z].level]),false)}</div></div></div></div>

   <div class="card" id="sec-nutrition"><h2>Nutrition watch</h2>
    <div class="note" style="margin-bottom:14px"><b>A photo cannot measure vitamins or minerals.</b> This list is based on how often you eat the main food sources, plus what is known about nutrition and acne. Only a blood test can show your real levels. Do not start high-dose supplements without a doctor or pharmacist.</div>
    ${r.scan.have_answers?'':'<div class="note info" style="margin-bottom:14px">Answer the food questions on the Scan page to unlock this section.</div>'}
    <div class="grid g2">${r.nutrition.map(n=>`<div class="nut"><div class="row" style="justify-content:space-between"><h4>${esc(n.name)}</h4>${likeChip(n.likelihood)}</div>
      <p class="small" style="margin:6px 0">${esc(n.why)}</p>
      <p class="tiny muted" style="margin:0"><b>Link to acne:</b> ${esc(n.skin)} <span class="chip grey">${esc(n.evidence)}</span></p>
      <p class="tiny" style="margin:6px 0 0"><b>Good sources:</b> ${n.foods.map(esc).join(', ')}.</p>
      <p class="tiny muted" style="margin:4px 0 0">${esc(n.amount)} ${esc(n.caution)}</p></div>`).join('')}</div></div>

   <div class="card" id="sec-plan"><h2>How to fix it: your action plan</h2>
    <div class="grid g2"><div><h3>Eat more of</h3>${P.eat_more.length?P.eat_more.map(e=>`<div class="step"><div class="n">+</div><div><b>${esc(e.nutrient)}</b><div class="small muted">${e.foods.map(esc).join(', ')}</div></div></div>`).join(''):'<p class="muted small">'+(r.scan.have_answers?'Your food answers look balanced. Keep it up.':'Answer the food questions to get personal food suggestions.')+'</p>'}
      <h3 style="margin-top:14px">Go easier on</h3>${P.limit.map(l=>`<div class="step"><div class="n">-</div><div><b>${esc(l.food)}</b> <span class="chip grey">${esc(l.evidence)}</span><div class="small muted">${esc(l.text)}</div></div></div>`).join('')}</div>
     <div><h3>A day of eating (${esc({omni:'no restrictions',veg:'vegetarian',vegan:'vegan'}[r.diet])})</h3>${P.sample_day.map(t=>`<div class="checkrow"><span>&#127869;</span><span class="small">${esc(t)}</span></div>`).join('')}
      <h3 style="margin-top:14px">Grocery checklist</h3>${P.grocery.length?P.grocery.map(g=>`<label class="checkrow"><input type="checkbox" data-act="groc" data-k="${esc(g.key)}" ${r.grocery_state[g.key]?'checked':''}> <span>${esc(g.label)} <span class="tiny muted">for ${esc(g.for)}</span></span></label>`).join(''):'<p class="muted small">Nothing to add right now.</p>'}</div></div>
    <h3 style="margin-top:16px">Habits that help</h3><div class="grid g2">${P.lifestyle.map(l=>`<div class="checkrow"><span class="chip">${esc(l.title)}</span><span class="small">${esc(l.text)}</span></div>`).join('')}</div></div>

   <div class="card" id="sec-skincare"><h2>Skincare routine</h2>
    <div class="grid g2"><div><h3>Morning</h3>${P.routine.am.map((st,i)=>`<div class="step"><div class="n">${i+1}</div><div><b>${esc(st.step)}</b><div class="small muted">${esc(st.why)}</div></div></div>`).join('')}</div>
     <div><h3>Evening</h3>${P.routine.pm.map((st,i)=>`<div class="step"><div class="n">${i+1}</div><div><b>${esc(st.step)}</b><div class="small muted">${esc(st.why)}</div></div></div>`).join('')}</div></div>
    <div class="grid g2" style="margin-top:14px"><div><h3>Avoid</h3><ul class="small">${P.routine.avoid.map(a=>`<li>${esc(a)}</li>`).join('')}</ul></div><div><h3>Safety</h3><ul class="small">${P.routine.safety.map(a=>`<li>${esc(a)}</li>`).join('')}</ul></div></div>
    <div class="note ${P.doctor.needed?'bad':'info'}" style="margin-top:8px"><b>${P.doctor.needed?'We suggest seeing a doctor or dermatologist.':'When to see a doctor'}</b>
     ${P.doctor.reasons.length?'<ul class="small">'+P.doctor.reasons.map(x=>`<li>${esc(x)}</li>`).join('')+'</ul>':''}
     <div class="small">See a doctor for: ${P.doctor.general.map(esc).join('; ')}.</div></div></div>

   <div class="card" id="sec-progress"><h2>Progress</h2>
    ${r.progress.previous?`<p>Previous score <b>${r.progress.previous.score}</b> &rarr; now <b>${s.overall}</b> (${s.overall-r.progress.previous.score>=0?'+':''}${s.overall-r.progress.previous.score}). Active spots ${r.progress.previous.spots} &rarr; ${r.counts.spots}.</p>${noiseNote(r.progress.change)}`:'<p class="muted">This is your first comparable scan. Scan again in a week to see change.</p>'}
    ${seriesN>1?'<canvas class="chart" id="progChart"></canvas>':''}
    <a href="#/history">Open full history and compare scans</a></div>
   <p class="tiny muted">SkinScope is an information tool, not a medical device or diagnosis. Talk to a doctor about any skin or health concern.</p>`;
}
function drawOverlay(id,view){
  const m=mem[id]; if(!m||!m.canvases[view]) return; const cv=$('#ovcanvas'); if(!cv) return;
  const src=m.canvases[view], info=m.views[view]; cv.width=src.width;cv.height=src.height;const g=cv.getContext('2d');g.drawImage(src,0,0);
  const lw=Math.max(2,src.width/450);
  if(info.box){g.lineWidth=lw;g.strokeStyle='#16a34a';g.strokeRect(...info.box.map((v,i)=>v));}
  g.lineWidth=Math.max(1,lw/2);for(const z in (info.zones||{})){const b=info.zones[z];g.strokeStyle=ZC[z]+'aa';g.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
  g.lineWidth=lw;for(const l of info.lesions){g.strokeStyle=LT[l.t];g.beginPath();g.arc(l.x,l.y,l.r+3,0,6.283);g.stroke();}
}
function afterResults(id){
  const m=mem[id]; if(m){const v=Object.keys(m.views).find(v=>m.views[v].found); if(v) drawOverlay(id,v);}
  const c=$('#progChart'); if(c&&window._prog){drawChart(c,[{color:cssv('--brand'),name:'Skin score',pts:window._prog.map(p=>[p.ts*1000,p.score])}],[]);}
}

/* ---------- review modal ---------- */
function openReview(id){
  const m=mem[id]; if(!m) return; const view=Object.keys(m.views).find(v=>m.views[v].found&&m.canvases[v]); if(!view) return;
  const snap=m.canvases[view], info=m.views[view], b=info.box, padx=b[2]*0.08, pady=b[3]*0.08;
  const cx0=Math.max(0,b[0]-padx),cy0=Math.max(0,b[1]-pady),cx1=Math.min(snap.width,b[0]+b[2]+padx),cy1=Math.min(snap.height,b[1]+b[3]+pady);
  const cw=cx1-cx0,ch=cy1-cy0,W=Math.min(900,Math.round(cw*2)),sc2=W/cw,cv=$('#rvcanvas');cv.width=W;cv.height=Math.round(ch*sc2);
  rv={snap,cx0,cy0,cw,ch,sc:sc2,dets:info.lesions.map(l=>({x:(l.x-cx0)*sc2,y:(l.y-cy0)*sc2,r:Math.max(9,(l.r+3)*sc2),fp:false})),missed:[]};
  $('#rvmodal').style.display='block';drawReview();
}
function drawReview(){
  const cv=$('#rvcanvas'),g=cv.getContext('2d');g.drawImage(rv.snap,rv.cx0,rv.cy0,rv.cw,rv.ch,0,0,cv.width,cv.height);g.lineWidth=2;
  rv.dets.forEach(d=>{g.strokeStyle=d.fp?'#94a3b8':'#e5484d';g.setLineDash(d.fp?[4,4]:[]);g.beginPath();g.arc(d.x,d.y,d.r,0,6.283);g.stroke();
    if(d.fp){g.beginPath();g.moveTo(d.x-d.r*.6,d.y-d.r*.6);g.lineTo(d.x+d.r*.6,d.y+d.r*.6);g.moveTo(d.x+d.r*.6,d.y-d.r*.6);g.lineTo(d.x-d.r*.6,d.y+d.r*.6);g.stroke();}});
  g.setLineDash([]);g.strokeStyle='#3b82f6';rv.missed.forEach(m=>{g.beginPath();g.arc(m.x,m.y,10,0,6.283);g.moveTo(m.x-6,m.y);g.lineTo(m.x+6,m.y);g.moveTo(m.x,m.y-6);g.lineTo(m.x,m.y+6);g.stroke();});
  $('#rvstat').textContent=`${rv.dets.length} detected, ${rv.dets.filter(d=>d.fp).length} marked wrong, ${rv.missed.length} missed`;
}
$('#rvcanvas').addEventListener('click',e=>{
  if(!rv)return;const cv=$('#rvcanvas'),rc=cv.getBoundingClientRect(),x=(e.clientX-rc.left)*cv.width/rc.width,y=(e.clientY-rc.top)*cv.height/rc.height;
  let hit=null,best=1e9;rv.dets.forEach(d=>{const dd=Math.hypot(d.x-x,d.y-y);if(dd<d.r+8&&dd<best){best=dd;hit=d;}});
  if(hit)hit.fp=!hit.fp;else{const i=rv.missed.findIndex(m=>Math.hypot(m.x-x,m.y-y)<12);if(i>=0)rv.missed.splice(i,1);else rv.missed.push({x,y});}drawReview();
});

/* ---------- history ---------- */
let hist=null;
function tplHistory(h){
  hist=h; const ok=h.scans.filter(s=>s.ok);
  return `<h2>History</h2>
   <div class="card"><h3>Skin score over time</h3><canvas class="chart" id="hChart"></canvas>${h.hidden?`<p class="tiny muted">${h.hidden} older scan(s) made with a different detector, version or sensitivity are kept but hidden.</p>`:''}</div>
   <div class="card"><h3>Compare two scans</h3>${ok.length<2?'<p class="muted small">Save at least two good scans to compare them.</p>':`<div class="row"><select id="cmpA" data-act="cmpChange" style="max-width:280px">${ok.map(s=>`<option value="${s.id}">${fmtDT(s.ts)} - score ${s.score}</option>`).join('')}</select><span class="muted">vs</span><select id="cmpB" data-act="cmpChange" style="max-width:280px">${ok.map(s=>`<option value="${s.id}">${fmtDT(s.ts)} - score ${s.score}</option>`).join('')}</select></div>
     <div class="grid g2" style="margin-top:12px"><div id="mapA"></div><div id="mapB"></div></div><div class="note info small" id="cmpSum" style="margin-top:10px"></div>`}</div>
   <div class="card"><h3>All scans</h3>${h.scans.length?`<table><thead><tr><th>Date</th><th>Score</th><th>Grade</th><th>Spots</th><th>Marks</th><th></th></tr></thead><tbody>${h.scans.slice().reverse().map(s=>`<tr><td><a href="#/results/${s.id}">${fmtDT(s.ts)}</a> ${s.ok?'':'<span class="chip warn">low quality</span>'}</td><td><b>${s.score}</b></td><td>${esc(s.grade)}</td><td>${s.spots}</td><td>${s.marks}</td><td><button class="btn sm danger" data-act="delScan" data-id="${s.id}">Delete</button></td></tr>`).join('')}</tbody></table>`:'<p class="muted">No scans yet. <a href="#/scan">Take your first scan</a>.</p>'}</div>`;
}
function matchLesions(A,B){const used=new Set();let kept=0;for(const b of B){let hit=-1;A.forEach((a,i)=>{if(hit<0&&!used.has(i)&&(a[3]==='m')===(b[3]==='m')&&Math.abs(a[0]-b[0])<0.05&&Math.abs(a[1]-b[1])<0.05)hit=i;});if(hit>=0){used.add(hit);kept++;}}return{kept,added:B.length-kept,cleared:A.length-kept};}
function renderCompare(){
  if(!hist||!$('#cmpA'))return;const a=hist.scans.find(s=>s.id==$('#cmpA').value),b=hist.scans.find(s=>s.id==$('#cmpB').value);if(!a||!b)return;
  $('#mapA').innerHTML=`<div class="small muted" style="text-align:center">${fmtDT(a.ts)} - score ${a.score}</div>`+lesionMap(a.lesions);
  $('#mapB').innerHTML=`<div class="small muted" style="text-align:center">${fmtDT(b.ts)} - score ${b.score}</div>`+lesionMap(b.lesions);
  const m=matchLesions(a.lesions,b.lesions);$('#cmpSum').innerHTML=`Score ${a.score} &rarr; <b>${b.score}</b> (${b.score-a.score>=0?'+':''}${b.score-a.score}). Active spots ${a.spots} &rarr; <b>${b.spots}</b>. About ${m.cleared} cleared, ${m.added} new, ${m.kept} still there. ${noiseSentence(a.spots,b.spots)}`;
}
function afterHistory(){
  if(!hist)return;const ok=hist.scans.filter(s=>s.ok);drawChart($('#hChart'),[{color:cssv('--brand'),name:'Skin score',pts:ok.map(s=>[s.ts*1000,s.score])}],[]);
  if($('#cmpA')&&ok.length>=2){$('#cmpA').value=ok[Math.max(0,ok.length-2)].id;$('#cmpB').value=ok[ok.length-1].id;renderCompare();}
}

/* ---------- diary ---------- */
let diaryData=null, bzone=null, diaryDay=todayStr();
function tplDiary(d){
  diaryData=d; const e=d.entries.find(x=>x.day===diaryDay)||{}; const r=e.routine||{};
  const steps=[['am_cleanse','Morning cleanse'],['spf','Sunscreen'],['pm_cleanse','Evening cleanse'],['treatment','Treatment / spot care'],['moisturiser','Moisturiser']];
  return `<h2>Diary</h2><div class="grid g2">
   <div class="card"><h3>Daily check-in</h3><label class="f">Day</label><input type="date" id="dDay" value="${diaryDay}" data-act="diaryDay">
    <label class="f">Sleep last night (hours)</label><input type="number" id="dSleep" min="0" max="14" step="0.5" value="${e.sleep!=null?e.sleep:7}">
    <label class="f">Stress (1 low - 5 high)</label><input type="number" id="dStress" min="1" max="5" value="${e.stress!=null?e.stress:2}">
    <label class="f">Water (glasses)</label><input type="number" id="dWater" min="0" max="30" value="${e.water!=null?e.water:6}">
    <div style="margin-top:10px">${[['sugar','Lots of sugar'],['dairy','Dairy'],['new_product','New product'],['sweat','Heavy sweat / exercise']].map(([k,l])=>`<label class="opt" style="margin:0 6px 6px 0"><input type="checkbox" id="d_${k}" ${e[k]?'checked':''}> ${l}</label>`).join('')}</div>
    <div class="row"><button class="btn p" data-act="saveDiary">Save check-in</button></div></div>
   <div class="card"><h3>Routine for ${esc(diaryDay)}</h3><p class="small muted">Streak: <b>${d.streak}</b> day(s)</p>${steps.map(([k,l])=>`<label class="checkrow"><input type="checkbox" data-act="routineTick" data-k="${k}" data-day="${diaryDay}" ${r[k]?'checked':''}> ${l}</label>`).join('')}
    <h3 style="margin-top:16px">Log a breakout</h3><p class="small muted">Tap the zone, then log a new pimple as soon as you notice it. The Insights page uses this to test whether the outlook works for you.</p>
    <div class="clickable">${faceMap(z=>z===bzone?'#ff5fa8':'#8b84b0',true)}</div>
    <div class="row" style="margin-top:8px"><select id="bSev" style="max-width:190px"><option value="1">Small / mild</option><option value="2">Moderate</option><option value="3">Large / painful</option></select><input type="date" id="bDay" value="${todayStr()}" style="max-width:170px"><button class="btn p" data-act="logBreakout">Log breakout</button></div></div></div>
   <div class="card"><h3>Recent breakouts</h3>${d.breakouts.length?d.breakouts.slice(-10).reverse().map(b=>`<div class="row" style="justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line)"><span>${fmtDate(b.ts)} &middot; ${esc(cfg.labels[b.zone])} &middot; size ${b.severity}</span><button class="btn sm danger" data-act="delBreakout" data-id="${b.id}">Delete</button></div>`).join(''):'<p class="muted small">None logged.</p>'}</div>`;
}

/* ---------- insights ---------- */
function tplInsights(i){
  const dt=i.detector, v=i.validation;
  return `<h2>Insights</h2>
   <div class="card"><h3>How accurate is spot detection on your face?</h3>${dt.n===0?'<div class="note info">No reviews yet. After a scan, open the results and press <b>Check detections</b>. Tap circles that are not pimples and tap places where one was missed. After a few reviews this shows how often the detector is right for you.</div>'
    :`<div class="note info">${dt.n} review(s): it marked ${dt.detected} spots and you confirmed ${dt.detected-dt.false_pos}. You found ${dt.missed} it missed.</div><table><tr><th>Precision (of what it marked, how much was real)</th><td><b>${pct(dt.precision)}</b></td></tr><tr><th>Recall (of the real pimples, how many it found)</th><td><b>${pct(dt.recall)}</b></td></tr></table><p class="tiny muted">This is your own judgement on your own face, a practical measure and not a clinical one. If precision is low, try Low sensitivity in Profile. If recall is low, try High.</p>`}</div>
   <div class="card"><h3>Does the breakout outlook work for you?</h3><div class="note">${esc(v.verdict)}</div>
    <p class="tiny muted">A zone counts as caught if a good scan flagged it between ${v.lead_hours} hours and ${v.horizon_days} days before you logged a breakout there. Only earlier scans are used for each score.</p>
    ${v.rows&&v.testable>0?`<table><thead><tr><th>Flag level</th><th>Breakouts caught</th><th>Flags followed by a breakout</th><th>Flags raised</th></tr></thead><tbody>${v.rows.map(r=>`<tr><td>score &ge; ${r.threshold}</td><td>${r.caught}/${v.testable} (${pct(r.catch_rate)})</td><td>${pct(r.precision)}</td><td>${r.flagged}</td></tr>`).join('')}<tr><td class="muted">Simple rule: broke out here in the last 2 weeks</td><td>${v.recurrence.caught}/${v.testable} (${pct(v.recurrence.catch_rate)})</td><td>${pct(v.recurrence.precision)}</td><td>${v.recurrence.flagged}</td></tr></tbody></table><p class="tiny muted">Random zone-week base rate: ${pct(v.base_rate)}. ${v.n_scans} good scans, ${v.events} breakouts logged.</p>`:''}</div>
   <div class="card"><h3>Possible triggers (from your diary)</h3><table><thead><tr><th>Factor</th><th>Breakout within 2 days: with</th><th>without</th></tr></thead><tbody>${i.triggers.map(r=>r.enough?`<tr><td>${esc(r.label)}</td><td>${pct(r.rate_with)} (${r.n_with} days)</td><td>${pct(r.rate_without)} (${r.n_without} days)</td></tr>`:`<tr><td>${esc(r.label)}</td><td colspan="2" class="muted">Not enough days yet (${r.n_with} with / ${r.n_without} without, need 5 each)</td></tr>`).join('')}</tbody></table><p class="tiny muted">This shows association, not cause, and small samples are noisy.</p></div>`;
}

/* ---------- profile ---------- */
function tplProfile(){
  const p=me.profile, opt=(v,cur,l)=>`<option value="${v}" ${cur===v?'selected':''}>${l}</option>`;
  return `<h2>Profile</h2><div class="grid g2"><div class="card"><h3>About you</h3>
   <label class="f">Name</label><input type="text" id="pName" value="${esc(me.name)}" maxlength="60">
   <label class="f">Email</label><input type="text" value="${esc(me.email)}" disabled>
   <label class="f">Main goal</label><select id="pGoal">${opt('',p.goal,'Choose...')}${['Clear active acne','Prevent breakouts','Fade dark marks','Keep skin healthy'].map(g=>opt(g,p.goal,g)).join('')}</select>
   <label class="f">How would you describe your skin?</label><select id="pType">${[['','Not sure'],['oily','Oily'],['dry','Dry'],['combination','Combination'],['normal','Normal'],['sensitive','Sensitive']].map(([v,l])=>opt(v,p.self_skin_type,l)).join('')}</select>
   <label class="f">Eating pattern</label><select id="pDiet">${[['omni','I eat everything'],['veg','Vegetarian'],['vegan','Vegan']].map(([v,l])=>opt(v,p.diet,l)).join('')}</select>
   <label class="f">Pregnant, planning a pregnancy or breastfeeding? (affects which ingredients we suggest)</label><select id="pPreg">${[['unknown','Prefer not to say'],['no','No'],['yes','Yes']].map(([v,l])=>opt(v,p.pregnancy,l)).join('')}</select>
   <label class="f">Ingredients you react to (optional)</label><input type="text" id="pAvoid" value="${esc(p.avoid)}" maxlength="120" placeholder="e.g. fragrance, nuts">
   <label class="f">Spot detection sensitivity</label><select id="pSens">${[[0,'Low: fewer, surer'],[1,'Normal'],[2,'High: finds more']].map(([v,l])=>opt(v,+p.sens,l)).join('')}</select>
   <p class="tiny muted">Spot detector in use: <b>${cfg.detector==='model'?'trained model':'rule-based'}</b>. Scans made with different detectors are not compared with each other.</p>
   <p class="tiny muted">Keep sensitivity the same between scans. Changing it hides older scans from trends because they are not directly comparable.</p>
   <div class="row"><button class="btn p" data-act="saveProfile">Save profile</button></div></div>
  <div class="card"><h3>Your data</h3><p class="small muted">Everything is stored on this computer. Photos are never saved.</p>
   <div class="row" style="margin-bottom:10px"><a class="btn" href="#/summary">Doctor summary (printable)</a></div>
   <div class="row"><a class="btn" href="/api/export.csv?kind=scans">Export scans (CSV)</a><a class="btn" href="/api/export.csv?kind=diary">Export diary (CSV)</a><a class="btn" href="/api/export.csv?kind=breakouts">Export breakouts (CSV)</a></div>
   <h3 style="margin-top:22px">Delete account</h3><p class="small muted">Permanently deletes your account, scans, diary and reviews.</p>
   <label class="f">Enter your password to confirm</label><input type="password" id="pDelPw"><div class="row" style="margin-top:8px"><button class="btn danger" data-act="deleteAccount">Delete my account</button></div></div></div>`;
}

/* ---------- trigger lab, noise rule, doctor summary ---------- */
function noiseNote(c){
  if(!c) return '';
  const d=c.delta, sg=(d>0?'+':'')+d;
  return c.within_noise
    ? `<p class="small muted">A change of ${sg} spots is within normal scan-to-scan variation (about &plusmn;${c.noise}), so it is not clearly a real change. Rule of thumb, not a measurement.</p>`
    : `<p class="small">A change of ${sg} spots is bigger than normal scan-to-scan variation (about &plusmn;${c.noise}). Check your next scan to confirm it holds.</p>`;
}
function noiseSentence(a,b){
  const n=Math.max(2,Math.sqrt(2*Math.max((a+b)/2,1))),d=b-a;
  return Math.abs(d)<=n?`A change of ${d>0?'+':''}${d} spots is within normal scan-to-scan variation (about &plusmn;${n.toFixed(1)}), so it may just be noise.`:`A change of ${d>0?'+':''}${d} spots is bigger than normal variation (about &plusmn;${n.toFixed(1)}).`;
}
function tplAnalysis(a,state){
  const V={better:['Looks like a real improvement','good'],worse:['Spots were higher','bad'],unclear:['No clear difference','grey'],need_more:['Not enough scans yet','warn'],unreliable:['Not a fair test yet','warn']}[a.verdict]||['','grey'];
  const ad=a.adherence;
  return `<div class="row"><span class="chip ${V[1]}">${V[0]}${state==='active'?' (so far)':''}</span></div><p class="small" style="margin:8px 0">${esc(a.text)}</p>
   ${a.mean_base!=null?`<table><tbody><tr><th>Average active spots before you started</th><td>${a.mean_base} <span class="muted small">(${a.n_base} scans)</span></td></tr><tr><th>Average during (after a ${a.lead_in_days}-day lead-in)</th><td>${a.mean_during} <span class="muted small">(${a.n_during} scans)</span></td></tr><tr><th>Difference</th><td><b>${a.diff>0?'+':''}${a.diff}</b> <span class="muted small">95% range ${a.low>0?'+':''}${a.low} to ${a.high>0?'+':''}${a.high}</span></td></tr><tr><th>Days you kept to it</th><td>${ad.kept} of ${ad.logged} logged <span class="muted small">(${ad.elapsed} days so far)</span></td></tr>${a.breakouts?`<tr><th>Breakouts you logged per week</th><td>${a.breakouts.before_per_week} before, ${a.breakouts.during_per_week} during</td></tr>`:''}</tbody></table>`:''}`;
}
function tplLab(d){
  const A=d.active;
  let body='';
  if(A){
    const pctDone=Math.round(100*A.day/A.days);
    body=`<div class="card"><div class="row" style="justify-content:space-between"><h3 style="margin:0">Running: ${esc(A.label)}</h3><span class="chip">Day ${A.day} of ${A.days}</span></div>
      <div class="prog" style="margin:12px 0"><i style="width:${pctDone}%"></i></div>
      <p style="margin:6px 0"><b>${esc(A.ask)}</b></p>
      <div class="row"><button class="btn ${A.today===true?'p':''}" data-act="labCheckin" data-ok="1">Yes</button><button class="btn ${A.today===false?'p':''}" data-act="labCheckin" data-ok="0">No</button><a class="btn" href="#/scan">Scan now</a><button class="btn sm danger" data-act="labStop" data-id="${A.id}">Stop test</button></div>
      <p class="tiny muted" style="margin-top:8px">The first ${A.analysis.lead_in_days} days are a lead-in and are not counted, because skin takes a few weeks to respond. Scan every 2 days for the most reliable result.</p>
      <h3 style="margin-top:16px">Where it stands</h3>${tplAnalysis(A.analysis,'active')}</div>`;
  } else {
    body=`<div class="card"><h3>Start a test</h3>
      <p class="small ${d.baseline_scans>=d.min_scans?'muted':''}">${d.baseline_scans>=d.min_scans?`You have ${d.baseline_scans} good scans from the last 6 weeks to use as your "before". `:`<b>You have ${d.baseline_scans} good scan(s) from the last 6 weeks.</b> Take at least ${d.min_scans} (ideally 6 or more, every 2 days) before starting, so there is a fair "before". `}Only one test can run at a time.</p>
      <div class="grid g2" style="margin-top:12px">${d.factors.map(f=>`<div class="nut"><div class="row" style="justify-content:space-between"><h4>${esc(f.label)}</h4>${d.suggested&&d.suggested.key===f.key?'<span class="chip good">Suggested for you</span>':''}</div>
        <p class="small muted" style="margin:6px 0">${esc(f.detail)}</p>
        ${d.suggested&&d.suggested.key===f.key?`<p class="tiny" style="margin:0 0 6px"><b>Why:</b> ${esc(d.suggested.why)}</p>`:''}
        <p class="tiny muted" style="margin:0 0 8px">Evidence it affects acne: <span class="chip grey">${esc(f.evidence)}</span></p>
        <div class="row"><select id="dur_${f.key}" style="max-width:150px"><option value="28">4 weeks</option><option value="42">6 weeks</option></select><button class="btn p" data-act="labStart" data-k="${f.key}">Start test</button></div></div>`).join('')}</div></div>`;
  }
  const past=d.past.length?`<div class="card"><h3>Finished tests</h3>${d.past.map(p=>`<div style="padding:10px 0;border-bottom:1px solid var(--line)"><div class="row" style="justify-content:space-between"><b>${esc(p.label)}</b><span class="muted small">${fmtDate(p.start_ts)} &middot; ${p.days} days${p.state==='stopped'?' &middot; stopped early':''}</span></div>${tplAnalysis(p.analysis,p.state)}</div>`).join('')}</div>`:'';
  return `<h2>Trigger Lab</h2>
   <div class="card"><h3>Find what triggers YOUR skin</h3><p class="small muted">Generic advice says "cut dairy". This tests it on you: change <b>one</b> thing for 4 to 6 weeks, keep scanning, answer one question a day, and SkinScope compares your spots with your own "before" using honest statistics. It only reports a change when it is clearly bigger than normal wobble, so <b>small effects will not show up</b>. In my simulations, scanning every 2 days found a large real improvement about 9 times in 10, while scanning every 4 days found it fewer than 5 times in 10.</p></div>
   ${body}${past}
   <div class="card"><h3>Read results carefully</h3><ul class="small">${d.caveats.map(c=>`<li>${esc(c)}</li>`).join('')}</ul></div>`;
}
function tplSummary(s){
  window._sum=s.scans;
  const dietL={omni:'No restrictions',veg:'Vegetarian',vegan:'Vegan'}[s.profile.diet]||s.profile.diet;
  const sr=s.self_report;
  return `<div class="row noprint" style="justify-content:space-between"><h2 style="margin:0">Skin history summary</h2><button class="btn p" data-act="print">Print / save as PDF</button></div>
   <p class="muted small">Prepared for ${esc(s.name)} on ${esc(s.generated)}. Self-tracked with SkinScope, an information tool that is not a medical device. Spot counts come from an unvalidated photo detector and may contain errors.</p>
   <div class="card"><h3>Overview</h3><table><tbody>
     <tr><th>Main goal</th><td>${esc(s.profile.goal||'not set')}</td></tr><tr><th>Skin type (own description)</th><td>${esc(s.profile.skin_type||'not set')}</td></tr>
     <tr><th>Eating pattern</th><td>${esc(dietL)}</td></tr><tr><th>Ingredients reacted to</th><td>${esc(s.profile.avoid||'none listed')}</td></tr>
     <tr><th>Routine logged (last 30 days)</th><td>${s.routine_days_30} of 30 days</td></tr>
     ${Object.keys(sr).length?`<tr><th>Latest self-report</th><td>${[sr.sleep!=null?'sleep '+sr.sleep+' h':'',sr.stress!=null?'stress '+sr.stress+'/5':'',sr.water!=null?'water '+sr.water+' glasses':'',(sr.supps&&sr.supps.length)?'supplements: '+sr.supps.join(', '):''].filter(Boolean).map(esc).join(' &middot; ')}</td></tr>`:''}</tbody></table>
     <p class="tiny muted">Add any medicines, allergies or other conditions yourself when you see a doctor.</p></div>
   <div class="card"><h3>Scan history (latest good scans)</h3>${s.scans.length?`<canvas class="chart" id="sumChart"></canvas><table><thead><tr><th>Date</th><th>Score</th><th>Grade</th><th>Active spots</th><th>Pustule-like</th><th>Dark marks</th></tr></thead><tbody>${s.scans.slice().reverse().map(x=>`<tr><td>${fmtDate(x.ts)}</td><td>${x.score}</td><td>${esc(x.grade)}</td><td>${x.spots}</td><td>${x.pustules}</td><td>${x.marks}</td></tr>`).join('')}</tbody></table>`:'<p class="muted">No good scans yet.</p>'}</div>
   <div class="grid g2"><div class="card"><h3>Latest scan by zone</h3>${s.latest_zones?`<table><thead><tr><th>Zone</th><th>Spots</th><th>Marks</th></tr></thead><tbody>${Object.entries(s.latest_zones).map(([z,v])=>`<tr><td>${esc(s.labels[z])}</td><td>${v.spots}</td><td>${v.marks}</td></tr>`).join('')}</tbody></table>`:'<p class="muted">No scans.</p>'}</div>
    <div class="card"><h3>Breakouts you logged (last 90 days)</h3><table><tbody>${Object.entries(s.breakouts_90d).map(([z,n])=>`<tr><td>${esc(s.labels[z])}</td><td>${n}</td></tr>`).join('')}</tbody></table></div></div>
   <div class="card"><h3>Possible triggers from your diary</h3>${s.triggers.length?`<table><thead><tr><th>Factor</th><th>Breakout within 2 days: with</th><th>without</th></tr></thead><tbody>${s.triggers.map(r=>`<tr><td>${esc(r.label)}</td><td>${pct(r.rate_with)} (${r.n_with} days)</td><td>${pct(r.rate_without)} (${r.n_without} days)</td></tr>`).join('')}</tbody></table><p class="tiny muted">Association only, small samples.</p>`:'<p class="muted small">Not enough diary days yet.</p>'}</div>
   <div class="card"><h3>Personal tests (Trigger Lab)</h3>${s.experiments.length?s.experiments.map(x=>`<div style="padding:6px 0;border-bottom:1px solid var(--line)"><b>${esc(x.label)}</b> <span class="muted small">${fmtDate(x.start_ts)}, ${x.days} days${x.state==='stopped'?', stopped early':''}</span><div class="small">${esc(x.text)}</div></div>`).join(''):'<p class="muted small">No finished tests.</p>'}</div>`;
}

/* ---------- routing ---------- */
const VIEWS={
  home:async()=>tplHome(await api('/api/home')),
  scan:async()=>{ if(!Object.keys(sc.q).length) sc.q={...(me.profile.last_q||{})}; return tplScan(); },
  results:async(arg)=>{
    let id=arg; if(!id){const h=await api('/api/history');const ok=h.scans.filter(s=>s.ok).concat(h.scans.filter(s=>!s.ok));const last=h.scans[h.scans.length-1];if(!last)return `<div class="card"><h2>No results yet</h2><p class="muted">Take your first scan to see your report.</p><a class="btn p" href="#/scan">Start a scan</a></div>`;id=last.id;}
    const r=await api('/api/scan/'+id); window._prog=r.progress.series; window._resId=r.scan.id; return tplResults(r);
  },
  history:async()=>tplHistory(await api('/api/history')),
  diary:async()=>tplDiary(await api('/api/diary')),
  insights:async()=>tplInsights(await api('/api/insights')),
  profile:async()=>tplProfile(),
  lab:async()=>tplLab(await api('/api/lab')),
  summary:async()=>tplSummary(await api('/api/summary')),
};
const AFTER={results:()=>afterResults(window._resId),history:afterHistory,summary:()=>{const c=$('#sumChart');if(c&&window._sum)drawChart(c,[{color:cssv('--brand'),name:'Active spots',pts:window._sum.map(x=>[x.ts*1000,x.spots])}],[]);}};
function mount(html){$('#app').innerHTML=html;}
async function route(){
  const parts=(location.hash.replace(/^#\/?/,'')||'home').split('/'),name=parts[0],arg=parts[1];
  if(!me){mount(tplAuth());return;}
  const fn=VIEWS[name]||VIEWS.home,key=VIEWS[name]?name:'home';
  busy(false);closeCam();window._key=key;
  mount(shell(key,'<div class="card loading"><span class="spinner"></span><span class="muted">Loading...</span></div>'));
  try{const html=await fn(arg);$('#view').innerHTML=html;window.scrollTo(0,0);fx($('#view'));if(AFTER[key])AFTER[key]();}
  catch(e){if(e.status===401){me=null;route();return;}$('#view').innerHTML=`<div class="card"><h3>Something went wrong</h3><p class="muted">${esc(e.message)}</p></div>`;}
}
window.addEventListener('hashchange',route);
window.addEventListener('resize',()=>{if(location.hash.startsWith('#/history'))afterHistory();});

/* ---------- actions ---------- */
async function reloadMe(){const j=await api('/api/me');me=j.user;if(j.questions){cfg={questions:j.questions,freq:j.freq,labels:j.labels,zones:j.zones,detector:j.detector};}}
const ACT={
  openCam:(a,e)=>{e.stopPropagation();openCam(a.dataset.v);},
  camClose:()=>closeCam(),
  camShoot:()=>camShoot(),
  camFlip:async()=>{if(!cam)return;cam.facing=cam.facing==='user'?'environment':'user';cam.good=0;await camStart();},
  camUpload:()=>{const v=cam?cam.view:'front';closeCam();pendingView=v;$('#fileIn').click();},
  theme:()=>{THEME=THEME==='dark'?'light':'dark';applyTheme();$$('[data-act=theme]').forEach(b=>b.innerHTML=ic(THEME==='dark'?'sun':'moon'));const k=window._key;if(me&&AFTER[k])AFTER[k]();},
  authMode:a=>{authMode=a.dataset.m;authErr='';mount(tplAuth());},
  authGo:async()=>{
    const body={email:$('#a_email').value,password:$('#a_pw').value};if(authMode==='signup')body.name=$('#a_name').value;
    try{await api(authMode==='signup'?'/api/signup':'/api/login',body);await reloadMe();location.hash='#/home';route();}
    catch(e){authErr=e.message;const el=$('#a_err');if(el)el.textContent=authErr;}
  },
  logout:async()=>{try{await api('/api/logout',{});}catch(e){}me=null;sc={photos:{},canvases:{},checks:{},q:{}};location.hash='';route();},
  go:a=>{location.hash='#/'+a.dataset.to;},
  pickPhoto:(a,e)=>{if(e.target.closest('[data-act=removePhoto]'))return;pendingView=a.dataset.v;$('#fileIn').click();},
  removePhoto:(a,e)=>{e.stopPropagation();const v=a.dataset.v;delete sc.photos[v];delete sc.canvases[v];delete sc.checks[v];refreshScan();},
  analyze:()=>doAnalyze(),
  jump:(a,e)=>{e.preventDefault();const el=document.getElementById(a.dataset.to);if(el)el.scrollIntoView({behavior:'smooth',block:'start'});},
  showView:a=>{drawOverlay(+a.dataset.id,a.dataset.v);$$('[data-act=showView]').forEach(b=>b.classList.toggle('p',b===a));},
  openReview:a=>openReview(+a.dataset.id),
  rvCancel:()=>{$('#rvmodal').style.display='none';rv=null;},
  rvSubmit:async()=>{try{await api('/api/review',{detected:rv.dets.length,false_pos:rv.dets.filter(d=>d.fp).length,missed:rv.missed.length});toast('Review saved. See Insights for accuracy.');$('#rvmodal').style.display='none';rv=null;}catch(e){toast(e.message);}},
  print:()=>window.print(),
  groc:async a=>{try{await api('/api/profile',{grocery:{[a.dataset.k]:a.checked}});me.profile.grocery={...me.profile.grocery,[a.dataset.k]:a.checked};}catch(e){toast(e.message);}},
  routineTick:async a=>{try{await api('/api/diary',{day:a.dataset.day||todayStr(),patch:{routine:{[a.dataset.k]:a.checked}}});}catch(e){toast(e.message);}},
  cmpChange:()=>renderCompare(),
  delScan:async a=>{if(!confirm('Delete this scan?'))return;await api('/api/delete',{kind:'scan',id:+a.dataset.id});route();},
  diaryDay:a=>{diaryDay=a.value||todayStr();route();},
  saveDiary:async()=>{
    try{await api('/api/diary',{day:$('#dDay').value,patch:{sleep:+$('#dSleep').value,stress:+$('#dStress').value,water:+$('#dWater').value,sugar:$('#d_sugar').checked,dairy:$('#d_dairy').checked,new_product:$('#d_new_product').checked,sweat:$('#d_sweat').checked}});toast('Check-in saved');}catch(e){toast(e.message);}
  },
  pickZone:a=>{bzone=a.dataset.z;$('#view').querySelector('.clickable').innerHTML=faceMap(z=>z===bzone?'#ff5fa8':'#8b84b0',true);},
  logBreakout:async()=>{if(!bzone){toast('Tap a zone on the face first');return;}
    try{await api('/api/breakout',{zone:bzone,severity:+$('#bSev').value,day:$('#bDay').value});toast('Breakout logged');bzone=null;route();}catch(e){toast(e.message);}},
  delBreakout:async a=>{if(!confirm('Delete this entry?'))return;await api('/api/delete',{kind:'breakout',id:+a.dataset.id});route();},
  labStart:async a=>{const k=a.dataset.k;try{await api('/api/lab/start',{factor:k,days:+$('#dur_'+k).value});toast('Test started. Scan every 2 days and answer the daily question.');route();}catch(e){toast(e.message);}},
  labCheckin:async a=>{try{await api('/api/lab/checkin',{ok:a.dataset.ok==='1'});toast('Saved');route();}catch(e){toast(e.message);}},
  labStop:async a=>{if(!confirm('Stop this test? You can still see the numbers so far.'))return;try{await api('/api/lab/stop',{id:+a.dataset.id});route();}catch(e){toast(e.message);}},
  saveProfile:async()=>{
    try{await api('/api/profile',{name:$('#pName').value,goal:$('#pGoal').value,self_skin_type:$('#pType').value,diet:$('#pDiet').value,pregnancy:$('#pPreg').value,avoid:$('#pAvoid').value,sens:+$('#pSens').value});await reloadMe();toast('Profile saved');route();}catch(e){toast(e.message);}
  },
  deleteAccount:async()=>{
    if(!confirm('This permanently deletes your account and all data. Continue?'))return;
    try{await api('/api/account/delete',{password:$('#pDelPw').value});me=null;location.hash='';toast('Account deleted');route();}catch(e){toast(e.message);}
  },
};
document.addEventListener('click',e=>{const a=e.target.closest('[data-act]');if(!a)return;const t=a.dataset.act;
  if(a.tagName==='INPUT'||a.tagName==='SELECT')return;   /* inputs are handled on change */
  if(ACT[t]){ACT[t](a,e);}});
document.addEventListener('change',e=>{
  const t=e.target;
  if(t.dataset.act&&(t.tagName==='INPUT'||t.tagName==='SELECT')&&ACT[t.dataset.act]){ACT[t.dataset.act](t,e);return;}
  if(t.dataset.q){const k=t.dataset.q;sc.q[k]=t.dataset.kind==='bool'?t.checked:(t.value===''?undefined:Number(t.value));return;}
  if(t.dataset.qm){const k=t.dataset.qm,cur=new Set(sc.q[k]||[]);t.checked?cur.add(t.value):cur.delete(t.value);sc.q[k]=[...cur];}
});
document.addEventListener('keydown',e=>{if(e.key==='Enter'&&!me&&(e.target.id==='a_pw'||e.target.id==='a_email'||e.target.id==='a_name'))ACT.authGo();});
$('#fileIn').addEventListener('change',e=>{const f=e.target.files[0];e.target.value='';if(f&&pendingView)loadPhotoFile(pendingView,f);});

/* ---------- boot ---------- */
(async()=>{try{await reloadMe();}catch(e){}route();})();
</script>
</body></html>
'''


if __name__ == "__main__":
    main()

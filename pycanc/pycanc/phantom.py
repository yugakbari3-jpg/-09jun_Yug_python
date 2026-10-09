"""
Procedural low-dose chest CT phantom (Hounsfield units) so the whole pipeline
and UI can be exercised without patient data.

Anatomy: body wall with fat layer, ribs, sternum, spine, scanner table, lungs
with branching vessel trees, trachea + main bronchi, heart, aorta, and an
optional (solid / part-solid / ground-glass, smooth or spiculated) nodule.
"""
from __future__ import annotations

import numpy as np

from .preprocess import CTVolume


def _segment_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[..., None] * ab), axis=-1)


def _paint_tube(vol, a, b, r, hu, spacing, where=None, mode="max"):
    """Rasterise a capsule from a to b (mm, z/y/x, y/x centred) with radius r into vol."""
    origin = np.array([0.0, -vol.shape[1] / 2 * spacing[1], -vol.shape[2] / 2 * spacing[2]])
    a, b = a - origin, b - origin
    lo = np.floor((np.minimum(a, b) - r - 1) / spacing).astype(int)
    hi = np.ceil((np.maximum(a, b) + r + 1) / spacing).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, vol.shape)
    if np.any(hi <= lo):
        return
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    P = np.stack(np.meshgrid(*[np.arange(l, h) * s for l, h, s in zip(lo, hi, spacing)], indexing="ij"), -1)
    d = _segment_dist(P, a, b)
    m = d <= r
    if where is not None:
        m &= where[sl]
    sub = vol[sl]
    sub[m] = np.maximum(sub[m], hu) if mode == "max" else hu


def make_phantom(
    seed: int = 7,
    nodule: bool = True,
    nodule_mm: float = 14.0,
    nodule_type: str = "solid",      # solid | part-solid | ground-glass
    spiculated: bool = True,
    nodule_side: str = "right",      # patient side
    nodule_level: float = 0.35,      # 0 = apex, 1 = base
    shape=(128, 256, 256),
    spacing=(2.5, 1.40625, 1.40625),
) -> CTVolume:
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    sz, sy, sx = spacing
    z = (np.arange(Z) * sz)[:, None, None]
    y = ((np.arange(Y) - Y / 2) * sy)[None, :, None]
    x = ((np.arange(X) - X / 2) * sx)[None, None, :]
    zlen = Z * sz
    zn = z / zlen  # 0 top .. 1 bottom

    vol = np.full(shape, -1000.0, np.float32)

    # ---------------- body --------------------------------------------------
    ax = 150 + 12 * np.sin(np.pi * np.clip(zn * 1.2, 0, 1))
    ay = 108 + 8 * np.sin(np.pi * np.clip(zn * 1.1, 0, 1))
    r_body = np.sqrt((x / ax) ** 2 + ((y + 5) / ay) ** 2)
    body = r_body < 1
    vol[body] = -95  # fat
    inner = r_body < 1 - 14 / ay
    vol[inner] = 35  # muscle / soft tissue

    # scanner table (curved couch)
    table = (np.abs(y - (ay.max() + 22 + 0.0006 * x ** 2)) < 3) & (np.abs(x) < 190)
    vol[np.broadcast_to(table, shape)] = 250

    # ---------------- lungs -------------------------------------------------
    lungs = np.zeros(shape, bool)
    for side in (-1, 1):  # -1 image-left = patient right
        cx = side * 68
        top, bot = 0.06 * zlen, (0.86 if side == -1 else 0.90) * zlen
        t = np.clip((z - top) / (bot - top), 0, 1)
        prof = np.sqrt(np.clip(np.sin(np.pi * np.clip(t * 0.62 + 0.02, 0, 1)), 0, 1))
        lx = (58 if side == -1 else 54) * prof
        ly = 82 * prof
        r = np.sqrt(((x - cx) / np.maximum(lx, 1e-3)) ** 2 + ((y - 8) / np.maximum(ly, 1e-3)) ** 2)
        lung = (r < 1) & (z > top) & (z < bot)
        # diaphragm dome
        dome = bot - 30 * (1 - np.clip(((x - cx) / 70) ** 2 + ((y - 8) / 90) ** 2, 0, 1))
        lung &= z < dome
        # mediastinal notch
        lung &= ~((np.abs(x) < 26 + 10 * zn) & (y < 40))
        lungs |= lung
    tex = rng.normal(0, 1, shape).astype(np.float32)
    # cheap smoothing of texture
    for ax_ in range(3):
        tex = (tex + np.roll(tex, 1, ax_) + np.roll(tex, -1, ax_)) / 3
    vol[lungs] = -865 + 18 * tex[lungs]
    # gravity-dependent density gradient (posterior lungs slightly denser)
    grad = np.broadcast_to(30 * (y / 100), shape)
    vol[lungs] += grad[lungs]

    # ---------------- mediastinum -------------------------------------------
    heart = (((x - 22) / 62) ** 2 + ((y + 12) / 52) ** 2 + ((z - 0.66 * zlen) / (0.22 * zlen)) ** 2) < 1
    # pericardial fat crescent
    peri = (((x - 22) / 68) ** 2 + ((y + 12) / 58) ** 2 + ((z - 0.66 * zlen) / (0.24 * zlen)) ** 2) < 1
    vol[peri & ~heart & ~lungs] = -80
    vol[heart] = 42

    spacing_a = np.array(spacing)
    # descending aorta + arch
    _paint_tube(vol, np.array([0.30 * zlen, 40, 22]), np.array([0.95 * zlen, 44, 18]), 13, 45, spacing_a, mode="set")
    _paint_tube(vol, np.array([0.30 * zlen, -20, 10]), np.array([0.30 * zlen, 40, 22]), 13, 45, spacing_a, mode="set")
    _paint_tube(vol, np.array([0.30 * zlen, -20, 10]), np.array([0.55 * zlen, -15, 10]), 15, 45, spacing_a, mode="set")
    # SVC / pulmonary trunk
    _paint_tube(vol, np.array([0.18 * zlen, -12, -28]), np.array([0.55 * zlen, -8, -30]), 9, 40, spacing_a, mode="set")
    _paint_tube(vol, np.array([0.42 * zlen, -25, 22]), np.array([0.52 * zlen, -30, 30]), 14, 45, spacing_a, mode="set")

    # trachea + bronchi (air with soft-tissue wall)
    carina = np.array([0.36 * zlen, -2, 0])
    tr_top = np.array([0.0, -18, 0])
    for a, b, r in [(tr_top, carina, 9),
                    (carina, carina + [28, 8, -34], 6.5),
                    (carina, carina + [36, 10, 38], 6.0)]:
        _paint_tube(vol, a, b, r + 2, 30, spacing_a, mode="set")
    for a, b, r in [(tr_top, carina, 9),
                    (carina, carina + [28, 8, -34], 6.5),
                    (carina, carina + [36, 10, 38], 6.0)]:
        _paint_tube(vol, a, b, r, -990, spacing_a, mode="set")

    # ---------------- skeleton ----------------------------------------------
    sp_r = np.sqrt(x ** 2 + (y - 78) ** 2)
    spine = np.broadcast_to(sp_r < 19, shape)
    vol[spine] = 260
    vol[np.broadcast_to((sp_r < 19) & (sp_r > 15.5), shape)] = 750
    canal = np.broadcast_to(np.sqrt(x ** 2 + (y - 104) ** 2) < 8, shape)
    lamina = np.broadcast_to((np.sqrt(x ** 2 + (y - 104) ** 2) < 14) & (y > 96), shape)
    vol[lamina] = 650
    vol[canal] = 25
    # vertebral discs
    disc = (np.mod(z, 26) < 4) & np.broadcast_to(sp_r < 19, shape)
    vol[disc] = 70
    # sternum
    st = np.broadcast_to((np.abs(x) < 14) & (np.abs(y + ay - 22) < 7), shape) & (zn > 0.2) & (zn < 0.75)
    vol[st] = 500
    # ribs: elliptical shell arcs, periodic in z, descending anteriorly
    theta = np.arctan2(y, x)
    for k in range(11):
        z0 = (0.05 + k * 0.083) * zlen
        drop = 22 * (1 - np.sin(theta)) * 0.5     # anterior ribs sit lower
        zc = z0 + drop
        rr = np.sqrt((x / (ax - 18)) ** 2 + ((y + 5) / (ay - 18)) ** 2)
        rib = (np.abs(rr - 1) < 7 / ay) & (np.abs(z - zc) < 4.5) & (np.sin(theta) > -0.85)
        vol[rib & ~lungs] = 620
        core = (np.abs(rr - 1) < 3.5 / ay) & (np.abs(z - zc) < 2.5) & (np.sin(theta) > -0.85)
        vol[core & ~lungs] = 300
    # scapulae
    for side in (-1, 1):
        sc = (np.abs(np.sqrt(((x - side * 95) / 55) ** 2 + ((y - 70) / 30) ** 2) - 1) < 0.08) & (zn > 0.05) & (zn < 0.45) & (y > 55)
        vol[sc & body] = 600

    # ---------------- pulmonary vessel trees ---------------------------------
    for side in (-1, 1):
        hilum = np.array([0.45 * zlen, 4, side * 38])
        roots = [[-1.0, -0.1, side * 0.6], [0.1, -0.4, side * 1.0], [0.2, 0.5, side * 0.9], [1.0, 0.2, side * 0.5]]
        stack = [(hilum, np.array(r0), 5.0, 0) for r0 in roots]
        while stack:
            p, d, r, depth = stack.pop()
            d = d / np.linalg.norm(d)
            q = p + d * 30 * (0.8 ** depth) * rng.uniform(0.8, 1.25)
            _paint_tube(vol, p, q, r, 50 + 10 * rng.normal(), spacing_a, where=lungs)
            if depth < 7 and r > 0.7:
                out = np.array([0, 0, side * 0.35])  # mild lateral bias, like real trees
                for _ in range(2):
                    stack.append((q, d + out + rng.normal(0, 0.6, 3), r * 0.74, depth + 1))

    # ---------------- nodule -------------------------------------------------
    nod_info = None
    if nodule:
        side = -1 if nodule_side == "right" else 1  # patient right -> image left
        lz = (0.15 + 0.6 * nodule_level) * zlen
        cands = np.argwhere(lungs[int(lz / sz)])
        cyx = cands[(cands[:, 1] - X / 2) * side > 0]
        # pick a peripheral, well-inside-lung location
        pick = cyx[rng.integers(len(cyx))] if len(cyx) else np.array([Y // 2, X // 2])
        c = np.array([lz, (pick[0] - Y / 2) * sy, (pick[1] - X / 2) * sx])
        R = nodule_mm / 2
        d = np.sqrt((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2)
        # lobulated surface
        ang = np.arctan2(y - c[1], x - c[2])
        lob = R * (1 + 0.12 * np.sin(3 * ang + 1.3) + 0.08 * np.sin(5 * ang))
        ggo_shell = d < lob * 1.6
        if nodule_type in ("part-solid", "ground-glass"):
            vol[ggo_shell & lungs] = np.maximum(vol[ggo_shell & lungs], -520 + 30 * tex[ggo_shell & lungs])
        if nodule_type in ("solid", "part-solid"):
            core = d < (lob if nodule_type == "solid" else lob * 0.55)
            vol[core] = 38 + 12 * tex[core]
        if spiculated:
            for _ in range(14):
                v = rng.normal(0, 1, 3)
                v[0] *= 0.6
                v /= np.linalg.norm(v)
                _paint_tube(vol, c, c + v * R * rng.uniform(1.6, 2.6), 0.9, 20, spacing_a, where=lungs)
        nod_info = {"center_mm": c.tolist(), "diameter_mm": nodule_mm, "type": nodule_type, "spiculated": spiculated,
                    "slice": int(round(c[0] / sz))}

    # ---------------- scanner noise (low dose) -------------------------------
    noise = rng.normal(0, 22, shape).astype(np.float32)
    vol[body] += noise[body]
    vol[~body] += noise[~body] * 0.3

    meta = {"source": "Synthetic phantom", "num_slices": Z, "nodule": nod_info, "seed": seed}
    return CTVolume(vol.astype(np.float32), tuple(spacing), meta)

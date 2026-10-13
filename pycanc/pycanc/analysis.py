"""
Quantitative scan analysis at native CT resolution.

    * nodule detection      multiscale Hessian "dot" filter (Li, Sone & Doi, Med Phys 2003)
    * nodule measurement    3D segmentation; long/short axis, mean diameter, volume, solid component,
                            type (solid / part-solid / ground-glass), calcification, sphericity,
                            solidity, margin (smooth / lobulated / spiculated, estimated), texture
    * Lung-RADS v2022       baseline category per nodule and per exam, with management
    * Brock / PanCan        nodule malignancy probability (McWilliams et al., NEJM 2013, full model)
    * emphysema             LAA-950 / LAA-910 %, Perc15, mean lung density, per lung and zone
    * CT physics            noise, air calibration, sharpness, coverage, inspiration, quality score
    * growth                volume doubling time between two scans of the same patient

All image measurements are automatic estimates from a research tool, not a radiologist read.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from .preprocess import (CTVolume, Prepared, body_mask, from_model_coords, lung_segmentation, mm_to_native, native_to_mm,
                         to_model_coords)

GRID_SPACING = (2.5, 1.40625, 1.40625)


# =========================================================================== #
#  helpers
# =========================================================================== #
def _ellipsoid(radius_mm: float, spacing) -> np.ndarray:
    r = [max(1, int(math.ceil(radius_mm / s))) for s in spacing]
    zz, yy, xx = np.ogrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    return (zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2 <= radius_mm ** 2


def _crop(center, radius_mm, spacing, shape):
    sl = []
    for c, s, n in zip(center, spacing, shape):
        r = int(math.ceil(radius_mm / s))
        sl.append(slice(max(0, c - r), min(n, c + r + 1)))
    return tuple(sl)


# =========================================================================== #
#  emphysema
# =========================================================================== #
def emphysema(hu: np.ndarray, lung: np.ndarray, spacing) -> dict:
    vox_ml = float(np.prod(spacing)) / 1000.0
    sm = ndi.gaussian_filter(hu.astype(np.float32), sigma=(0.0, 0.7, 0.7))  # light in-plane noise suppression
    zs = np.where(lung.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return {"available": False}
    z0, z1 = zs[0], zs[-1] + 1
    xs = np.where(lung.any(axis=(0, 1)))[0]
    xmid = (xs[0] + xs[-1]) / 2

    def stats(m):
        v = sm[m]
        if v.size < 50:
            return None
        return {"volume_l": round(v.size * vox_ml / 1000, 3),
                "laa950_pct": round(float((v < -950).mean() * 100), 2),
                "laa910_pct": round(float((v < -910).mean() * 100), 2),
                "perc15_hu": round(float(np.percentile(v, 15)), 1),
                "mean_hu": round(float(v.mean()), 1)}

    out = {"available": True, "whole": stats(lung), "zones": {}}
    thirds = np.linspace(z0, z1, 4).round().astype(int)
    xgrid = np.arange(hu.shape[2])[None, None, :]
    for side, sel in (("right", xgrid < xmid), ("left", xgrid >= xmid)):   # image left = patient right
        m_side = lung & sel
        out[side] = stats(m_side)
        for zi, name in enumerate(("upper", "middle", "lower")):
            m = np.zeros_like(lung)
            m[thirds[zi]:thirds[zi + 1]] = True
            out["zones"][f"{side}_{name}"] = stats(m_side & m)
    w = out["whole"]
    laa = w["laa950_pct"] if w else 0
    out["severity"] = "none / trace" if laa < 5 else "mild" if laa < 15 else "moderate" if laa < 25 else "severe"
    out["present"] = laa >= 5
    up = np.mean([out["zones"][f"{s}_upper"]["laa950_pct"] for s in ("right", "left") if out["zones"].get(f"{s}_upper")] or [0])
    lo = np.mean([out["zones"][f"{s}_lower"]["laa950_pct"] for s in ("right", "left") if out["zones"].get(f"{s}_lower")] or [0])
    out["distribution"] = ("upper-zone predominant (centrilobular pattern)" if up > 1.5 * lo + 1 else
                           "lower-zone predominant (panlobular pattern)" if lo > 1.5 * up + 1 else "diffuse / even")
    hist, edges = np.histogram(sm[lung], bins=66, range=(-1024, -496))
    out["histogram"] = {"counts": hist.tolist(), "edges": edges.tolist()}
    return out


# =========================================================================== #
#  CT physics / quality
# =========================================================================== #
def ct_physics(ct: CTVolume, hu: np.ndarray, lung: np.ndarray, body: np.ndarray) -> dict:
    sz, sy, sx = ct.spacing
    Z = hu.shape[0]
    zs = np.linspace(0, Z - 1, min(Z, 12)).round().astype(int)
    air_vals, air_noise, tis_noise, sharp = [], [], [], []
    for z in zs:
        sl = hu[z].astype(np.float32)
        hp = sl - ndi.gaussian_filter(sl, 2.0)
        outside = ~ndi.binary_dilation(body[z], iterations=6) & (sl < -800)
        if outside.sum() > 500:
            air_vals.append(np.median(sl[outside])); air_noise.append(np.std(hp[outside]) * 1.12)
        tissue = ndi.binary_erosion(body[z] & (sl > -20) & (sl < 100), iterations=3)
        if tissue.sum() > 500:
            tis_noise.append(np.std(hp[tissue]) * 1.12)
        lz = lung[z]
        if lz.sum() > 2000:
            ys, xs_ = np.where(lz)
            cy, cx = int(np.median(ys)), int(np.median(xs_))
            r = 24
            patch = sl[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            if patch.shape == (2 * r, 2 * r):
                f = np.abs(np.fft.fftshift(np.fft.fft2(patch - patch.mean()))) ** 2
                yy, xx = np.indices(f.shape) - r
                rad = np.sqrt(yy ** 2 + xx ** 2) / r
                sharp.append(float(f[(rad > 0.5) & (rad <= 1)].sum() / max(f[rad <= 1].sum(), 1e-9)))
    q = {}
    q["slice_spacing_mm"] = round(sz, 3)
    q["pixel_spacing_mm"] = [round(sy, 4), round(sx, 4)]
    q["matrix"] = list(hu.shape[1:])
    q["slices"] = int(Z)
    q["coverage_mm"] = round(Z * sz, 1)
    q["fov_mm"] = round(max(sy * hu.shape[1], sx * hu.shape[2]), 1)
    q["air_hu"] = round(float(np.median(air_vals)), 1) if air_vals else None
    q["noise_air_hu"] = round(float(np.median(air_noise)), 1) if air_noise else None
    q["noise_tissue_hu"] = round(float(np.median(tis_noise)), 1) if tis_noise else None
    q["sharpness_index"] = round(float(np.median(sharp)), 3) if sharp else None
    vox_l = sz * sy * sx / 1e6
    q["lung_volume_l"] = round(float(lung.sum() * vox_l), 2)
    lz = np.where(lung.any(axis=(1, 2)))[0]
    q["apex_included"] = bool(len(lz) and lz[0] > 0)
    q["base_included"] = bool(len(lz) and lz[-1] < Z - 1)

    flags, score = [], 100
    def flag(msg, pen, level="warn"):
        nonlocal score
        flags.append({"level": level, "msg": msg}); score -= pen
    if sz > 2.5 + 1e-3:
        flag(f"Slice spacing {sz:.2g} mm is above the 2.5 mm the model was trained on.", 15)
    if q["air_hu"] is not None and abs(q["air_hu"] + 1000) > 50:
        flag(f"Air reads {q['air_hu']:.0f} HU (expected about -1000): calibration or reconstruction issue.", 15)
    n = q["noise_tissue_hu"]
    if n is not None:
        q["noise_level"] = "low" if n < 20 else "typical low-dose" if n < 45 else "high"
        if n >= 45:
            flag(f"High image noise ({n:.0f} HU in soft tissue): emphysema % and small-nodule edges are less reliable.", 10)
    s = q["sharpness_index"]
    if s is not None:
        q["kernel_estimate"] = "smooth / soft-tissue" if s < 0.08 else "standard" if s < 0.2 else "sharp / lung"
    if not q["apex_included"]:
        flag("Lung apex appears cut off at the top of the scan.", 10)
    if not q["base_included"]:
        flag("Lung bases appear cut off at the bottom of the scan.", 10)
    if q["lung_volume_l"] < 3.0:
        flag(f"Lung volume {q['lung_volume_l']:.1f} L is low: possible incomplete inspiration or partial coverage.", 10)
    if q["fov_mm"] < 250:
        flag("Narrow field of view; peripheral lung may be outside the image.", 10)
    q["flags"] = flags
    q["quality_score"] = max(0, score)
    return q


# =========================================================================== #
#  nodule detection (model grid) + measurement (native grid)
# =========================================================================== #
def detect_candidates(hu_grid: np.ndarray, lung_grid: np.ndarray, max_candidates: int = 40) -> list[tuple]:
    """Multiscale Hessian dot filter inside the lungs. Returns [(z, y, x, scale_mm, response)] on the model grid."""
    hu = hu_grid.astype(np.float32)
    lung = ndi.binary_dilation(lung_grid > 0, iterations=2)
    zs, ys, xs = np.where(lung)
    if len(zs) == 0:
        return []
    m = 8
    box = tuple(slice(max(0, a.min() - m), b.max() + m + 1) for a, b in ((zs, zs), (ys, ys), (xs, xs)))
    sub, lsub = hu[box], lung[box]
    cand = lsub & (sub > -750)
    idx = np.nonzero(cand)
    if len(idx[0]) == 0:
        return []
    best = np.zeros(len(idx[0]), np.float32)
    best_scale = np.zeros(len(idx[0]), np.float32)
    sp = np.array(GRID_SPACING)
    for s_mm in (1.6, 2.4, 3.5, 5.0, 7.0):
        sig = s_mm / sp
        H = np.empty((len(idx[0]), 3, 3), np.float32)
        for i in range(3):
            for j in range(i, 3):
                order = [0, 0, 0]; order[i] += 1; order[j] += 1
                d = ndi.gaussian_filter(sub, sig, order=order)[idx] / (sp[i] * sp[j]) * s_mm ** 2
                H[:, i, j] = d; H[:, j, i] = d
        ev = np.linalg.eigvalsh(H)                                   # ascending
        l_abs = np.sort(np.abs(ev), axis=1)                          # |l3| <= |l2| <= |l1|
        allneg = (ev < 0).all(axis=1)
        resp = np.where(allneg, l_abs[:, 0] ** 2 / np.maximum(l_abs[:, 2], 1e-6), 0)
        upd = resp > best
        best[upd] = resp[upd]; best_scale[upd] = s_mm
    R = np.zeros(sub.shape, np.float32); R[idx] = best
    S = np.zeros(sub.shape, np.float32); S[idx] = best_scale
    peaks = (R == ndi.maximum_filter(R, size=(3, 5, 5))) & (R > 40)
    pz, py, px = np.nonzero(peaks)
    order = np.argsort(-R[pz, py, px])[:max_candidates]
    off = [b.start for b in box]
    return [(int(pz[k] + off[0]), int(py[k] + off[1]), int(px[k] + off[2]), float(S[pz[k], py[k], px[k]]),
             float(R[pz[k], py[k], px[k]])) for k in order]


def detect_subsolid(hu_grid: np.ndarray, lung_grid: np.ndarray, max_candidates: int = 20) -> list[tuple]:
    """Ground-glass / part-solid candidates: compact regions of -750..-300 HU after smoothing and opening."""
    sp = np.array(GRID_SPACING)
    lung = lung_grid > 0
    if not lung.any():
        return []
    sm = ndi.gaussian_filter(hu_grid.astype(np.float32), sigma=1.5 / sp)
    band = lung & (sm > -720) & (sm < -250)
    band = ndi.binary_opening(band, structure=_ellipsoid(2.5, tuple(sp)))
    lab, n = ndi.label(band)
    out = []
    for k, slc in enumerate(ndi.find_objects(lab), start=1):
        if slc is None:
            continue
        m = lab[slc] == k
        vol = m.sum() * float(np.prod(sp))
        if vol < 4 / 3 * math.pi * 3.0 ** 3 or vol > 4 / 3 * math.pi * 25 ** 3:
            continue
        c = np.argwhere(m).mean(0) + [s.start for s in slc]
        out.append((int(round(c[0])), int(round(c[1])), int(round(c[2])), float((3 * vol / (4 * math.pi)) ** (1 / 3)), float(vol)))
    out.sort(key=lambda t: -t[4])
    return out[:max_candidates]


def _axes_2d(mask2d: np.ndarray, sy: float, sx: float) -> tuple[float, float]:
    """Long axis and perpendicular short axis (mm) of a 2D mask, Lung-RADS style."""
    ys, xs = np.nonzero(mask2d)
    if len(ys) == 0:
        return 0.0, 0.0
    pts = np.column_stack([ys * sy, xs * sx])
    # pixel corners so tiny nodules are not measured as zero
    pts = np.concatenate([pts + [dy, dx] for dy in (-sy / 2, sy / 2) for dx in (-sx / 2, sx / 2)])
    try:
        from scipy.spatial import ConvexHull
        pts = pts[ConvexHull(pts).vertices]
    except Exception:
        pass
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    long_ = float(d[i, j])
    u = (pts[j] - pts[i]) / max(long_, 1e-9)
    perp = np.array([-u[1], u[0]])
    proj = pts @ perp
    return long_, float(proj.max() - proj.min())


def _surface_metrics(mask: np.ndarray, spacing) -> tuple[float, float, float]:
    """(mesh volume mm3, surface area mm2, convex-hull volume mm3)."""
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=spacing)
        area = float(mesh_surface_area(verts, faces))
        tri = verts[faces]
        vol = float(abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6)
        from scipy.spatial import ConvexHull
        hull = float(ConvexHull(verts).volume)
        return vol, area, hull
    except Exception:
        v = mask.sum() * float(np.prod(spacing))
        faces = sum(np.count_nonzero(np.diff(np.pad(mask, 1).astype(np.int8), axis=a)) for a in range(3))
        return v, faces * float(np.mean(spacing)) ** 2, v


def measure_nodule(ct: CTVolume, lung: np.ndarray, center_native, scale_mm: float = 4.0) -> dict | None:
    hu, sp = ct.hu, ct.spacing
    roi = _crop(center_native, 32, sp, hu.shape)
    h = hu[roi].astype(np.float32)
    lroi = ndi.binary_dilation(lung[roi], structure=_ellipsoid(4, sp))
    c = [cn - r.start for cn, r in zip(center_native, roi)]
    # snap the seed to the densest voxel within 3 mm
    near = _crop(c, 3, sp, h.shape)
    hs = ndi.gaussian_filter(h, 0.6)
    k = np.unravel_index(np.argmax(hs[near]), hs[near].shape)
    seed = tuple(int(a.start + b) for a, b in zip(near, k))
    if h[seed] < -750:
        return None
    m = (h > -750) & lroi
    lab, _ = ndi.label(m)
    comp = lab == lab[seed]
    nod = None
    # detach vessels / chest wall: smallest opening that leaves an isolated blob around the seed
    for r_open in (1.2, 1.8, 2.5, 3.5, 4.5, 6.0):
        if r_open > max(2.0, scale_mm * 1.6):
            break
        opened = ndi.binary_opening(comp, structure=_ellipsoid(r_open, sp))
        lab2, n2 = ndi.label(opened)
        if n2 == 0:
            break
        if lab2[seed]:
            core = lab2 == lab2[seed]
        else:
            idx = ndi.distance_transform_edt(lab2 == 0, sampling=sp, return_distances=True, return_indices=True)
            if idx[0][seed] > 3:
                continue
            core = lab2 == lab2[tuple(d[seed] for d in idx[1])]
        cand = ndi.binary_dilation(core, structure=_ellipsoid(r_open + 0.5, sp)) & comp
        if any(cand.take(0, a).any() or cand.take(-1, a).any() for a in range(3)):
            continue
        nod = cand
        break
    if nod is None:
        return None
    for z in range(nod.shape[0]):
        nod[z] = ndi.binary_fill_holes(nod[z])
    voxels = int(nod.sum())
    if voxels < 4:
        return None
    lung_frac = float(lung[roi][nod].mean())
    # vessels leaving the blob: thick (> ~2 mm) branches crossing a 2-6 mm shell around it
    shell = ndi.binary_dilation(nod, structure=_ellipsoid(6, sp)) & ~ndi.binary_dilation(nod, structure=_ellipsoid(2, sp))
    thick = ndi.binary_opening(comp & ~nod, structure=_ellipsoid(1.0, sp)) & shell
    branches = int(ndi.label(thick)[1])

    vox = float(np.prod(sp))
    vals = h[nod]
    # size is measured on the nodule body: thin spicules / attached vessel stubs removed
    eq_full = (6 * voxels * vox / math.pi) ** (1 / 3)
    body = ndi.binary_opening(nod, structure=_ellipsoid(1.6, sp)) if eq_full >= 6 else nod
    if body.sum() >= 4:
        lb, nb = ndi.label(body)
        body = lb == (np.argmax(ndi.sum(body, lb, index=np.arange(1, nb + 1))) + 1)
    else:
        body = nod
    solid = body & (h > -300)
    if solid.any() and solid.sum() < 0.6 * body.sum():
        # subsolid nodule: the solid component is the compact solid blob nearest the centre;
        # vessels crossing the ground glass are thin and off-centre, so they do not count
        solid = ndi.binary_opening(solid, structure=_ellipsoid(1.5, sp))
        sl, sn = ndi.label(solid)
        if sn:
            bc = np.argwhere(body).mean(0)
            r_b = (3 * body.sum() * vox / (4 * math.pi)) ** (1 / 3)
            best, bd = 0, 1e9
            for k in range(1, sn + 1):
                d = np.linalg.norm((np.argwhere(sl == k).mean(0) - bc) * np.array(sp))
                if d < bd:
                    best, bd = k, d
            solid = (sl == best) if bd < 0.5 * r_b else np.zeros_like(solid)
    vol_v = float(body.sum()) * vox
    mesh_v, area, hull_v = _surface_metrics(nod, sp)
    zs = np.where(body.any(axis=(1, 2)))[0]
    zmax = zs[np.argmax([body[z].sum() for z in zs])]
    long_, short = _axes_2d(body[zmax], sp[1], sp[2])
    mean_d = (long_ + short) / 2
    solid_mean = 0.0
    if solid.any():
        zsol = np.where(solid.any(axis=(1, 2)))[0]
        zs_max = zsol[np.argmax([solid[z].sum() for z in zsol])]
        sl, ss = _axes_2d(solid[zs_max], sp[1], sp[2])
        solid_mean = (sl + ss) / 2
    solid_frac = float(solid.sum() / max(body.sum(), 1))
    calc_frac = float((vals > 200).mean())
    if calc_frac > 0.1:
        ntype = "calcified"
    elif solid_frac >= 0.55:
        ntype = "solid"
    elif solid_mean < 1.5 or solid_frac < 0.08:
        ntype = "ground-glass"
    else:
        ntype = "part-solid"
    eq_d = (6 * vol_v / math.pi) ** (1 / 3)
    sphericity = float(math.pi ** (1 / 3) * (6 * mesh_v) ** (2 / 3) / max(area, 1e-6))
    solidity = float(mesh_v / max(hull_v, 1e-6))
    # spicules: thin strands radiating from the body. Look in a halo up to ~0.8 R (max 10 mm) around it,
    # keep what is connected to the body, and count separate strands beyond a 1.5 mm rim.
    r_body = (3 * max(body.sum(), 1) * vox / (4 * math.pi)) ** (1 / 3)
    halo = comp & ndi.binary_dilation(body, structure=_ellipsoid(min(max(0.8 * r_body, 3.0), 10.0), sp))
    hl, _ = ndi.label(halo | body)
    halo = hl == hl[tuple(np.argwhere(body)[0])]
    plab, pn = ndi.label(halo & ~ndi.binary_dilation(body, structure=_ellipsoid(1.5, sp)))
    spicules = 0
    for k, slc in enumerate(ndi.find_objects(plab), start=1):
        if slc is None:
            continue
        pts = np.argwhere(plab[slc] == k) * np.array(sp)
        if len(pts) >= 3 and np.ptp(pts, axis=0).max() >= 2.5:
            spicules += 1
    b_v, _, b_hull = _surface_metrics(body, sp)
    body_solidity = float(b_v / max(b_hull, 1e-6))
    if mean_d >= 5 and (spicules >= 5 or solidity < 0.72):
        margin = "spiculated (estimated)"
    elif mean_d >= 5 and body_solidity < 0.86:
        margin = "lobulated (estimated)"
    else:
        margin = "smooth"
    coords = np.argwhere(body)
    if len(coords) >= 4:
        cov = np.cov((coords * np.array(sp)).T) + np.diag(np.array(sp) ** 2 / 12)   # include voxel extent
        ev = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 1e-9))
        elong = float(ev[-1] / ev[0])
    else:
        elong = 1.0
    cen = coords.mean(0) + [r.start for r in roi]
    hist, _ = np.histogram(vals, bins=32, range=(-1000, 200))
    p = hist[hist > 0] / hist.sum()
    return {
        "center_native": [float(v) for v in cen],
        "center_mm": [float(v) for v in native_to_mm(ct, *cen)],
        "volume_mm3": round(vol_v, 1), "volume_with_spicules_mm3": round(voxels * vox, 1),
        "equivalent_diameter_mm": round(eq_d, 1),
        "long_axis_mm": round(long_, 1), "short_axis_mm": round(short, 1), "mean_diameter_mm": round(mean_d, 1),
        "solid_mean_diameter_mm": round(solid_mean, 1), "solid_fraction": round(solid_frac, 2),
        "type": ntype, "calcified_fraction": round(calc_frac, 3),
        "mean_hu": round(float(vals.mean()), 1), "std_hu": round(float(vals.std()), 1),
        "max_hu": round(float(vals.max()), 1), "entropy_bits": round(float(-(p * np.log2(p)).sum()), 2),
        "sphericity": round(sphericity, 2), "solidity": round(body_solidity, 2), "outline_solidity": round(solidity, 2), "spicules": spicules, "elongation": round(elong, 2),
        "margin": margin, "surface_area_mm2": round(area, 1),
        "lung_fraction": round(lung_frac, 2), "attached_vessels": branches,
        "_mask_roi": (nod, roi),
    }


def lung_rads(n: dict) -> str:
    """Lung-RADS v2022 baseline category for one nodule (size-based; growth handled separately)."""
    d, t = n["mean_diameter_mm"], n["type"]
    if t == "calcified":
        return "1"
    if t == "solid":
        cat = "2" if d < 6 else "3" if d < 8 else "4A" if d < 15 else "4B"
    elif t == "part-solid":
        s = n["solid_mean_diameter_mm"]
        cat = "2" if d < 6 else "3" if s < 6 else "4A" if s < 8 else "4B"
    else:
        cat = "2" if d < 30 else "3"
    if cat in ("3", "4A", "4B") and n["margin"].startswith("spiculated"):
        cat = "4X"
    return cat


RADS_ORDER = ["1", "2", "3", "4A", "4B", "4X"]
RADS_MANAGEMENT = {
    "1": "Negative. Continue annual screening with LDCT in 12 months.",
    "2": "Benign appearance or behaviour. Continue annual screening with LDCT in 12 months.",
    "3": "Probably benign. 6-month LDCT.",
    "4A": "Suspicious. 3-month LDCT; PET/CT may be used when there is a solid component of 8 mm or more.",
    "4B": "Very suspicious. Diagnostic chest CT with or without contrast, PET/CT and/or tissue sampling.",
    "4X": "Category 3 or 4 nodule with additional features that increase suspicion. Diagnostic CT, PET/CT and/or tissue sampling.",
}


def brock(n: dict, patient: dict, count: int, emphysema_present: bool) -> dict:
    """PanCan / Brock full model (with spiculation): probability that this nodule is malignant."""
    size = max(n["long_axis_mm"], 1.0)
    age = float(patient.get("age") or 62)
    female = str(patient.get("sex", "")).lower() == "female"
    fam = bool(patient.get("family_lung_cancer"))
    ntype = {"ground-glass": -0.1276, "part-solid": 0.377}.get(n["type"], 0.0)
    logit = (-6.7892 + 0.0287 * (age - 62) + 0.6011 * female + 0.2961 * fam + 0.2953 * emphysema_present
             - 5.3854 * ((size / 10) ** -0.5 - 1.58113883) + ntype + 0.6581 * (n["zone"] == "upper")
             - 0.0824 * (count - 4) + 0.7729 * n["margin"].startswith("spiculated"))
    assumed = [k for k, v in (("age", patient.get("age")), ("sex", patient.get("sex")),
                              ("family history", "family_lung_cancer" in patient)) if v in (None, "", "unspecified")]
    return {"probability": 1 / (1 + math.exp(-logit)), "assumed_defaults": assumed}


# =========================================================================== #
#  main entry point
# =========================================================================== #
def analyze(ct: CTVolume, prep: Prepared, progress=lambda m: None) -> dict:
    hu = ct.hu
    progress("Segmenting lungs at native resolution")
    lung = lung_segmentation(hu, ct.spacing)
    body = body_mask(hu)
    progress("Quantifying emphysema")
    emph = emphysema(hu, lung, ct.spacing)
    progress("Measuring CT physics and image quality")
    qc = ct_physics(ct, hu, lung, body)

    progress("Detecting nodules (multiscale Hessian)")
    hu_grid = prep.hu_grid if prep.hu_grid is not None else (-1000 + prep.wide.astype(np.float32) * 2500 / 255)
    cands = detect_candidates(hu_grid, prep.lung_mask) + detect_subsolid(hu_grid, prep.lung_mask)
    progress(f"Measuring {len(cands)} candidates")
    lz = np.where(lung.any(axis=(1, 2)))[0]
    lz0, lz1 = (lz[0], lz[-1]) if len(lz) else (0, hu.shape[0] - 1)
    xs = np.where(lung.any(axis=(0, 1)))[0]
    xmid = (xs[0] + xs[-1]) / 2 if len(xs) else hu.shape[2] / 2
    nodules = []
    for (z, y, x, s_mm, resp) in cands:
        cn = mm_to_native(ct, *from_model_coords(ct, z, y, x))
        if not lung[cn] and not ndi.binary_dilation(lung[_crop(cn, 4, ct.spacing, lung.shape)]).any():
            continue
        if any(np.linalg.norm((np.array(cn) - n["center_native"]) * ct.spacing) < 5 for n in nodules):
            continue
        n = measure_nodule(ct, lung, cn, s_mm)
        # vessel / hilum rejection: tubular, or a junction of several thick branches
        if (n is None or not (3.0 <= n["mean_diameter_mm"] <= 50) or n["elongation"] > 1.6
                or n["lung_fraction"] < 0.5 or n["attached_vessels"] >= 6
                or (n["attached_vessels"] >= 2 and n["elongation"] > 1.3 and not n["margin"].startswith("spic"))):
            continue
        if any(np.linalg.norm((np.array(n["center_native"]) - m["center_native"]) * ct.spacing) < 4 for m in nodules):
            continue
        n["detector_response"] = round(resp, 1)
        nodules.append(n)

    for n in nodules:
        cz, cy, cx = (int(round(v)) for v in n["center_native"])
        n["side"] = "right" if cx < xmid else "left"
        frac = (cz - lz0) / max(lz1 - lz0, 1)
        n["zone"] = "upper" if frac < 1 / 3 else "middle" if frac < 2 / 3 else "lower"
        roi = _crop((cz, cy, cx), 35, ct.spacing, lung.shape)
        d = ndi.distance_transform_edt(lung[roi], sampling=ct.spacing)[cz - roi[0].start, cy - roi[1].start, cx - roi[2].start]
        n["pleural_distance_mm"] = round(float(d), 1)
        n["juxtapleural"] = bool(d < 3)
        n["model_zyx"] = list(to_model_coords(ct, *n["center_mm"]))
        n["lung_rads"] = lung_rads(n)
        n.pop("_mask_roi", None)
    nodules.sort(key=lambda n: (-RADS_ORDER.index(n["lung_rads"]), -n["mean_diameter_mm"]))
    for i, n in enumerate(nodules):
        n["id"] = f"N{i + 1}"
    exam = max((n["lung_rads"] for n in nodules), key=RADS_ORDER.index, default="1")
    lungc = np.argwhere(lung).mean(0) if lung.any() else np.array(hu.shape) / 2
    progress("Done")
    return {
        "nodules": nodules,
        "lung_rads": {"category": exam, "management": RADS_MANAGEMENT[exam],
                      "note": "Automatic baseline category from size, type and margin; prior scans, "
                              "perifissural/benign features and the radiologist's judgement can change it."},
        "emphysema": emph,
        "ct_physics": qc,
        "lung_centroid_mm": [float(v) for v in native_to_mm(ct, *lungc)],
    }


def decorate(analysis: dict, patient: dict | None = None, hotspots: list[dict] | None = None) -> dict:
    """Add Brock probabilities (needs patient data) and link nodules to the model's attention hotspots."""
    out = {**analysis, "nodules": [dict(n) for n in analysis["nodules"]]}
    patient = patient or {}
    emph_present = bool(analysis["emphysema"].get("present"))
    for n in out["nodules"]:
        if n["type"] != "calcified":
            n["brock"] = brock(n, patient, len(out["nodules"]), emph_present)
        if hotspots:
            best = None
            for k, h in enumerate(hotspots):
                d = math.dist([(h["z"] - n["model_zyx"][0]) * 2.5, (h["y"] - n["model_zyx"][1]) * 1.40625,
                               (h["x"] - n["model_zyx"][2]) * 1.40625], [0, 0, 0])
                if d < 20 and (best is None or h["score"] > best[1]):
                    best = (k + 1, h["score"])
            n["attention"] = {"hotspot": best[0], "score": round(best[1], 3)} if best else None
    return out


# =========================================================================== #
#  growth between two scans
# =========================================================================== #
def compare(prior: dict, current: dict, interval_days: float) -> dict:
    shift = np.array(current["lung_centroid_mm"]) - np.array(prior["lung_centroid_mm"])
    rows, used = [], set()
    for n in current["nodules"]:
        best, bd = None, 15.0
        for j, p in enumerate(prior["nodules"]):
            d = float(np.linalg.norm(np.array(n["center_mm"]) - (np.array(p["center_mm"]) + shift)))
            if d < bd and j not in used:
                best, bd = j, d
        row = {"id": n["id"], "current_volume_mm3": n["volume_mm3"], "current_mean_diameter_mm": n["mean_diameter_mm"]}
        if best is None:
            row.update(status="new", note="Not seen on the prior scan.")
        else:
            used.add(best)
            p = prior["nodules"][best]
            v1, v2 = p["volume_mm3"], n["volume_mm3"]
            row.update(prior_id=p["id"], prior_volume_mm3=v1, prior_mean_diameter_mm=p["mean_diameter_mm"],
                       match_distance_mm=round(bd, 1), volume_change_pct=round((v2 / v1 - 1) * 100, 1),
                       diameter_change_mm=round(n["mean_diameter_mm"] - p["mean_diameter_mm"], 1))
            if v2 > v1 * 1.05:
                vdt = interval_days * math.log(2) / math.log(v2 / v1)
                row["vdt_days"] = round(vdt)
                row["status"] = ("fast growth (VDT < 400 d, suspicious)" if vdt < 400 else
                                 "indeterminate growth (VDT 400-600 d)" if vdt <= 600 else "slow growth (VDT > 600 d)")
            else:
                row["vdt_days"] = None
                row["status"] = "stable or shrinking"
            if interval_days <= 365 and row["diameter_change_mm"] > 1.5:
                row["lung_rads_growth"] = "Growing by Lung-RADS (> 1.5 mm in 12 months)"
        rows.append(row)
    resolved = [p["id"] for j, p in enumerate(prior["nodules"]) if j not in used]
    return {"interval_days": interval_days, "nodules": rows, "resolved_prior_nodules": resolved}

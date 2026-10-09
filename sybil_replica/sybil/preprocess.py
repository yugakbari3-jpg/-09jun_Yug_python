"""
CT loading + the exact Sybil preprocessing chain.

    HU volume
      -> lung window (center -600, width 1500) to 16 bit, then //256 -> 0..255
      -> per-slice resize to 256 x 256                      ("scale_2d")
      -> 3 channels, (x - 128.1722) / 87.1849               ("force_num_chan_2d", "normalize_2d")
      -> trilinear resample to 0.703125 x 0.703125 x 2.5 mm ("resample_pixel_spacing")
      -> centre crop / zero pad to 200 x 256 x 256          (torchio CropOrPad)

Slices are ordered superior -> inferior, like the reference loader.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMG_SIZE = 256
NUM_IMAGES = 200
IMG_MEAN = 128.1722
IMG_STD = 87.1849
WINDOW_CENTER = -600
WINDOW_WIDTH = 1500
TARGET_SPACING = (2.5, 0.703125, 0.703125)  # z, y, x  (mm, at a nominal 512 matrix)


@dataclass
class CTVolume:
    hu: np.ndarray                      # (Z, Y, X) float32 Hounsfield units, superior -> inferior
    spacing: tuple[float, float, float]  # (z, y, x) mm
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Readers
# --------------------------------------------------------------------------- #
def _read_dicom_datasets(datasets) -> CTVolume:
    import pydicom  # noqa: F401  (imported for side effects / clear error)
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    slices = [d for d in datasets if hasattr(d, "PixelData") and hasattr(d, "ImagePositionPatient")]
    if not slices:
        raise ValueError("No CT image slices with ImagePositionPatient found.")
    # keep the largest series only
    by_series: dict[str, list] = {}
    for d in slices:
        by_series.setdefault(getattr(d, "SeriesInstanceUID", "0"), []).append(d)
    slices = max(by_series.values(), key=len)
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]), reverse=True)

    hu = np.stack([apply_modality_lut(d.pixel_array, d).astype(np.float32) for d in slices])
    zs = [float(d.ImagePositionPatient[2]) for d in slices]
    dz = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 else float(getattr(slices[0], "SliceThickness", 2.5))
    ps = [float(v) for v in getattr(slices[0], "PixelSpacing", [0.703125, 0.703125])]
    d0 = slices[0]
    meta = {
        "source": "DICOM",
        "num_slices": len(slices),
        "manufacturer": str(getattr(d0, "Manufacturer", "")),
        "study_date": str(getattr(d0, "StudyDate", "")),
        "series_description": str(getattr(d0, "SeriesDescription", "")),
        "kvp": str(getattr(d0, "KVP", "")),
    }
    return CTVolume(hu, (dz or 2.5, ps[0], ps[1]), meta)


def load_dicom_dir(path: Path) -> CTVolume:
    import pydicom

    files = [p for p in Path(path).rglob("*") if p.is_file()]
    ds = []
    for p in files:
        try:
            ds.append(pydicom.dcmread(p, force=True))
        except Exception:
            continue
    return _read_dicom_datasets(ds)


def load_dicom_zip(data: bytes) -> CTVolume:
    import pydicom

    ds = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.endswith("/") or "__MACOSX" in name:
                continue
            try:
                ds.append(pydicom.dcmread(io.BytesIO(z.read(name)), force=True))
            except Exception:
                continue
    return _read_dicom_datasets(ds)


def load_nifti(path: Path) -> CTVolume:
    import nibabel as nib  # optional dependency

    img = nib.as_closest_canonical(nib.load(str(path)))  # RAS
    arr = np.asarray(img.dataobj, dtype=np.float32)     # (X, Y, Z)
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    hu = np.transpose(arr, (2, 1, 0))[::-1, ::-1, :]     # -> (Z sup->inf, Y ant->post, X)
    return CTVolume(np.ascontiguousarray(hu), (sz, sy, sx), {"source": "NIfTI"})


def load_npz(data: bytes) -> CTVolume:
    """.npz with `hu` (Z,Y,X) and optional `spacing` (z,y,x)."""
    f = np.load(io.BytesIO(data), allow_pickle=False)
    hu = f["hu"].astype(np.float32)
    spacing = tuple(float(v) for v in f["spacing"]) if "spacing" in f else (2.5, 0.703125, 0.703125)
    return CTVolume(hu, spacing, {"source": "NPZ"})


def load_any(path_or_bytes, filename: str = "") -> CTVolume:
    name = filename.lower()
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        if p.is_dir():
            return load_dicom_dir(p)
        name = name or p.name.lower()
        if name.endswith((".nii", ".nii.gz")):
            return load_nifti(p)
        path_or_bytes = p.read_bytes()
    if name.endswith(".zip"):
        return load_dicom_zip(path_or_bytes)
    if name.endswith(".npz"):
        return load_npz(path_or_bytes)
    if name.endswith((".nii", ".nii.gz")):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".nii.gz" if name.endswith(".gz") else ".nii") as t:
            t.write(path_or_bytes)
            t.flush()
            return load_nifti(Path(t.name))
    raise ValueError(f"Unsupported file type: {filename or path_or_bytes!r}")


# --------------------------------------------------------------------------- #
#  Sybil preprocessing
# --------------------------------------------------------------------------- #
def to_model_coords(ct: CTVolume, z_mm: float, y_mm: float, x_mm: float) -> tuple[int, int, int]:
    """Map a physical point (z from first slice, y/x from image centre, mm) to
    voxel indices in the 200 x 256 x 256 model input."""
    Z, Y, X = ct.hu.shape
    sz, sy, sx = ct.spacing
    out = []
    for pos_mm, n_src, spacing, n_tgt, centred in [
        (z_mm, Z * sz / TARGET_SPACING[0], TARGET_SPACING[0], NUM_IMAGES, False),
        # in-plane, one model voxel spans 512 / 256 nominal pixels
        (y_mm, Y * sy / (TARGET_SPACING[1] * 512 / IMG_SIZE), TARGET_SPACING[1] * 512 / IMG_SIZE, IMG_SIZE, True),
        (x_mm, X * sx / (TARGET_SPACING[2] * 512 / IMG_SIZE), TARGET_SPACING[2] * 512 / IMG_SIZE, IMG_SIZE, True),
    ]:
        n = max(1, round(n_src))
        idx = pos_mm / spacing + (n / 2 if centred else 0)
        off = (n_tgt - n) // 2 if n < n_tgt else -((n - n_tgt) // 2)
        out.append(int(round(idx + off)))
    return tuple(out)


def apply_windowing(hu: np.ndarray, center=WINDOW_CENTER, width=WINDOW_WIDTH, bit_size=16) -> np.ndarray:
    y_max = 2 ** bit_size - 1
    c, w = center - 0.5, width - 1
    out = ((hu - c) / w + 0.5) * y_max
    out[hu <= c - w / 2] = 0
    out[hu > c + w / 2] = y_max
    return np.clip(out, 0, y_max).astype(np.uint16)


@dataclass
class Prepared:
    tensor: torch.Tensor    # (1, 3, 200, 256, 256) normalised model input
    display: np.ndarray     # (200, 256, 256) uint8, exactly what the model sees (lung window)
    wide: np.ndarray        # (200, 256, 256) uint8, HU -1000..1500 on the same grid (for 3D / other windows)
    lung_mask: np.ndarray   # (200, 256, 256) uint8 {0, 255}
    valid: tuple[int, int]  # first/last slice index that contains real data

WIDE_LO, WIDE_HI = -1000.0, 1500.0


def _to_model_grid(vol: np.ndarray, spacing, fill: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-slice resize to 256, trilinear resample to target spacing, centre crop/pad."""
    x = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[:, None]   # Z, 1, Y, X
    Z, (Y, X) = x.shape[0], x.shape[-2:]
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    x = x[:, 0][None, None]                                                    # 1, 1, Z, 256, 256

    # Spacing is expressed at a nominal 512 matrix in the reference model.
    sz, sy, sx = spacing
    tz, ty, tx = TARGET_SPACING
    new = (max(1, round(Z * sz / tz)),
           max(1, round(IMG_SIZE * sy * Y / 512.0 / ty)),
           max(1, round(IMG_SIZE * sx * X / 512.0 / tx)))
    x = F.interpolate(x, size=new, mode="trilinear", align_corners=False)[0, 0]

    out = torch.full((NUM_IMAGES, IMG_SIZE, IMG_SIZE), float(fill))
    zmask = torch.zeros(NUM_IMAGES, dtype=torch.bool)
    src, dst = [], []
    for n, t in zip(x.shape, (NUM_IMAGES, IMG_SIZE, IMG_SIZE)):
        if n >= t:
            s = (n - t) // 2
            src.append(slice(s, s + t)); dst.append(slice(0, t))
        else:
            s = (t - n) // 2
            src.append(slice(0, n)); dst.append(slice(s, s + n))
    out[dst[0], dst[1], dst[2]] = x[src[0], src[1], src[2]]
    zmask[dst[0]] = True
    return out, zmask


def lung_segmentation(hu: np.ndarray) -> np.ndarray:
    """Rough lung mask: low-density components not touching the border, holes filled."""
    from scipy import ndimage as ndi

    air = hu < -400
    lab, n = ndi.label(air)
    if n == 0:
        return np.zeros_like(air)
    border = np.unique(np.concatenate([lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    sizes = ndi.sum(air, lab, index=np.arange(1, n + 1))
    sizes[border[border > 0] - 1] = 0
    keep = np.argsort(sizes)[::-1][:2]
    keep = keep[sizes[keep] > 0.05 * sizes.max()] + 1 if sizes.max() > 0 else []
    mask = np.isin(lab, keep)
    mask = ndi.binary_closing(mask, iterations=2)
    for z in range(mask.shape[0]):
        mask[z] = ndi.binary_fill_holes(mask[z])
    return mask


def preprocess(ct: CTVolume) -> Prepared:
    # model path: window -> 8 bit -> resize -> normalise -> resample -> crop/pad (pad value 0)
    win = (apply_windowing(ct.hu) // 256).astype(np.float32)
    x, zmask = _to_model_grid((win - IMG_MEAN) / IMG_STD, ct.spacing, fill=0.0)

    display = torch.clamp(x * IMG_STD + IMG_MEAN, 0, 255)
    display[~zmask] = 0

    hu, _ = _to_model_grid(ct.hu, ct.spacing, fill=-1000.0)
    hu = hu.numpy()
    wide = np.clip((hu - WIDE_LO) / (WIDE_HI - WIDE_LO) * 255, 0, 255).round().astype(np.uint8)
    lungs = lung_segmentation(hu)

    idx = torch.nonzero(zmask).flatten()
    return Prepared(
        tensor=x[None, None].expand(1, 3, -1, -1, -1).contiguous(),
        display=display.round().to(torch.uint8).numpy(),
        wide=wide,
        lung_mask=(lungs * 255).astype(np.uint8),
        valid=(int(idx[0]), int(idx[-1])),
    )

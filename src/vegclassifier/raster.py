"""
vegclassifier/raster.py
=======================
Full-image windowed inference.

Tiles exist for preprocessing and training only. Prediction and evaluation run
against the FULL-RESOLUTION source GeoTIFF and emit single rasters matching the
source extent, CRS, and transform exactly.

Two problems this solves that per-tile prediction does not:

1. Dropped tiles. Tiling used skip_empty / min_valid_fraction, so mosaicking the
   tiles on disk would leave holes. Reading the source avoids that entirely.

2. Seam artefacts. Entropy(R) and GLCM are neighbourhood operations; at a tile
   edge the neighbourhood is truncated and border pixels get wrong values. Here
   each window is read with a HALO of context, features are computed on the
   padded array, and only the valid interior is written.

GLCM phase alignment: `glcm_contrast_map` evaluates patches on a stride grid
starting at the array origin. If windows were placed arbitrarily, that grid
would shift between windows and produce faint discontinuities. Block size and
halo are therefore forced to multiples of the stride so every window's local
grid coincides with one global grid.
"""

from collections.abc import Iterator

import numpy as np
import rasterio as rio
from rasterio.windows import Window

from .features import build_feature_stack, compute_hsv
from .sampling import make_nonveg_mask

NODATA = 255

CLASS_NAMES: dict[int, str] = {0: "Background", 1: "Phragmites", 2: "FAV", 3: "Tule"}


def scope_class_names(scope: str, classes: list[int]) -> dict[int, str]:
    """Human-readable names for a scope's label space (merged folds Tule into BG)."""
    if scope == "merged":
        return {c: ("BG+Tule" if c == 0 else CLASS_NAMES[c]) for c in classes}
    return {c: CLASS_NAMES[c] for c in classes}


def glcm_stride(glcm_patch_size: int) -> int:
    """Stride used inside glcm_contrast_map — mirrors that implementation."""
    return max(1, (glcm_patch_size // 2) // 2)


def compute_halo(feature_params: dict, erode_px: int = 0) -> int:
    """
    Context margin needed so windowed features match whole-image features.

    Driven by the largest neighbourhood: entropy radius, GLCM patch + smoothing,
    and (for eval) the erosion structuring element. Rounded up to a stride
    multiple to keep the GLCM grid phase-aligned.
    """
    gp     = feature_params["glcm_patch_size"]
    stride = glcm_stride(gp)
    need   = max(feature_params["entropy_r"], gp // 2 + stride, erode_px) + gp
    return int(np.ceil(need / stride) * stride)


def align_block(block: int, feature_params: dict) -> int:
    """
    Round a requested block size up to a multiple of the GLCM stride.

    Window origins fall on multiples of the block size. If the block is not a
    stride multiple, the GLCM grid shifts between windows and output changes
    with block size. Rounding up keeps every window on one global grid.
    """
    stride = glcm_stride(feature_params["glcm_patch_size"])
    return int(np.ceil(block / stride) * stride)


def iter_windows(H: int, W: int, block: int, halo: int
                 ) -> Iterator[tuple[Window, Window, tuple[slice, slice]]]:
    """
    Yield (write_window, read_window, crop) covering the raster.

    read_window is write_window grown by `halo` (clipped at edges); `crop` slices
    the computed arrays back down to the write region.
    """
    for row0 in range(0, H, block):
        h = min(block, H - row0)
        r0 = max(0, row0 - halo)
        r1 = min(H, row0 + h + halo)
        for col0 in range(0, W, block):
            w = min(block, W - col0)
            c0 = max(0, col0 - halo)
            c1 = min(W, col0 + w + halo)
            yield (
                Window(col0, row0, w, h),
                Window(c0, r0, c1 - c0, r1 - r0),
                (slice(row0 - r0, row0 - r0 + h), slice(col0 - c0, col0 - c0 + w)),
            )


def normalise_block(arr_u8: np.ndarray, lo, hi) -> tuple[np.ndarray, ...]:
    """
    Apply the model's normalisation to a (3, h, w) uint8 block -> r, g, b floats.

    Mirrors io.load_tile semantics: explicit lo/hi is a fixed linear stretch
    (identity = 0/255); lo/hi None falls back to a per-block 2-98 percentile
    stretch. Per-block is inferior (02_B) and only reachable if the model was
    trained with normalization.mode = per_tile.
    """
    a = arr_u8[:3].astype(np.float32)
    if lo is None or hi is None:
        lo_ = float(np.percentile(a, 2))
        hi_ = float(np.percentile(a, 98))
    else:
        lo_, hi_ = float(lo), float(hi)
    a = np.clip((a - lo_) / (hi_ - lo_ + 1e-9), 0.0, 1.0)
    return a[0], a[1], a[2]


def image_valid_mask(src, block_arr: np.ndarray, window: Window) -> np.ndarray:
    """
    Validity for imagery WITHOUT a mask file (the predict path).

    Uses the dataset nodata value if declared; otherwise treats all-zero pixels
    across bands as nodata, which is how the orthomosaic pads its extent.
    """
    if src.nodata is not None:
        return ~np.all(block_arr[:3] == src.nodata, axis=0)
    return ~np.all(block_arr[:3] == 0, axis=0)


def open_like(path, src, count: int, dtype: str, nodata=None):
    """Create an output raster matching src geometry/CRS, LZW-compressed, tiled."""
    return rio.open(
        path, "w", driver="GTiff",
        height=src.height, width=src.width, count=count, dtype=dtype,
        crs=src.crs, transform=src.transform,
        compress="lzw", tiled=True, blockxsize=512, blockysize=512,
        nodata=nodata,
    )


def block_features(rgb_block: np.ndarray, norm: dict, feature_params: dict):
    """
    Normalise a (3, h, w) uint8 block and build the 4-feature stack.

    Returns (X_flat, nonveg, val, sat) for the padded block.
    """
    r, g, b = normalise_block(rgb_block, norm.get("lo"), norm.get("hi"))
    X_flat, _ = build_feature_stack(r, g, b, feature_params)
    _, sat, val = compute_hsv(r, g, b)
    nonveg = make_nonveg_mask(
        val, sat,
        feature_params.get("nonveg_val_thresh", 0.35),
        feature_params.get("nonveg_sat_thresh", 0.35),
    )
    return X_flat, nonveg, val, sat

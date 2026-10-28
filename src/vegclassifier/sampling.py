"""
vegclassifier/sampling.py
=========================
Non-veg masking, boundary erosion, scope label-mapping, and per-tile training
sample collection.

`make_nonveg_mask` and `erode_mask` are ported verbatim from the legacy utils
package. `SCOPES`, `remap_mask_for_scope`, and `collect_tile_samples` reproduce
the scope-partitioning logic from 03_B (cells 1, 7, 9/13).
"""


import numpy as np
from scipy.ndimage import binary_erosion

ImgArray  = np.ndarray
MaskArray = np.ndarray
BoolArray = np.ndarray

BACKGROUND, PHRAGMITES, FAV, TULE, NODATA = 0, 1, 2, 3, 255
ALL_CLASSES = [BACKGROUND, PHRAGMITES, FAV, TULE]

# Classification scopes — locked in 03_B cell 1.
# label_map: original class (0..3) -> new label, or -1 to exclude from the scope.
SCOPES: dict[str, dict] = {
    "binary": {
        "classes":     [0, 1],
        "label_map":   {0: 0, 1: 1, 2: -1, 3: -1},   # exclude FAV + Tule
        "description": "Phragmites vs Background only",
    },
    "multiclass": {
        "classes":     [0, 1, 2, 3],
        "label_map":   {0: 0, 1: 1, 2: 2, 3: 3},
        "description": "BG, Phragmites, FAV, Tule (full 4-class)",
    },
    "merged": {
        "classes":     [0, 1, 2],
        "label_map":   {0: 0, 1: 1, 2: 2, 3: 0},     # Tule merged into BG
        "description": "BG+Tule (merged), Phragmites, FAV",
    },
}


def make_nonveg_mask(val: ImgArray, sat: ImgArray,
                     val_thresh: float = 0.35, sat_thresh: float = 0.35) -> BoolArray:
    """
    True where a pixel is non-vegetated (water, mud, bare soil) and should be
    excluded. Default thresholds (Val<0.35 AND Sat<0.35) from 02_B. Ported.
    """
    return (val < val_thresh) & (sat < sat_thresh)


def erode_mask(mask: MaskArray, erode_px: int) -> MaskArray:
    """
    Erode each class inward by erode_px, setting the border zone to 255 (ignore).
    Removes ambiguous hand-drawn-polygon edge pixels. Ported verbatim.
    """
    if erode_px <= 0:
        return mask.copy()
    struct = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), dtype=bool)
    eroded = np.full(mask.shape, NODATA, dtype=np.uint8)
    for cls in ALL_CLASSES:
        eroded[binary_erosion(mask == cls, structure=struct)] = cls
    return eroded


def remap_mask_for_scope(mask_int: MaskArray, scope: str) -> MaskArray:
    """
    Apply a scope's label_map to a mask. Excluded classes (-1) and nodata
    become 255. From 03_B remap_mask_for_scope.
    """
    label_map = SCOPES[scope]["label_map"]
    out = np.full_like(mask_int, NODATA)
    for orig_cls, new_label in label_map.items():
        if new_label < 0:
            continue
        out[mask_int == orig_cls] = new_label
    out[mask_int == NODATA] = NODATA
    return out


def collect_tile_samples(X_flat: np.ndarray, nonveg: BoolArray,
                         mask_eroded: MaskArray, scope: str,
                         max_px_per_tile: int, rng) -> dict[int, np.ndarray]:
    """
    Sample up to max_px_per_tile vegetated pixels per (remapped) class from one
    tile, for a given scope. Faithful to 03_B cell 7.

    Parameters
    ----------
    X_flat : (H*W, n_features) float32
    nonveg : (H, W) bool   — non-veg pixels to exclude
    mask_eroded : (H, W) int — eroded RAW class mask (labels 0..3, 255 ignore)
    scope : str
    max_px_per_tile : int
    rng : np.random.Generator

    Returns
    -------
    dict {new_label: (k, n_features) array} for labels present in this tile.
    """
    label_map = SCOPES[scope]["label_map"]
    veg  = (~nonveg).ravel()
    m    = mask_eroded.ravel()
    out: dict[int, list[np.ndarray]] = {}

    for orig_cls in ALL_CLASSES:
        new_label = label_map.get(orig_cls, -1)
        if new_label < 0:
            continue
        idx = np.where((m == orig_cls) & veg)[0]
        if len(idx) == 0:
            continue
        if len(idx) > max_px_per_tile:
            idx = rng.choice(idx, max_px_per_tile, replace=False)
        out.setdefault(new_label, []).append(X_flat[idx])

    # merged scope can map two orig classes (0 and 3) to label 0 -> concat
    return {lab: np.vstack(parts) for lab, parts in out.items()}

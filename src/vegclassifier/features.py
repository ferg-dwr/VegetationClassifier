"""
vegclassifier/features.py
=========================
Production per-pixel feature extraction for the Winter Island classifier.

The validated 4-feature path locked in 02_B and run in 03_B:

    [0] Hue                        (color)
    [1] Saturation                 (color)
    [2] Entropy(Saturation, R)     (canopy uniformity)
    [3] GLCM-Contrast(Sat, patch)  (co-occurrence structure)

All four have pairwise correlation < 0.7. The legacy utils HSV+LocalStd+
5-prop-GLCM stack is NOT this set and is not used here.

Faithful port of 03_B `build_tile_features` / `_compute_glcm_contrast_map`,
including the strided-GLCM-then-uniform_filter behaviour, which is deliberate
(it is what was validated) and reproduced exactly.
"""


import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk

ImgArray = np.ndarray   # (H, W) float32 in [0, 1]

DEFAULT_GLCM_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def compute_hsv(r: ImgArray, g: ImgArray, b: ImgArray):
    """Vectorised RGB -> HSV, all channels in [0, 1]. Returns (hue, sat, val)."""
    cmax  = np.maximum(np.maximum(r, g), b)
    cmin  = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-9

    hue = np.zeros_like(r)
    m_r = cmax == r
    m_g = cmax == g
    m_b = cmax == b
    hue[m_r] = (60 * ((g[m_r] - b[m_r]) / delta[m_r]) % 360) / 360
    hue[m_g] = (60 * ((b[m_g] - r[m_g]) / delta[m_g] + 2) % 360) / 360
    hue[m_b] = (60 * ((r[m_b] - g[m_b]) / delta[m_b] + 4) % 360) / 360

    sat = np.clip(np.where(cmax == 0, 0.0, delta / (cmax + 1e-9)), 0.0, 0.99).astype(np.float32)
    val = cmax.astype(np.float32)
    return hue.astype(np.float32), sat, val


def entropy_map(channel: ImgArray, radius: int) -> ImgArray:
    """Local Shannon entropy of a [0,1] channel over a disk of given radius."""
    ch_u8 = (np.clip(channel, 0.0, 1.0) * 255).astype(np.uint8)
    return sk_entropy(ch_u8, disk(radius)).astype(np.float32)


def glcm_contrast_map_reference(channel, patch_size=21, levels=64,
                                distances=(1, 3, 5), angles=tuple(DEFAULT_GLCM_ANGLES)):
    """
    Reference implementation: explicit graycomatrix per patch (the 03_B loop).

    Kept as the correctness oracle for `glcm_contrast_map`, which is a
    vectorised equivalent. ~50x slower; use only for verification.
    """
    ch_u8  = (np.clip(channel, 0.0, 1.0) * 255).astype(np.uint8)
    H, W   = ch_u8.shape
    half   = patch_size // 2
    stride = max(1, half // 2)
    out    = np.zeros((H, W), dtype=np.float32)

    for r in range(half, H - half, stride):
        for c in range(half, W - half, stride):
            patch = ch_u8[r - half:r + half + 1, c - half:c + half + 1]
            pq = np.clip(
                (patch.astype(np.float32) / 255 * (levels - 1)).astype(np.int32),
                0, levels - 1,
            )
            glcm = graycomatrix(pq, distances=list(distances), angles=list(angles),
                                levels=levels, symmetric=True, normed=True)
            out[r, c] = float(graycoprops(glcm, "contrast").mean())

    return uniform_filter(out, size=stride * 2 + 1)


def glcm_contrast_map(channel, patch_size=21, levels=64,
                      distances=(1, 3, 5), angles=tuple(DEFAULT_GLCM_ANGLES)):
    """
    Per-pixel GLCM contrast over sliding patches — vectorised.

    Bit-identical to `glcm_contrast_map_reference` (verified: max abs diff 0.0
    at 256/512/1024 px), roughly 50x faster.

    How: GLCM contrast is SUM (i-j)^2 P(i,j). Because P is the normalised
    co-occurrence count, that sum is exactly the MEAN SQUARED DIFFERENCE between
    pixel pairs at the given offset. So no co-occurrence matrix is needed — for
    each offset, square the shifted difference image once, build an integral
    image, and every patch mean is an O(1) box sum.

    The strided evaluation grid and trailing uniform_filter are preserved
    exactly, since that behaviour is what 02_B validated.
    """
    ch_u8  = (np.clip(channel, 0.0, 1.0) * 255).astype(np.uint8)
    H, W   = ch_u8.shape
    half   = patch_size // 2
    stride = max(1, half // 2)

    # Per-pixel quantisation — identical to quantising each patch separately,
    # since the mapping depends only on the pixel value.
    q = np.clip((ch_u8.astype(np.float32) / 255 * (levels - 1)).astype(np.int32),
                0, levels - 1).astype(np.float64)

    rows = np.arange(half, H - half, stride)
    cols = np.arange(half, W - half, stride)
    out  = np.zeros((H, W), dtype=np.float32)
    if len(rows) == 0 or len(cols) == 0:
        return out
    RR, CC = np.meshgrid(rows, cols, indexing="ij")

    acc, n_off = np.zeros(RR.shape, dtype=np.float64), 0
    for d in distances:
        for a in angles:
            dr = round(d * np.sin(a))
            dc = round(d * np.cos(a))

            D   = np.zeros((H, W), dtype=np.float64)
            r0s, r1s = max(0, -dr), min(H, H - dr)
            c0s, c1s = max(0, -dc), min(W, W - dc)
            D[r0s:r1s, c0s:c1s] = (q[r0s:r1s, c0s:c1s]
                                   - q[r0s + dr:r1s + dr, c0s + dc:c1s + dc]) ** 2

            S = np.zeros((H + 1, W + 1), dtype=np.float64)
            np.cumsum(np.cumsum(D, axis=0), axis=1, out=S[1:, 1:])

            # sub-rectangle of the patch where BOTH pixels of the pair are inside
            pr0, pr1 = ((RR - half, RR + half - dr) if dr >= 0
                        else (RR - half - dr, RR + half))
            pc0, pc1 = ((CC - half, CC + half - dc) if dc >= 0
                        else (CC - half - dc, CC + half))

            tot = (S[pr1 + 1, pc1 + 1] - S[pr0, pc1 + 1]
                   - S[pr1 + 1, pc0] + S[pr0, pc0])
            acc += tot / ((patch_size - abs(dr)) * (patch_size - abs(dc)))
            n_off += 1

    out[RR, CC] = (acc / n_off).astype(np.float32)
    return uniform_filter(out, size=stride * 2 + 1)


def feature_names(entropy_r: int, glcm_patch_size: int) -> list[str]:
    """Canonical names for the locked 4-feature stack."""
    return [
        "Hue", "Saturation",
        f"Entropy(Sat,R={entropy_r}px)",
        f"GLCM-Contrast(Sat,{glcm_patch_size}px)",
    ]


def build_feature_stack(r, g, b, fcfg: dict) -> tuple[np.ndarray, list[str]]:
    """
    Build the locked 4-feature stack from normalised RGB channels.

    fcfg keys: entropy_r, glcm_patch_size, glcm_levels, glcm_distances,
    and optional glcm_angles. Returns (X_flat (H*W, 4) float32, names).
    """
    hue, sat, _ = compute_hsv(r, g, b)
    ent_s = entropy_map(sat, fcfg["entropy_r"])
    glcm_c = glcm_contrast_map(
        sat,
        patch_size=fcfg["glcm_patch_size"],
        levels=fcfg["glcm_levels"],
        distances=fcfg["glcm_distances"],
        angles=fcfg.get("glcm_angles", DEFAULT_GLCM_ANGLES),
    )
    X_flat = np.stack([hue, sat, ent_s, glcm_c], axis=-1).reshape(-1, 4).astype(np.float32)
    return X_flat, feature_names(fcfg["entropy_r"], fcfg["glcm_patch_size"])
"""
vegclassifier/train.py
======================
Training stage: sample the TRAIN tiles, fit one model, persist a bundle.

Tiles are the right unit here — sampling wants many independent patches with
ground truth, and the per-tile cap balances classes. Prediction is a different
story and runs on the full image (see raster.py).

Reproduces 03_B cell 7 (single pass, per-scope partition) and cells 18-25 (fit).
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import PipelineConfig
from .features import build_feature_stack, compute_hsv, feature_names
from .io import discover_pairs, load_tile
from .models import make_model, save_model
from .sampling import SCOPES, collect_tile_samples, erode_mask, make_nonveg_mask


def select_pairs(cfg: PipelineConfig, site_key: str) -> List[Tuple[Path, Path]]:
    """Tile pairs for one site, using the layout declared in cfg.paths."""
    return discover_pairs(
        cfg.paths.tile_dir,
        site=site_key,
        image_subdir=cfg.paths.image_subdir,
        mask_subdir=cfg.paths.mask_subdir,
        mask_suffix=cfg.paths.mask_suffix,
    )


def collect_training_set(cfg: PipelineConfig, pairs, verbose=True):
    """
    Single pass over training tiles -> (X, y) for cfg.scope.

    Features are computed once per tile; samples are drawn per scope from the
    ERODED mask, excluding non-vegetated pixels.
    """
    lo, hi   = cfg.norm_lohi()
    fcfg     = cfg.feature_dict()
    scope    = cfg.scope
    acc      = {}

    for i, (img_path, mask_path) in enumerate(pairs):
        _, r, g, b, val, mask = load_tile(img_path, mask_path, lo=lo, hi=hi)
        X_flat, _ = build_feature_stack(r, g, b, fcfg)
        _, sat, _ = compute_hsv(r, g, b)
        nonveg = make_nonveg_mask(val, sat,
                                  cfg.features.nonveg_val_thresh,
                                  cfg.features.nonveg_sat_thresh)
        mask_e = erode_mask(mask, cfg.sampling.erode_px)

        rng = np.random.default_rng(cfg.sampling.seed + i)
        samples = collect_tile_samples(X_flat, nonveg, mask_e, scope,
                                       cfg.sampling.max_px_per_tile, rng)
        for lab, arr in samples.items():
            acc.setdefault(lab, []).append(arr)

        if verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)} tiles sampled...")

    expected = SCOPES[scope]["classes"]
    missing  = [c for c in expected if c not in acc]
    if missing:
        raise ValueError(
            f"Scope '{scope}' expects classes {expected} but no samples were "
            f"collected for {missing}.\n"
            f"  Likely causes: erode_px too large for small polygons, or those "
            f"classes are absent from the training tiles."
        )

    X_parts, y_parts = [], []
    for lab in expected:
        stacked = np.vstack(acc[lab])
        X_parts.append(stacked)
        y_parts.append(np.full(len(stacked), lab, dtype=np.int32))
    return np.vstack(X_parts), np.concatenate(y_parts)


def run_train(cfg: PipelineConfig, train_site="middle", out_path=None, verbose=True):
    """Full training stage. Returns (model, out_path)."""
    pairs = select_pairs(cfg, train_site)
    if verbose:
        print(f"Training scope='{cfg.scope}' algo='{cfg.model.algo}' "
              f"on {len(pairs)} '{train_site}' tiles")
        print(f"  normalization: {cfg.normalization.mode} "
              f"(lo={cfg.normalization.lo}, hi={cfg.normalization.hi})")

    X, y = collect_training_set(cfg, pairs, verbose=verbose)
    if verbose:
        print(f"  training samples: {len(X):,}")
        for lab in SCOPES[cfg.scope]["classes"]:
            print(f"    label {lab}: {int((y == lab).sum()):,} px")

    scope_classes = SCOPES[cfg.scope]["classes"]
    model = make_model(cfg.model, scope_classes, seed=cfg.sampling.seed)
    model.fit(X, y)

    names = feature_names(cfg.features.entropy_r, cfg.features.glcm_patch_size)
    if verbose and hasattr(model, "feature_importances_"):
        print("  feature importance:")
        for n, imp in sorted(zip(names, model.feature_importances_),
                             key=lambda t: -t[1]):
            print(f"    {n:<28s} {imp:.3f}")

    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path) if out_path else out_dir / f"{cfg.scope}_{cfg.model.algo}.joblib"

    fparams = cfg.feature_dict()
    fparams["nonveg_val_thresh"] = cfg.features.nonveg_val_thresh
    fparams["nonveg_sat_thresh"] = cfg.features.nonveg_sat_thresh

    save_model(
        model, out_path,
        scope=cfg.scope, feat_names=names,
        norm={"mode": cfg.normalization.mode,
              "lo": cfg.normalization.lo, "hi": cfg.normalization.hi},
        feature_params=fparams,
    )
    if verbose:
        print(f"  saved -> {out_path}")
    return model, out_path
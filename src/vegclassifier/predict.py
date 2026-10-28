"""
vegclassifier/predict.py
========================
Prediction stage: apply a saved model to a FULL-RESOLUTION image and write one
class raster plus one probability raster per class.

Runs on any imagery — no ground-truth mask required. Validity comes from the
image itself (declared nodata, else all-zero pixels), so a fresh flight works.

Inference rules ported from 03_B export_classified_tile:
    - predict every pixel
    - force non-vegetated pixels to Background (0)
    - nodata pixels -> 255 in the class map, 0 in probability maps
"""

from pathlib import Path

import numpy as np
import rasterio as rio

from .models import load_model
from .raster import (
    block_features,
    compute_halo,
    image_valid_mask,
    iter_windows,
    open_like,
    scope_class_names,
)

NODATA = 255


def run_predict(model_path, image_path, out_dir, block=1024,
                save_probabilities=True, verbose=True) -> dict[str, Path]:
    """
    Apply a saved model to one full-resolution image.

    Returns {"class": Path, "prob_<ClassName>": Path, ...}. Output rasters match
    the source extent, CRS, and transform exactly.
    """
    model, meta = load_model(model_path)
    scope   = meta["scope"]
    classes = list(meta["classes"])
    fparams = meta["feature_params"]
    norm    = meta["norm"]
    names   = scope_class_names(scope, classes)

    image_path = Path(image_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    halo = compute_halo(fparams)
    stem = f"{image_path.stem}_{scope}_{meta['kind']}"
    written: dict[str, Path] = {}

    with rio.open(image_path) as src:
        if src.count < 3:
            raise ValueError(
                f"{image_path.name} has {src.count} band(s); the feature stack "
                f"needs 3 (RGB)."
            )
        H, W = src.height, src.width
        if verbose:
            print(f"Predicting {image_path.name}  {W}x{H} px  "
                  f"scope={scope} algo={meta['kind']}")
            print(f"  halo={halo}px, block={block}px, "
                  f"normalization={norm.get('mode')}")

        class_path = out_dir / f"{stem}_class.tif"
        dst_class  = open_like(class_path, src, 1, "uint8", nodata=NODATA)
        written["class"] = class_path

        dst_prob = {}
        if save_probabilities:
            for cls in classes:
                p = out_dir / f"{stem}_prob_{names[cls]}.tif"
                dst_prob[cls] = open_like(p, src, 1, "float32", nodata=0.0)
                written[f"prob_{names[cls]}"] = p

        try:
            n_blocks = ((H + block - 1) // block) * ((W + block - 1) // block)
            for k, (wwin, rwin, crop) in enumerate(iter_windows(H, W, block, halo)):
                arr = src.read(window=rwin)
                X_flat, nonveg, _, _ = block_features(arr, norm, fparams)

                bh, bw = arr.shape[1], arr.shape[2]
                valid  = image_valid_mask(src, arr, rwin)

                pred = model.predict(X_flat).reshape(bh, bw).astype(np.uint8)
                pred[nonveg] = 0             # non-veg -> Background
                pred[~valid] = NODATA        # preserve nodata

                dst_class.write(pred[crop][np.newaxis], window=wwin)

                if dst_prob:
                    proba = model.predict_proba(X_flat).reshape(bh, bw, -1)
                    for ci, cls in enumerate(classes):
                        band = proba[:, :, ci].astype(np.float32)
                        band[nonveg] = 0.0
                        band[~valid] = 0.0
                        dst_prob[cls].write(band[crop][np.newaxis], window=wwin)

                if verbose and (k + 1) % 10 == 0:
                    print(f"  {k+1}/{n_blocks} blocks...")
        finally:
            dst_class.close()
            for d in dst_prob.values():
                d.close()

    if verbose:
        print(f"  wrote {len(written)} raster(s) -> {out_dir}")
    return written

"""
vegclassifier/evaluate.py
=========================
Evaluation stage: apply a saved model to a full-resolution image THAT HAS a
ground-truth mask, then report pooled metrics and write per-class
misclassification rasters.

Distinct from predict by necessity: FP/FN only exist where labels exist. This
runs on the held-out site (south).

Scoring follows 03_B: predict every pixel, but score only against the ERODED
mask so ambiguous hand-drawn polygon boundaries are excluded.

Metrics are pooled by accumulating one confusion matrix across all windows, so
the result is exact — identical to scoring the whole image at once.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio as rio

from .metrics import MISCLASS_CODES, per_class_error_map
from .models import load_model
from .raster import (block_features, compute_halo, iter_windows, open_like,
                     scope_class_names)
from .sampling import erode_mask, remap_mask_for_scope

NODATA = 255


def metrics_from_confusion(cm: np.ndarray, classes: List[int],
                           names: Dict[int, str]) -> Dict:
    """
    Derive per-class and averaged metrics from a pooled confusion matrix.

    cm[i, j] = pixels of true class i predicted as class j. Same definitions as
    metrics.compute_metrics, but computed from accumulated counts so windowed
    evaluation matches whole-image evaluation exactly.
    """
    out = {"per_class": {}, "macro": {}, "weighted": {}}
    precs, recs, f1s, sups = [], [], [], []

    for i, cls in enumerate(classes):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        out["per_class"][cls] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(support), "name": names[cls],
        }
        precs.append(precision); recs.append(recall)
        f1s.append(f1); sups.append(support)

    total = sum(sups)
    out["macro"] = {"precision": float(np.mean(precs)),
                    "recall":    float(np.mean(recs)),
                    "f1":        float(np.mean(f1s))}
    if total > 0:
        w = np.array(sups) / total
        out["weighted"] = {"precision": float(np.dot(w, precs)),
                           "recall":    float(np.dot(w, recs)),
                           "f1":        float(np.dot(w, f1s))}
    else:
        out["weighted"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return out


def run_evaluate(model_path, image_path, mask_path, out_dir, erode_px=10,
                 block=1024, write_error_maps=True, verbose=True) -> Dict:
    """
    Evaluate a saved model against ground truth on a full-resolution image.

    Writes one misclassification raster per class, coded
    TN=0 / TP=1 / FP=2 / FN=3 / nodata=255, and returns pooled metrics.
    """
    model, meta = load_model(model_path)
    scope   = meta["scope"]
    classes = list(meta["classes"])
    fparams = meta["feature_params"]
    norm    = meta["norm"]
    names   = scope_class_names(scope, classes)

    image_path, mask_path = Path(image_path), Path(mask_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    halo = compute_halo(fparams, erode_px=erode_px)
    stem = f"{image_path.stem}_{scope}_{meta['kind']}"
    cm   = np.zeros((len(classes), len(classes)), dtype=np.int64)
    written: Dict[str, Path] = {}

    with rio.open(image_path) as src, rio.open(mask_path) as msk:
        if (msk.height, msk.width) != (src.height, src.width):
            raise ValueError(
                f"Mask/image size mismatch: image is {src.width}x{src.height}, "
                f"mask is {msk.width}x{msk.height}. They must align pixel-for-pixel."
            )
        H, W = src.height, src.width
        if verbose:
            print(f"Evaluating {image_path.name}  {W}x{H} px  "
                  f"scope={scope} algo={meta['kind']}  erode={erode_px}px")

        dst_err = {}
        if write_error_maps:
            for cls in classes:
                p = out_dir / f"{stem}_err_{names[cls]}.tif"
                dst_err[cls] = open_like(p, src, 1, "uint8", nodata=NODATA)
                written[f"err_{names[cls]}"] = p

        try:
            n_blocks = ((H + block - 1) // block) * ((W + block - 1) // block)
            for k, (wwin, rwin, crop) in enumerate(iter_windows(H, W, block, halo)):
                arr = src.read(window=rwin)
                X_flat, nonveg, _, _ = block_features(arr, norm, fparams)
                bh, bw = arr.shape[1], arr.shape[2]

                mask_raw = msk.read(1, window=rwin)
                pred = model.predict(X_flat).reshape(bh, bw).astype(np.uint8)
                pred[nonveg] = 0
                pred[mask_raw == NODATA] = NODATA

                # Erode on the padded block, then remap to scope label space
                gt = remap_mask_for_scope(erode_mask(mask_raw, erode_px), scope)

                pc, gc = pred[crop], gt[crop]
                v = gc != NODATA
                if v.any():
                    t = gc[v].astype(np.int64)
                    p = pc[v].astype(np.int64)
                    for i, ci in enumerate(classes):
                        sel = t == ci
                        if not sel.any():
                            continue
                        for j, cj in enumerate(classes):
                            cm[i, j] += int(np.count_nonzero(p[sel] == cj))

                for cls in dst_err:
                    err = per_class_error_map(pred, gt, cls)
                    dst_err[cls].write(err[crop][np.newaxis], window=wwin)

                if verbose and (k + 1) % 10 == 0:
                    print(f"  {k+1}/{n_blocks} blocks...")
        finally:
            for d in dst_err.values():
                d.close()

    results = metrics_from_confusion(cm, classes, names)
    results["confusion_matrix"] = cm.tolist()
    results["classes"] = classes
    results["class_names"] = {int(c): names[c] for c in classes}
    results["outputs"] = {k: str(v) for k, v in written.items()}

    if verbose:
        print(f"\n  Pooled metrics ({scope}, {meta['kind']}):")
        for cls in classes:
            m = results["per_class"][cls]
            print(f"    {m['name']:<12s} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f}  n={m['support']:,}")
        print(f"    {'MACRO':<12s} P={results['macro']['precision']:.3f} "
              f"R={results['macro']['recall']:.3f} F1={results['macro']['f1']:.3f}")
        print(f"  error map codes: TN={MISCLASS_CODES['TN']} TP={MISCLASS_CODES['TP']} "
              f"FP={MISCLASS_CODES['FP']} FN={MISCLASS_CODES['FN']} "
              f"nodata={MISCLASS_CODES['nodata']}")
    return results

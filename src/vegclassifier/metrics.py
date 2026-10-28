"""
utils/metrics.py
================
Evaluation metrics for pixel-level vegetation classification.

Designed for multiclass problems (background, Phragmites, FAV, Tule)
with support for ignored pixels (nodata=255, eroded edges), asymmetric
cost weighting, and per-tile aggregation.

Public API
----------
compute_metrics(y_true, y_pred, classes, ignore_label)
    Per-class precision, recall, F1, and support from flat label arrays.
    Handles ignored pixels and absent classes gracefully.

confusion_matrix(y_true, y_pred, classes, ignore_label)
    Compute the unnormalised confusion matrix as a numpy array.

aggregate_metrics(tile_metrics)
    Aggregate a list of per-tile metric dicts into dataset-level totals
    using macro and weighted averaging.

bhattacharyya_sep(a, b, n_bins)
    Bhattacharyya overlap coefficient between two 1D distributions.
    Used during feature selection to measure class separability.
    Returns separation score in [0, 1] — higher is better.

weighted_f1(metrics, class_weights)
    Compute a cost-weighted F1 score given per-class weights.
    Use to reflect the asymmetric cost structure of this problem:
    false positives on Tule (native species) are more costly than
    false positives on FAV (also invasive).

Notes
-----
Asymmetric cost structure
    Not all misclassifications have equal cost in this application:
        FP on Tule  → spraying a native species (HIGH cost)
        FP on FAV   → spraying another invasive (LOW cost)
        FN on Phrag → missing a Phragmites patch (MEDIUM cost)
        FN on FAV   → missing a FAV patch (LOW cost)

    Use weighted_f1 with appropriate class_weights to reflect this
    rather than relying on unweighted macro F1.

Ignored pixels
    Pixels with label 255 (nodata or eroded boundary) are excluded
    from all metric calculations. Pass ignore_label=255 (default).
"""


import numpy as np

# ── Type aliases ──────────────────────────────────────────────────────────────
LabelArray = np.ndarray   # shape (N,), dtype int

# ── Class constants ───────────────────────────────────────────────────────────
BACKGROUND  = 0
PHRAGMITES  = 1
FAV         = 2
TULE        = 3
NODATA      = 255
ALL_CLASSES = [BACKGROUND, PHRAGMITES, FAV, TULE]

CLASS_NAMES: dict[int, str] = {
    BACKGROUND: "Background",
    PHRAGMITES: "Phragmites",
    FAV:        "FAV",
    TULE:       "Tule",
}

# Default cost weights reflecting the asymmetric cost structure.
# Tule FPs are the most costly (native species), FAV FPs are least costly.
DEFAULT_CLASS_WEIGHTS: dict[int, float] = {
    BACKGROUND: 1.0,
    PHRAGMITES: 1.5,
    FAV:        0.5,
    TULE:       2.0,
}


def compute_metrics(
            y_true:       LabelArray,
            y_pred:       LabelArray,
            classes:      list[int] | None = None,
            ignore_label: int                 = NODATA,
            ) -> dict[str, float | dict] :
    """
    Compute per-class and aggregate metrics from flat label arrays.

    Handles multiclass problems with ignored pixels. Returns precision,
    recall, F1, and support per class, plus macro and weighted averages.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,), int
        Ground truth class labels. Flat array — call .ravel() on 2D masks
        before passing.
    y_pred : np.ndarray, shape (N,), int
        Predicted class labels. Same shape as y_true.
    classes : List[int] or None
        Which classes to evaluate. If None, uses ALL_CLASSES [0, 1, 2, 3].
        Pass [1, 2] for a binary Phragmites vs FAV evaluation.
    ignore_label : int, default 255
        Label value to exclude from all calculations. Pixels where
        y_true == ignore_label are skipped entirely.

    Returns
    -------
    metrics : Dict
        {
            "per_class": {
                cls_int: {
                    "precision": float,
                    "recall":    float,
                    "f1":        float,
                    "support":   int,    # true pixels of this class
                    "name":      str,
                }
            },
            "macro": {
                "precision": float,  # unweighted mean across classes
                "recall":    float,
                "f1":        float,
            },
            "weighted": {
                "precision": float,  # support-weighted mean
                "recall":    float,
                "f1":        float,
            },
            "accuracy":    float,    # overall pixel accuracy
            "n_valid":     int,      # number of non-ignored pixels scored
        }

    Notes
    -----
    Classes absent from y_true are included in the output with
    precision=recall=f1=0 and support=0, rather than being silently
    dropped. This ensures consistent dict structure across tiles even
    when some classes are not present in a given tile.
    """
    if classes is None:
        classes = ALL_CLASSES

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Exclude ignored pixels
    valid   = y_true != ignore_label
    y_true  = y_true[valid]
    y_pred  = y_pred[valid]
    n_valid = int(valid.sum())

    accuracy = float((y_true == y_pred).mean()) if n_valid > 0 else 0.0

    per_class: dict[int, dict] = {}
    for cls in classes:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        support   = int((y_true == cls).sum())

        per_class[cls] = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   support,
            "name":      CLASS_NAMES.get(cls, f"class_{cls}"),
        }

    # Macro average — unweighted mean across classes with support > 0
    active = [c for c in classes if per_class[c]["support"] > 0]
    macro: dict[str, float] = {
        "precision": float(np.mean([per_class[c]["precision"] for c in active]))
                     if active else 0.0,
        "recall":    float(np.mean([per_class[c]["recall"]    for c in active]))
                     if active else 0.0,
        "f1":        float(np.mean([per_class[c]["f1"]        for c in active]))
                     if active else 0.0,
    }

    # Weighted average — support-weighted mean
    total_support = sum(per_class[c]["support"] for c in classes)
    if total_support > 0:
        weighted: dict[str, float] = {
            key: float(sum(
                per_class[c][key] * per_class[c]["support"]
                for c in classes
            ) / total_support)
            for key in ["precision", "recall", "f1"]
        }
    else:
        weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "per_class": per_class,
        "macro":     macro,
        "weighted":  weighted,
        "accuracy":  accuracy,
        "n_valid":   n_valid,
    }


def confusion_matrix(
            y_true:       LabelArray,
            y_pred:       LabelArray,
            classes:      list[int] | None = None,
            ignore_label: int                 = NODATA,
            ) -> tuple[np.ndarray, list[str]] :
    """
    Compute the unnormalised confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,), int
        Ground truth labels. Call .ravel() on 2D masks before passing.
    y_pred : np.ndarray, shape (N,), int
        Predicted labels.
    classes : List[int] or None
        Classes to include. If None, uses ALL_CLASSES [0, 1, 2, 3].
    ignore_label : int, default 255
        Label value to exclude.

    Returns
    -------
    cm : np.ndarray, shape (n_classes, n_classes), int64
        Confusion matrix where cm[i, j] is the number of pixels with
        true class i predicted as class j.
        Rows = true class, columns = predicted class.
    class_names : List[str]
        Class name strings in the same order as rows/columns.
    """
    if classes is None:
        classes = ALL_CLASSES

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    valid  = y_true != ignore_label
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n      = len(classes)
    cm     = np.zeros((n, n), dtype=np.int64)
    {cls: i for i, cls in enumerate(classes)}

    for i, true_cls in enumerate(classes):
        rows = y_true == true_cls
        for j, pred_cls in enumerate(classes):
            cm[i, j] = int(((y_pred == pred_cls) & rows).sum())

    class_names = [CLASS_NAMES.get(c, f"class_{c}") for c in classes]
    return cm, class_names


def aggregate_metrics(
            tile_metrics: list[dict],
            classes:      list[int] | None = None,
        ) -> dict[str, float | dict] :
    """
    Aggregate per-tile metric dicts into dataset-level totals.

    Computes two aggregations:
        - Pooled: treats all tiles as one large prediction (accurate
          for dataset-level summary, dominated by large tiles).
        - Macro tile: unweighted mean of per-tile F1 scores (treats
          all tiles equally regardless of size).

    Parameters
    ----------
    tile_metrics : List[Dict]
        List of metric dicts returned by compute_metrics, one per tile.
    classes : List[int] or None
        Classes to aggregate. If None, uses ALL_CLASSES.

    Returns
    -------
    agg : Dict
        {
            "pooled": {
                "per_class": { cls: {"precision", "recall", "f1", "support"} },
                "macro":     {"precision", "recall", "f1"},
                "weighted":  {"precision", "recall", "f1"},
                "accuracy":  float,
            },
            "macro_tile": {
                "f1":        float,   # mean F1 across tiles
                "f1_std":    float,   # std of per-tile F1
                "f1_median": float,
                "f1_min":    float,
                "f1_max":    float,
            },
            "n_tiles": int,
        }
    """
    if classes is None:
        classes = ALL_CLASSES

    # ── Pooled aggregation ────────────────────────────────────────────────────
    # Sum TP, FP, FN across all tiles per class, then recompute metrics
    tp_sum: dict[int, int] = {c: 0 for c in classes}
    fp_sum: dict[int, int] = {c: 0 for c in classes}
    fn_sum: dict[int, int] = {c: 0 for c in classes}
    n_correct = 0
    n_total   = 0

    for tm in tile_metrics:
        for cls in classes:
            if cls not in tm["per_class"]:
                continue
            pc  = tm["per_class"][cls]
            sup = pc["support"]
            pre = pc["precision"]
            rec = pc["recall"]
            # Back-calculate TP, FP, FN from precision/recall/support
            tp  = round(rec * sup)
            fn  = sup - tp
            fp  = round(tp / (pre + 1e-9) - tp)
            tp_sum[cls] += tp
            fp_sum[cls] += fp
            fn_sum[cls] += fn
        n_correct += int(tm["accuracy"] * tm["n_valid"])
        n_total   += tm["n_valid"]

    pooled_per_class: dict[int, dict] = {}
    for cls in classes:
        tp, fp, fn = tp_sum[cls], fp_sum[cls], fn_sum[cls]
        pre = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1  = 2 * pre * rec / (pre + rec + 1e-9)
        pooled_per_class[cls] = {
            "precision": float(pre),
            "recall":    float(rec),
            "f1":        float(f1),
            "support":   tp + fn,
            "name":      CLASS_NAMES.get(cls, f"class_{cls}"),
        }

    active       = [c for c in classes if pooled_per_class[c]["support"] > 0]
    total_sup    = sum(pooled_per_class[c]["support"] for c in classes)
    pooled_macro = {
        key: float(np.mean([pooled_per_class[c][key] for c in active]))
        if active else 0.0
        for key in ["precision", "recall", "f1"]
    }
    pooled_weighted = {
        key: float(sum(
            pooled_per_class[c][key] * pooled_per_class[c]["support"]
            for c in classes
        ) / (total_sup + 1e-9))
        for key in ["precision", "recall", "f1"]
    }

    # ── Macro tile aggregation ────────────────────────────────────────────────
    tile_f1s = [tm["macro"]["f1"] for tm in tile_metrics]
    macro_tile = {
        "f1":        float(np.mean(tile_f1s)),
        "f1_std":    float(np.std(tile_f1s)),
        "f1_median": float(np.median(tile_f1s)),
        "f1_min":    float(np.min(tile_f1s)),
        "f1_max":    float(np.max(tile_f1s)),
    }

    return {
        "pooled": {
            "per_class": pooled_per_class,
            "macro":     pooled_macro,
            "weighted":  pooled_weighted,
            "accuracy":  float(n_correct / (n_total + 1e-9)),
        },
        "macro_tile": macro_tile,
        "n_tiles":    len(tile_metrics),
    }


def bhattacharyya_sep(
            a:      np.ndarray,
            b:      np.ndarray,
            n_bins: int = 60,
            ) -> float :
    """
    Compute the Bhattacharyya separation score between two distributions.

    Measures how well two 1D sample arrays can be distinguished.
    Used during feature selection to quantify class separability before
    committing to a feature.

    The Bhattacharyya coefficient BC = sum(sqrt(p_i * q_i)) measures
    overlap — BC=1 means identical distributions, BC=0 means no overlap.
    The separation score returned here is (1 - BC), so higher is better.

    Parameters
    ----------
    a : np.ndarray, shape (N,)
        Sample values for class A (e.g. Phragmites pixels).
    b : np.ndarray, shape (N,)
        Sample values for class B (e.g. Background pixels).
    n_bins : int, default 60
        Number of histogram bins. The bin range is set to cover both
        distributions (min to max across a and b).

    Returns
    -------
    sep : float
        Separation score in [0, 1].
        0.0 = identical distributions (no discriminating power).
        1.0 = perfectly separated distributions.

    Notes
    -----
    Interpretation guide (derived empirically on Winter Island tiles):
        > 0.15 : Good — reliable feature for this class pair
        > 0.08 : Moderate — useful but may need combining with others
        < 0.08 : Poor — limited discriminating power on its own

    Reference
    ---------
    https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    lo   = min(a.min(), b.min())
    hi   = max(a.max(), b.max()) + 1e-9
    bins = np.linspace(lo, hi, n_bins)

    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)

    ha_n = ha / (ha.sum() + 1e-9)
    hb_n = hb / (hb.sum() + 1e-9)

    bc  = float(np.sum(np.sqrt(ha_n * hb_n)))
    return float(1.0 - bc)


def weighted_f1(
            metrics:        dict,
            class_weights:  dict[int, float] | None = None,
            ) -> float:
    """
    Compute a cost-weighted F1 score reflecting asymmetric misclassification
    costs across classes.

    Standard macro F1 treats all classes equally. This function allows
    you to weight classes by their ecological or operational importance.
    For Winter Island: Tule FPs are most costly (spraying native plants),
    FAV FPs are least costly (also an invasive treated with same herbicide).

    Parameters
    ----------
    metrics : Dict
        Output of compute_metrics — must contain the "per_class" key.
    class_weights : Dict[int, float] or None
        Mapping of class label to weight. Higher weight = more important.
        If None, uses DEFAULT_CLASS_WEIGHTS:
            Background  (0): 1.0
            Phragmites  (1): 1.5
            FAV         (2): 0.5
            Tule        (3): 2.0

    Returns
    -------
    wf1 : float
        Weighted F1 score in [0, 1]. Higher is better.

    Example
    -------
    >>> metrics = compute_metrics(y_true, y_pred)
    >>> # Standard macro F1
    >>> print(metrics["macro"]["f1"])
    >>> # Cost-weighted F1 (penalises Tule FPs more)
    >>> print(weighted_f1(metrics))
    >>> # Custom weights
    >>> print(weighted_f1(metrics, class_weights={0:1.0, 1:2.0, 2:0.5, 3:3.0}))
    """
    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS

    per_class = metrics["per_class"]
    total_w   = 0.0
    wf1       = 0.0

    for cls, w in class_weights.items():
        if cls not in per_class:
            continue
        f1       = per_class[cls]["f1"]
        wf1     += w * f1
        total_w += w

    return float(wf1 / (total_w + 1e-9))


def print_metrics(
            metrics:        dict,
            class_weights:  dict[int, float] | None = None,
            title:          str                        = "Metrics",
            ) -> None :
    """
    Pretty-print a metrics dict from compute_metrics.
 
    Parameters
    ----------
    metrics : Dict
        Output of compute_metrics.
    class_weights : Dict[int, float] or None
        If provided, also prints the weighted F1 score.
        If None, uses DEFAULT_CLASS_WEIGHTS.
    title : str, default "Metrics"
        Header string printed above the table.
    """
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Class':15s}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>10}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")
 
    for cls, pc in sorted(metrics["per_class"].items()):
        print(
            f"  {pc['name']:15s}  "
            f"{pc['precision']:>10.3f}  "
            f"{pc['recall']:>8.3f}  "
            f"{pc['f1']:>8.3f}  "
            f"{pc['support']:>10,}"
        )
 
    print(f"  {'-'*15}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")
    m = metrics["macro"]
    w = metrics["weighted"]
    print(f"  {'Macro avg':15s}  {m['precision']:>10.3f}  {m['recall']:>8.3f}  {m['f1']:>8.3f}")
    print(f"  {'Weighted avg':15s}  {w['precision']:>10.3f}  {w['recall']:>8.3f}  {w['f1']:>8.3f}")
    print(f"\n  Accuracy  : {metrics['accuracy']:.3f}")
    if 'n_valid' in metrics:
        print(f"  N valid px: {metrics['n_valid']:,}")
    print(f"  Cost-weighted F1: {weighted_f1(metrics, class_weights):.3f}")
    print(sep)

# ── Per-class misclassification coding ────────────────────────────────────────
# Codes for the per-class error rasters written by the eval stage.
MISCLASS_CODES = {
    "TN":     0,     # not this class, predicted not this class
    "TP":     1,     # this class, predicted this class
    "FP":     2,     # not this class, predicted this class
    "FN":     3,     # this class, predicted something else
    "nodata": 255,   # ignored (eroded boundary, nodata, non-scored)
}


def per_class_error_map(pred, gt, cls, ignore_label=NODATA):
    """
    Build a per-class misclassification raster coded TN/TP/FP/FN/nodata.

    For class `cls`, every scored pixel becomes 0=TN, 1=TP, 2=FP, 3=FN.
    Pixels where gt == ignore_label become 255. Returns uint8, same shape
    as inputs. One of these is written per class by the eval stage.
    """
    pred = np.asarray(pred)
    gt   = np.asarray(gt)

    coded = np.full(pred.shape, MISCLASS_CODES["nodata"], dtype=np.uint8)
    valid = gt != ignore_label
    p = pred == cls
    t = gt == cls

    coded[valid & ~p & ~t] = MISCLASS_CODES["TN"]
    coded[valid &  p &  t] = MISCLASS_CODES["TP"]
    coded[valid &  p & ~t] = MISCLASS_CODES["FP"]
    coded[valid & ~p &  t] = MISCLASS_CODES["FN"]
    return coded

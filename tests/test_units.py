"""Unit tests: features, config, sampling, metrics, windowing."""

import numpy as np
import pytest

from vegclassifier.config import PipelineConfig
from vegclassifier.evaluate import metrics_from_confusion
from vegclassifier.features import (
    build_feature_stack,
    glcm_contrast_map,
    glcm_contrast_map_reference,
)
from vegclassifier.metrics import compute_metrics, per_class_error_map
from vegclassifier.raster import align_block, glcm_stride, iter_windows
from vegclassifier.sampling import erode_mask, make_nonveg_mask, remap_mask_for_scope


# ── features ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("shape", [(64, 64), (97, 130)])
def test_glcm_vectorised_matches_reference(shape):
    ch = np.random.default_rng(0).random(shape).astype(np.float32)
    assert np.array_equal(glcm_contrast_map(ch), glcm_contrast_map_reference(ch))


def test_feature_stack_shape():
    rng = np.random.default_rng(0)
    r, g, b = (rng.random((50, 60)).astype(np.float32) for _ in range(3))
    X, names = build_feature_stack(r, g, b, PipelineConfig().feature_dict())
    assert X.shape == (50 * 60, 4)
    assert X.dtype == np.float32
    assert len(names) == 4


# ── config ────────────────────────────────────────────────────────────────────
def test_default_config_is_valid():
    PipelineConfig().validate()


def test_shipped_config_loads():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = PipelineConfig.from_yaml(root / "configs" / "rf_multiclass.yaml")
    assert cfg.scope == "multiclass"


@pytest.mark.parametrize("field,value", [
    ("scope", "trinary"), ("model.algo", "xgb"), ("normalization.mode", "zscore"),
])
def test_invalid_config_raises(field, value):
    cfg = PipelineConfig()
    obj = cfg
    *parents, leaf = field.split(".")
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, leaf, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_yaml_roundtrip(tmp_path):
    cfg = PipelineConfig()
    cfg.model.algo = "svm"
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    assert PipelineConfig.from_yaml(p) == cfg


# ── sampling ──────────────────────────────────────────────────────────────────
def test_scope_remap():
    m = np.array([[0, 1, 2, 3, 255]], np.uint8)
    assert remap_mask_for_scope(m, "multiclass").tolist() == [[0, 1, 2, 3, 255]]
    assert remap_mask_for_scope(m, "merged").tolist()     == [[0, 1, 2, 0, 255]]
    assert remap_mask_for_scope(m, "binary").tolist()     == [[0, 1, 255, 255, 255]]


def test_erode_zero_is_copy():
    m = np.ones((5, 5), np.uint8)
    out = erode_mask(m, 0)
    assert np.array_equal(out, m) and out is not m


def test_erode_marks_boundaries():
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:15] = 1
    out = erode_mask(m, 2)
    assert out[10, 10] == 1          # interior kept
    assert out[5, 5] == 255          # class edge ignored


def test_nonveg_mask():
    val = np.array([0.1, 0.1, 0.9])
    sat = np.array([0.1, 0.9, 0.1])
    assert make_nonveg_mask(val, sat).tolist() == [True, False, False]


# ── metrics ───────────────────────────────────────────────────────────────────
def test_error_map_codes():
    gt   = np.array([1, 1, 0, 0, 255])
    pred = np.array([1, 0, 1, 0, 1])
    assert per_class_error_map(pred, gt, 1).tolist() == [1, 3, 2, 0, 255]


def test_pooled_metrics_match_flat_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 2000)
    y_pred = np.where(rng.random(2000) < 0.7, y_true, rng.integers(0, 4, 2000))
    classes = [0, 1, 2, 3]
    cm = np.array([[np.sum((y_true == i) & (y_pred == j)) for j in classes] for i in classes])
    pooled = metrics_from_confusion(cm, classes, {c: str(c) for c in classes})
    flat = compute_metrics(y_true, y_pred, classes=classes)
    for c in classes:
        for k in ("precision", "recall", "f1"):
            assert pooled["per_class"][c][k] == pytest.approx(flat["per_class"][c][k])


# ── windowing ─────────────────────────────────────────────────────────────────
def test_align_block_rounds_up_to_stride():
    fp = {"glcm_patch_size": 21}
    s = glcm_stride(21)
    assert s == 5
    assert align_block(1024, fp) == 1025
    assert align_block(1025, fp) == 1025
    assert align_block(1, fp) == 5


def test_windows_cover_every_pixel_once():
    H, W = 103, 77
    count = np.zeros((H, W), int)
    for wwin, rwin, crop in iter_windows(H, W, block=25, halo=10):
        count[wwin.row_off:wwin.row_off + wwin.height,
              wwin.col_off:wwin.col_off + wwin.width] += 1
        assert crop[0].stop - crop[0].start == wwin.height
        assert crop[1].stop - crop[1].start == wwin.width
    assert (count == 1).all()

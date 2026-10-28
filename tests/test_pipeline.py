"""
End-to-end tests on the synthetic site: preprocess -> train -> predict -> evaluate.

These check that the stages run and stay consistent. They do not check accuracy.
"""

from pathlib import Path

import numpy as np
import rasterio as rio

from vegclassifier.evaluate import run_evaluate
from vegclassifier.io import discover_pairs
from vegclassifier.predict import run_predict
from vegclassifier.train import run_train


def _read_all(paths: dict) -> dict:
    out = {}
    for k, p in paths.items():
        with rio.open(p) as src:
            out[k] = src.read()
    return out


def test_preprocess_writes_mask_and_tiles(cfg, tiles):
    for s in ("middle", "south"):
        assert len(tiles[s]) > 0
        assert (Path(cfg.paths.image_dir) / f"{s}_mask.tif").exists()
    pairs = discover_pairs(cfg.paths.tile_dir, site="middle",
                           image_subdir=cfg.paths.image_subdir,
                           mask_subdir=cfg.paths.mask_subdir,
                           mask_suffix=cfg.paths.mask_suffix)
    assert len(pairs) == len(tiles["middle"])


def test_burned_mask_has_all_classes(cfg, tiles):
    with rio.open(Path(cfg.paths.image_dir) / "south_mask.tif") as m:
        values = set(np.unique(m.read(1)).tolist())
    assert {0, 1, 2, 3, 255} <= values


def test_train_writes_model(model_path):
    assert Path(model_path).exists()


def test_retraining_is_deterministic(cfg, model_path, tmp_path):
    out = tmp_path / "again.joblib"
    run_train(cfg, out_path=out, verbose=False)
    image = Path(cfg.paths.image_dir) / "south_winter2025.tif"
    a = _read_all(run_predict(model_path, image, tmp_path / "a", verbose=False))
    b = _read_all(run_predict(out, image, tmp_path / "b", verbose=False))
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_predict_matches_source_geometry(cfg, model_path, tmp_path):
    image = Path(cfg.paths.image_dir) / "south_winter2025.tif"
    written = run_predict(model_path, image, tmp_path, verbose=False)
    assert "class" in written and len(written) == 5      # class + 4 probabilities
    with rio.open(image) as src, rio.open(written["class"]) as out:
        assert (out.height, out.width) == (src.height, src.width)
        assert out.crs == src.crs
        assert out.transform == src.transform
        cls = out.read(1)
    assert set(np.unique(cls).tolist()) <= {0, 1, 2, 3, 255}
    assert (cls[:, -30:] == 255).all()                    # nodata strip preserved


def test_predict_is_block_size_invariant(cfg, model_path, tmp_path):
    image = Path(cfg.paths.image_dir) / "south_winter2025.tif"
    ref = _read_all(run_predict(model_path, image, tmp_path / "one", block=5000, verbose=False))
    for block in (64, 137):
        got = _read_all(run_predict(model_path, image, tmp_path / str(block),
                                    block=block, verbose=False))
        for k in ref:
            assert np.array_equal(ref[k], got[k]), f"{k} differs at block={block}"


def test_evaluate_is_block_size_invariant(cfg, model_path, tmp_path):
    image = Path(cfg.paths.image_dir) / "south_winter2025.tif"
    mask = Path(cfg.paths.image_dir) / "south_mask.tif"
    ref = run_evaluate(model_path, image, mask, tmp_path / "one", block=5000, verbose=False)
    got = run_evaluate(model_path, image, mask, tmp_path / "small", block=64, verbose=False)
    assert ref["confusion_matrix"] == got["confusion_matrix"]
    for k in ref["outputs"]:
        with rio.open(ref["outputs"][k]) as a, rio.open(got["outputs"][k]) as b:
            assert np.array_equal(a.read(), b.read()), k

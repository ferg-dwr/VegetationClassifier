"""
Shared fixtures. Tests run on a small synthetic site built in a temp folder, so
no project data is needed. Synthetic results say nothing about model quality;
they only check that the pipeline runs and behaves consistently.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin
from shapely.geometry import box

from vegclassifier.config import PipelineConfig

H, W = 420, 410          # deliberately not multiples of the tile, block, or stride
PIXEL = 0.1
X0, Y0 = 600000.0, 4220000.0
CRS = "EPSG:32610"

# (row0, row1, col0, col1) per class, in pixels
REGIONS = {1: (20, 180, 20, 190), 2: (220, 400, 20, 190), 3: (20, 180, 220, 380)}
COLORS = {0: (120, 110, 90), 1: (150, 140, 80), 2: (60, 140, 60), 3: (90, 120, 70)}
NAMES = {1: "Phragmites", 2: "FAV", 3: "Tule"}


def _write_site(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    lab = np.zeros((H, W), np.uint8)
    for c, (r0, r1, c0, c1) in REGIONS.items():
        lab[r0:r1, c0:c1] = c
    img = np.zeros((3, H, W), np.uint8)
    for c, col in COLORS.items():
        m = lab == c
        for b in range(3):
            img[b][m] = np.clip(col[b] + rng.normal(0, 8 + 6 * c, m.sum()), 1, 255)
    img[:, :, W - 30:] = 0                     # nodata strip along one edge
    with rio.open(path, "w", driver="GTiff", height=H, width=W, count=3,
                  dtype="uint8", crs=CRS, transform=from_origin(X0, Y0, PIXEL, PIXEL)) as d:
        d.write(img)


@pytest.fixture(scope="session")
def site(tmp_path_factory):
    """Synthetic orthomosaics for 'middle' and 'south' plus class polygons."""
    root = tmp_path_factory.mktemp("site")
    (root / "imagery").mkdir()
    (root / "polygons").mkdir()
    _write_site(root / "imagery" / "middle_winter2025.tif", seed=0)
    _write_site(root / "imagery" / "south_winter2025.tif", seed=1)
    for c, (r0, r1, c0, c1) in REGIONS.items():
        geom = box(X0 + c0 * PIXEL, Y0 - r1 * PIXEL, X0 + c1 * PIXEL, Y0 - r0 * PIXEL)
        gpd.GeoDataFrame(geometry=[geom], crs=CRS).to_file(
            root / "polygons" / f"{NAMES[c]}.geojson")
    return root


@pytest.fixture(scope="session")
def cfg(site):
    """Default config pointed at the synthetic site, with a small forest for speed."""
    c = PipelineConfig()
    c.paths.image_dir  = str(site / "imagery")
    c.paths.data_dir   = str(site / "polygons")
    c.paths.tile_dir   = str(site / "tiles")
    c.paths.output_dir = str(site / "models")
    c.model.rf_n_estimators = 20
    c.validate()
    return c


@pytest.fixture(scope="session")
def tiles(cfg):
    """Run preprocess for both sites once per test session."""
    from vegclassifier.preprocess import run_preprocess
    out = {}
    for s in ("middle", "south"):
        out[s] = run_preprocess(cfg, site=s, tile_size=200, verbose=False)
    return out


@pytest.fixture(scope="session")
def model_path(cfg, tiles):
    """Train once per test session."""
    from vegclassifier.train import run_train
    _, path = run_train(cfg, verbose=False)
    return path

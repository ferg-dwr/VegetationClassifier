# VegetationClassifier

Per-pixel classification of invasive and native wetland vegetation from RGB drone
imagery. Four commands take an orthomosaic and hand-digitised polygons to a
classified raster, probability rasters, and accuracy metrics.

| value | class | role |
|-------|-------|------|
| 0 | Background | water, mud, bare ground, non-target vegetation |
| 1 | Phragmites | target invasive |
| 2 | FAV | floating aquatic vegetation, invasive |
| 3 | Tule | native, high protection priority |
| 255 | nodata | outside flight extent / masked |

By default, models train on the `middle` site and are evaluated on the held-out
`south` site.

This README covers installing and running the pipeline. Project context, method
rationale, data locations, and infrastructure are documented separately.

---

## Requirements

- Python. The rest of these you can learn as you go.
- [Miniforge](https://github.com/conda-forge/miniforge) (conda with the
  conda-forge channel). See the
  [conda user guide](https://docs.conda.io/projects/conda/en/stable/user-guide/).
- [git](https://git-scm.com), to clone and update the repo.
- [QGIS](https://qgis.org), to look at the output rasters. Optional.

Input imagery must be three-band RGB, uint8, as GeoTIFF.

---

## Setup

The geospatial stack (rasterio, geopandas) depends on GDAL system libraries.
Install it with conda. pip alone is a common source of setup failures.

```bash
git clone https://github.com/ferg-dwr/VegetationClassifier.git
cd VegetationClassifier
conda env create -f environment.yml
conda activate invasives
pip install -e .          # installs the `vegclassifier` command
vegclassifier --help
pytest                    # optional: confirms the install works, ~30 s
```

Already have a working geospatial environment? `pip install -e .` on its own is
enough. Add `pip install -e ".[test]"` to get pytest.

### Record the environment

Package versions can change results. Record the exact environment alongside
each set of outputs:

```bash
mkdir -p outputs
conda env export > outputs/environment.lock.yml
```

---

## Configure a run

**Nothing about the paths is hardcoded.** Data lives outside the repo; a YAML
config tells the pipeline where. Start from a copy:

```bash
cp configs/rf_multiclass.yaml configs/my_run.yaml
```

Then edit the `paths:` block. Absolute paths are the safer choice.

**Run every command from the repo root.** Relative paths in the config resolve
against the directory you run `vegclassifier` from, not against the config file.

```yaml
paths:
  image_dir:  /path/to/orthomosaics     # full-resolution GeoTIFFs; burned masks are written here
  data_dir:   /path/to/polygons         # digitised class polygons
  tile_dir:   /path/to/tiles            # temporary tiles (created by `preprocess`)
  output_dir: models                    # trained models

  # Tile layout that `preprocess` writes and `train` reads. Leave as is.
  image_subdir: rgb_images
  mask_subdir:  class_masks
  mask_suffix:  _mask
```

### File naming expectations

| what | expected | how to override |
|------|----------|-----------------|
| orthomosaic | `<site>_winter2025.tif` in `image_dir` | `--image-name` |
| burned mask | `<site>_mask.tif` in `image_dir` | `--mask-name` |
| polygons | `Phragmites`, `FAV`, `Tule` in `data_dir` (`.geojson`, `.shp`, `.gpkg`, or `.json`; case-insensitive) | point `data_dir` at a folder with these names |

`<site>` is whatever you pass to `--site` (here: `middle`, `south`). If your
imagery is named differently, pass `--image-name` / `--mask-name` rather than
renaming your data.

Polygon files can be in any CRS. They are reprojected to the orthomosaic's CRS
when the mask is burned. Each polygon file must have its CRS defined (for a
shapefile, a `.prj` file).

---

## Run the pipeline

### 1. Preprocess: burn masks and cut tiles

```bash
vegclassifier preprocess --config configs/my_run.yaml --site middle
vegclassifier preprocess --config configs/my_run.yaml --site south
```

Burns the `Phragmites` / `FAV` / `Tule` polygons into a mask aligned to the
orthomosaic (`<site>_mask.tif` in `image_dir`), then tiles image and mask into
`tile_dir`. Useful flags: `--tile-size` (default 1024), `--min-valid-fraction`
(default 0.3, drops mostly-nodata tiles), and `--skip-burn` to reuse an existing
mask.

When polygons overlap, the higher class wins: Phragmites > FAV > Tule >
Background.

### 2. Train

```bash
vegclassifier train --config configs/my_run.yaml
```

Trains on the `middle` tiles and writes `models/<scope>_<algo>.joblib`, here
`models/multiclass_rf.joblib`. Override the config without editing it:

```bash
vegclassifier train --config configs/my_run.yaml --scope binary --algo svm
```

| `--scope` | classes |
|-----------|---------|
| `binary` | Background, Phragmites (FAV and Tule left out of training) |
| `multiclass` | Background, Phragmites, FAV, Tule |
| `merged` | Background + Tule, Phragmites, FAV |

| `--algo` | model |
|----------|-------|
| `gm` | per-class Gaussian |
| `gmm` | Gaussian mixture, 2 components for Background |
| `rf` | random forest, uses all CPU cores |
| `svm` | RBF SVM, single-threaded, subsamples above 40,000 training pixels |

### 3. Predict on imagery

```bash
vegclassifier predict \
  --model models/multiclass_rf.joblib \
  --image /path/to/orthomosaics/south_winter2025.tif \
  --out   outputs/predictions
```

Writes, at full source resolution and georeferencing:

- `<image>_<scope>_<algo>_class.tif`: one band, uint8, class per pixel, nodata 255
- `<image>_<scope>_<algo>_prob_<Class>.tif`: one float32 raster per class

No ground-truth mask needed. Works on any new flight.

Probability rasters are model scores, not calibrated probabilities. Read them as
relative confidence.

### 4. Evaluate against ground truth

```bash
vegclassifier evaluate \
  --model models/multiclass_rf.joblib \
  --image /path/to/orthomosaics/south_winter2025.tif \
  --mask  /path/to/orthomosaics/south_mask.tif \
  --out   outputs/evaluation \
  --json  outputs/evaluation/metrics_multiclass_rf.json
```

Prints pooled precision/recall/F1 per class, writes the metrics and confusion
matrix to the JSON file, and writes one misclassification raster per class:

`<image>_<scope>_<algo>_err_<Class>.tif`, coded:

| code | meaning |
|------|---------|
| 0 | true negative |
| 1 | true positive |
| 2 | **false positive** |
| 3 | **false negative** |
| 255 | not scored (nodata or eroded boundary) |

Open in QGIS and style 2 and 3 to see where the model fails.

### 5. Clean up

Tiles are temporary. Once training is done, delete them:

```bash
rm -rf /path/to/tiles        # the tile_dir from your config
```

Keep the trained models, the burned masks (`evaluate` reads them), the outputs,
and `environment.lock.yml`.

### Running every scope and algorithm

To compare all twelve combinations on the same tiles, run step 1 once, loop over
steps 2 and 4, then clean up (step 5):

```bash
IMG=/path/to/orthomosaics/south_winter2025.tif
MASK=/path/to/orthomosaics/south_mask.tif

for scope in binary multiclass merged; do
  for algo in gm gmm rf svm; do
    vegclassifier train --config configs/my_run.yaml --scope $scope --algo $algo
    vegclassifier evaluate --model models/${scope}_${algo}.joblib \
      --image $IMG --mask $MASK --out outputs/evaluation \
      --json outputs/evaluation/metrics_${scope}_${algo}.json
  done
done
```

---

## Configuration reference

One YAML fully defines a run. Keep the config with the outputs it produced. The
defaults in `configs/rf_multiclass.yaml` are the values chosen for this project;
the reasons are in the project documentation.

| key | what it controls |
|-----|------------------|
| `scope` | which classes are trained and scored (see table above) |
| `paths.*` | where inputs are read and outputs written |
| `normalization.mode`, `lo`, `hi` | how pixel values are scaled before features. Default `identity` (0 to 255) |
| `features.entropy_r` | entropy window radius, px |
| `features.glcm_patch_size`, `glcm_levels`, `glcm_distances` | GLCM texture window and settings |
| `features.nonveg_val_thresh`, `nonveg_sat_thresh` | pixels below both thresholds are treated as non-vegetated and set to Background |
| `sampling.max_px_per_tile` | max training pixels per class per tile |
| `sampling.erode_px` | px stripped from polygon edges before training and scoring |
| `sampling.seed` | random seed. Same seed and config give the same model |
| `model.algo` and `model.*` | algorithm and its hyperparameters |

Feature radii are in pixels. At a different image resolution they cover a
different ground distance.

---

## Tiles vs. full images

Tiles are for preprocessing and training only. `predict` and `evaluate` read the
full orthomosaic in windows, each with a margin of context, and write one raster
covering the whole site. Output does not depend on `--block`. Set it for memory
only. The block is rounded up to a multiple of the GLCM stride (1024 runs as
1025); the log prints the value used.

---

## Tests

```bash
pytest
```

Runs in about 30 seconds on a small synthetic site, so no project data is needed.
Covers configuration checks, feature and metric functions, and a full
preprocess → train → predict → evaluate run. Checks that results do not change
with `--block` and that retraining with the same config gives the same model.
Synthetic results say nothing about model accuracy.

Run the tests after any code change.

---

## Common errors

| message | likely cause |
|---------|--------------|
| `Config file not found` | not running from the repo root, or wrong `--config` path |
| `Source image not found` | `paths.image_dir` is wrong, or the file is not named `<site>_winter2025.tif` (use `--image-name`) |
| `No Phragmites or FAV vector file found` | `paths.data_dir` is wrong, or the polygon files are named differently |
| `No image/mask tile pairs found` | `preprocess` has not been run for that site, or `paths.tile_dir` changed |
| `Scope ... expects classes ... but no samples were collected` | a class is missing from the training tiles, or `sampling.erode_px` removed small polygons |

---

## Package layout

```
src/vegclassifier/
├── io.py         # tile loading, mask burning, tiling, tile discovery
├── features.py   # HSV, entropy, GLCM contrast, 4-feature stack
├── sampling.py   # non-veg mask, erosion, scope remap, per-tile sampling
├── metrics.py    # metrics, confusion matrix, per-class error coding
├── config.py     # YAML config dataclasses + validation
├── models.py     # 4 algorithms behind one interface + save/load
├── raster.py     # windowed full-image inference
├── preprocess.py # mask burning + tiling
├── train.py      # training stage
├── predict.py    # prediction stage
├── evaluate.py   # evaluation stage
└── cli.py        # subcommands
tests/            # pytest suite on synthetic data
```

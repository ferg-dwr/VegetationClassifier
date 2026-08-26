# VegetationClassifier

Per-pixel classification of invasive and native wetland vegetation from RGB drone
imagery (0.1 m/px) over Winter Island, CA. Converts the analysis in the
`InvasivePlantDetection` notebooks into a reproducible, CLI-driven pipeline.

| value | class | role |
|-------|-------|------|
| 0 | Background | water, mud, bare ground, non-target vegetation |
| 1 | Phragmites | target invasive |
| 2 | FAV | floating aquatic vegetation — invasive |
| 3 | Tule | native, high protection priority |
| 255 | nodata | outside flight extent / masked |

Models train on `middle` tiles and are evaluated on the held-out `south` site.

---

## Setup

The geospatial stack (rasterio, geopandas) depends on GDAL system libraries.
Install it with conda — pip alone is a common source of setup failures.

```bash
conda env create -f environment.yml
conda activate invasives
pip install -e .          # installs the `vegclassifier` command
vegclassifier --help
```

Already have a working geospatial environment? `pip install -e .` on its own is
enough.

---

## Pointing it at your data

**Nothing about the paths is hardcoded.** Data lives outside the repo; a YAML
config tells the pipeline where. Start from a copy:

```bash
cp configs/rf_multiclass.yaml configs/my_run.yaml
```

Then edit the `paths:` block. Absolute paths are fine and are the safer choice
if your checkout is not a sibling of the data directories.

```yaml
paths:
  image_dir:  /path/to/orthomosaics     # full-resolution GeoTIFFs + masks
  data_dir:   /path/to/shapefiles       # digitised polygons
  tile_dir:   /path/to/tiles            # tiles (created by `preprocess`)
  output_dir: models                    # trained models (inside the repo)

  # How tiles are laid out on disk. Change these to match an existing tile set,
  # or leave them and let `preprocess` create this structure for you:
  #   <tile_dir>/<image_subdir>/<site>/<stem>.tif
  #   <tile_dir>/<mask_subdir>/<site>/<stem><mask_suffix>.tif
  image_subdir: rgb_images
  mask_subdir:  class_masks
  mask_suffix:  _mask
```

### File naming expectations

| what | expected | how to override |
|------|----------|-----------------|
| orthomosaic | `<site>_winter2025.tif` in `image_dir` | `--image-name` |
| burned mask | `<site>_mask.tif` in `image_dir` | `--mask-name` |
| polygons | `Phragmites`, `FAV`, `Tule` in `data_dir` (`.geojson`, `.shp`, `.gpkg`, or `.json`; case-insensitive) | rename the files |
| tiles | `<stem>_r000_c000.tif` + `<stem>_r000_c000_mask.tif` | `mask_suffix` |

`<site>` is whatever you pass to `--site` (here: `middle`, `south`). If your
files are named differently, pass `--image-name` / `--mask-name` rather than
renaming your data.

If you already have tiles from another process, skip `preprocess` entirely and
set `image_subdir` / `mask_subdir` / `mask_suffix` to match what is on disk. Run
a quick check that discovery finds them:

```bash
python -c "
from vegclassifier.config import PipelineConfig
from vegclassifier.io import discover_pairs
c = PipelineConfig.from_yaml('configs/my_run.yaml')
discover_pairs(c.paths.tile_dir, site='middle',
               image_subdir=c.paths.image_subdir,
               mask_subdir=c.paths.mask_subdir,
               mask_suffix=c.paths.mask_suffix)"
```

---

## Assumptions this pipeline makes

Read this before applying it to a new site. These are baked in, and several are
research findings rather than arbitrary defaults.

**Fixed by the code**

- **Four classes with fixed meanings:** 0 Background, 1 Phragmites, 2 FAV,
  3 Tule, 255 nodata. Changing the class scheme means editing `sampling.py` and
  `io.py`, not a config value.
- **Three-band RGB uint8 input.** No NIR, no multispectral.
- **Burn priority** when polygons overlap: Phragmites > FAV > Tule > Background.
- **Non-vegetation rule:** Value < 0.35 AND Saturation < 0.35 is treated as
  non-vegetated and forced to Background at inference. Tune per site in
  `features.nonveg_*`.
- **Tiles train; full images predict.** Prediction and evaluation always read
  the full orthomosaic (see below).

**Validated for this site — re-check before trusting elsewhere**

- **The 4-feature stack** (Hue, Saturation, Entropy(Sat, R=61 px),
  GLCM-Contrast(Sat, 21 px)) was selected for these species at 0.1 m/px. The
  radii are physical scales (~6.1 m and ~2.1 m); at a different resolution they
  describe different real-world sizes.
- **Identity normalization** won 28 of 36 channel x pair combinations here.
- **Erosion of 10 px (~1 m)** removes ambiguous polygon boundaries. Larger
  values silently delete small polygons entirely.
- **Sampling cap of 2000 px per class per tile.**

**A different site or sensor needs revalidation, not just a new config.** The
feature study lives in the `InvasivePlantDetection` notebooks (`02_B`); rerun it
before assuming these features transfer.

---

## Runbook

Copy-paste these. Four commands cover everything.

### 1. Preprocess — burn masks and cut tiles

```bash
vegclassifier preprocess --config configs/my_run.yaml --site middle
vegclassifier preprocess --config configs/my_run.yaml --site south
```

Burns `Phragmites` / `FAV` / `Tule` polygons into a mask aligned to the
orthomosaic, then tiles image and mask into `tile_dir`. Useful flags:
`--tile-size` (default 1024), `--min-valid-fraction` (default 0.3, drops
mostly-nodata tiles), and `--skip-burn` to reuse an existing mask.

Already have masks and tiles? Skip this step.

### 2. Train

```bash
vegclassifier train --config configs/rf_multiclass.yaml
```

Writes `models/multiclass_rf.joblib`. Override the config without editing it:

```bash
vegclassifier train --config configs/rf_multiclass.yaml --scope binary --algo svm
```

### 3. Predict on imagery

```bash
vegclassifier predict \
  --model models/multiclass_rf.joblib \
  --image data/imagery/south_winter2025.tif \
  --out   outputs/predictions
```

Produces, at full source resolution and georeferencing:

- `<image>_<scope>_<algo>_class.tif` — one band, uint8, class per pixel, nodata 255
- `<image>_<scope>_<algo>_prob_<Class>.tif` — one float32 raster per class

No ground-truth mask needed. Works on any new flight.

### 4. Evaluate against ground truth

```bash
vegclassifier evaluate \
  --model models/multiclass_rf.joblib \
  --image data/imagery/south_winter2025.tif \
  --mask  data/imagery/south_mask.tif \
  --out   outputs/evaluation \
  --json  outputs/evaluation/metrics.json
```

Prints pooled precision/recall/F1 per class and writes one misclassification
raster per class:

`<image>_<scope>_<algo>_err_<Class>.tif`, coded:

| code | meaning |
|------|---------|
| 0 | true negative |
| 1 | true positive |
| 2 | **false positive** |
| 3 | **false negative** |
| 255 | not scored (nodata or eroded boundary) |

Open in QGIS and style 2 and 3 to see exactly where the model fails.

---

## Tiles vs. full images

**Tiles are for preprocessing and training only.** Prediction and evaluation read
the full-resolution orthomosaic directly and emit one raster covering the whole
site. Two reasons this is not negotiable:

1. **Tiling drops data.** `min_valid_fraction` discards mostly-nodata tiles, so
   mosaicking tiles would leave holes and not reconstruct the source extent.
2. **Per-tile features have edge artefacts.** Entropy (R=61 px) and GLCM (21 px)
   are neighbourhood operations; at a tile border the neighbourhood is truncated
   and those pixels get wrong values — a visible grid pattern across a mosaic.

Full-image inference processes windows with a **halo** of context, computing
features on a padded block and writing only the valid interior. Block size and
halo are forced to multiples of the GLCM stride so the co-occurrence grid stays
phase-aligned across windows.

*Verified:* output is bit-identical whether processed as one window or 25
(`--block 64` vs `--block 1024`) — zero differing pixels, zero probability
difference, identical confusion matrices. Tune `--block` for memory only; it
cannot change results.

---

## Configuration

One YAML fully defines a run. Key locked values (from `02_B` / `03_B`):

- **Features:** Hue, Saturation, Entropy(Sat, R=61 px), GLCM-Contrast(Sat, 21 px).
  All pairwise correlations < 0.7. Changing these invalidates the feature study.
- **Normalization:** `identity` (lo=0, hi=255) — won 28 of 36 channel × pair
  combinations. `per_tile` destroys absolute colour information; avoid it.
- **Erosion:** 10 px (~1 m) removes ambiguous polygon edges without the
  small-polygon filtering bias that larger values introduce.
- **Sampling cap:** 2000 px per class per tile.

## Scopes

| scope | classes | rationale |
|-------|---------|-----------|
| `binary` | BG, Phragmites | simplest detection case |
| `multiclass` | BG, Phragmites, FAV, Tule | full 4-class |
| `merged` | BG+Tule, Phragmites, FAV | Tule and BG are both "no treatment" |

## Algorithms

`gm` (per-class Gaussian), `gmm` (Background bimodal, 2 components), `rf`
(RandomForest, parallel), `svm` (RBF, single-threaded, subsamples above 40k).

## Package layout

```
src/vegclassifier/
├── io.py         # tile loading, mask burning, tiling, tile discovery
├── features.py   # HSV, entropy, GLCM contrast, 4-feature stack
├── sampling.py   # non-veg mask, erosion, scope remap, per-tile sampling
├── metrics.py    # metrics, confusion matrix, per-class error coding
├── config.py     # YAML config dataclasses + validation
├── models.py     # 4 algorithms behind one interface + save/load
├── raster.py     # windowed full-image inference (halo, phase alignment)
├── preprocess.py # mask burning + tiling
├── train.py      # training stage
├── predict.py    # prediction stage
├── evaluate.py   # evaluation stage
└── cli.py        # subcommands
```

## Performance

GLCM contrast was 92% of feature-extraction runtime (40,401 `graycomatrix` calls
per 1024x1024 tile). It is now computed by a vectorised route:

> GLCM contrast = SUM (i-j)^2 P(i,j). Since P is the normalised co-occurrence
> count, that sum is exactly the mean squared difference between pixel pairs at
> the offset — so squaring one shifted difference image and taking integral-image
> box sums gives every patch mean in O(1), with no co-occurrence matrix.

*Verified bit-identical* to the original loop (`glcm_contrast_map_reference`,
retained as the oracle): max absolute difference 0.0 at 256/512/1024 px, square
and non-square, across parameter sets, and on whole feature matrices.

| | before | after |
|---|---|---|
| GLCM, 1024x1024 tile | 25.4 s | 0.52 s (~49x) |
| Full feature stack per tile | ~32.4 s | ~3.3 s |
| 116-tile training pass | ~63 min | ~7 min |

Entropy (R=61) is now the largest remaining cost at ~2.6 s/tile.

To re-verify equivalence after any change:

```python
from vegclassifier.features import glcm_contrast_map, glcm_contrast_map_reference
import numpy as np
ch = np.random.default_rng(0).random((256, 256)).astype(np.float32)
assert np.array_equal(glcm_contrast_map(ch), glcm_contrast_map_reference(ch))
```

## Known limitations

Carried from the analysis; see `CAVEATS.md` in the notebook project.

- **RGB only, no NIR.** The Phragmites/FAV confusion is largely a sensor
  limitation, not a modelling one.
- **Tule is concentrated in 6 training tiles.** Tule performance claims are weakly
  supported at the training site.
- **Single date (June 2025).** No phenological variation; generalization to other
  seasons is untested.
- **RandomForest `predict_proba` is not calibrated.** Treat probability rasters as
  relative confidence, not true probabilities, unless validated on held-out data.
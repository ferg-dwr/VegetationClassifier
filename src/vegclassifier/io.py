"""
utils/io.py
===========
Functions for loading imagery, burning multiclass masks, tiling, and
computing dataset-wide statistics.
 
Public API
----------
load_tile(image_path, mask_path, lo, hi)
    Load a GeoTIFF tile and its mask. Normalisation can be per-tile
    (default) or driven by externally computed global statistics, making
    it compatible with SVM and CNN pipelines.
 
burn_multiclass_mask(image_path, output_mask_path, phrag_shp, fav_shp, tule_shp)
    Burns Phragmites (1), FAV (2), and Tule (3) polygons into a single
    uint8 mask aligned to the source image.
 
tile_image_and_mask(image_path, mask_path, output_dir, ..., overlap)
    Tiles a full-resolution image and mask into subtiles with optional
    pixel overlap between adjacent tiles (needed for CNN inference).
 
rebuild_pairs(tile_dir)
    Rebuilds the list of (image_path, mask_path) pairs from disk.
    Use after a kernel restart to avoid rerunning the tiling step.
 
compute_dataset_stats(pairs, sample_n)
    Computes dataset-wide per-channel mean, std, and 2/98 percentiles
    across a list of tile pairs. Use for SVM StandardScaler fitting and
    CNN normalisation layer initialisation.
 
class_balance(pairs, sample_n)
    Compute class pixel counts across a list of (image, mask) pairs.
 
Class map
---------
    0   = background (other vegetation, valid pixel)
    1   = Phragmites
    2   = FAV
    3   = Tule
    255 = nodata
 
Notes
-----
CNN readiness
    - load_tile accepts external lo/hi so all tiles are normalised
      consistently using global dataset statistics rather than per-tile
      percentiles.
    - tile_image_and_mask supports pixel overlap between adjacent tiles
      so boundary pixels are seen in multiple tiles and can be averaged
      at inference time.
    - compute_dataset_stats provides the per-channel statistics needed
      to initialise a CNN normalisation layer (torchvision.transforms.Normalize
      expects per-channel mean and std in [0, 1] space).
"""

import math
from collections import Counter
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window

# ── Type aliases ──────────────────────────────────────────────────────────────
PathLike  = Union[str, Path]
TilePair  = tuple[Path, Path]       # (image_path, mask_path)
MaskArray = np.ndarray              # shape (H, W), dtype uint8
ImgArray  = np.ndarray              # shape (H, W), dtype float32

# ── Class map ─────────────────────────────────────────────────────────────────
CLASS_MAP: dict[str, int] = {
    "background": 0,
    "phragmites": 1,
    "fav":        2,
    "tule":       3,
    "nodata":     255,
}


def load_tile(
        image_path: PathLike,
        mask_path:  PathLike,
        lo:         float | None = None,
        hi:         float | None = None,
        ) -> tuple[ImgArray,
                   ImgArray,
                   ImgArray,
                   ImgArray,
                   ImgArray,
                   MaskArray] :
    """
    Load a GeoTIFF tile and its corresponding mask.
 
    Normalises pixel values to [0, 1] using a 2-98 percentile stretch.
    If lo and hi are provided (e.g. from compute_dataset_stats), they are
    used directly — this ensures consistent normalisation across all tiles,
    which is required for SVM (StandardScaler) and CNN pipelines. If lo
    and hi are not provided, they are computed per-tile, which is the
    default behaviour for classical ML exploration.
 
    Parameters
    ----------
    image_path : PathLike
        Path to the image GeoTIFF tile.
    mask_path : PathLike
        Path to the corresponding mask GeoTIFF tile.
    lo : float or None
        Lower clip value for normalisation. If None, computed as the
        2nd percentile of the current tile.
    hi : float or None
        Upper clip value for normalisation. If None, computed as the
        98th percentile of the current tile.
 
    Returns
    -------
    rgb_img : np.ndarray, shape (H, W, 3), float32
        Normalised RGB image.
    r : np.ndarray, shape (H, W), float32
        Red channel.
    g : np.ndarray, shape (H, W), float32
        Green channel.
    b : np.ndarray, shape (H, W), float32
        Blue channel.
    val : np.ndarray, shape (H, W), float32
        HSV Value channel (= max of r, g, b). Precomputed here since it
        is needed by both features.py and sampling.py.
    mask : np.ndarray, shape (H, W), uint8
        Class mask: 0=background, 1=Phragmites, 2=FAV, 3=Tule, 255=nodata.
    """
    image_path = Path(image_path)
    mask_path  = Path(mask_path)
 
    with rio.open(image_path) as src:
        data = src.read(resampling=Resampling.nearest).astype(np.float32)
 
    data = data[:3]  # keep RGB only, drop alpha if present
 
    # Use provided lo/hi for global normalisation, else compute per-tile
    if lo is None or hi is None:
        lo = float(np.percentile(data, 2))
        hi = float(np.percentile(data, 98))
 
    data = np.clip((data - lo) / (hi - lo + 1e-9), 0.0, 1.0)
 
    r, g, b = data[0], data[1], data[2]
    rgb_img = np.moveaxis(data, 0, -1)
    val     = np.maximum(np.maximum(r, g), b)
 
    with rio.open(mask_path) as src:
        mask = src.read(1)
 
    return rgb_img, r, g, b, val, mask


def burn_multiclass_mask(
            image_path:       PathLike,
            output_mask_path: PathLike,
            phrag_shp:        PathLike,
            fav_shp:          PathLike,
            tule_shp:         PathLike | None = None,
            ) -> MaskArray :
    """
    Burn Phragmites, FAV, and (optionally) Tule polygons into a single
    multiclass uint8 mask aligned to the source image.
 
    Burn priority (highest overwrites lower):
        Phragmites (1) > FAV (2) > Tule (3) > background (0)
 
    Parameters
    ----------
    image_path : PathLike
        Source GeoTIFF to align the mask to.
    output_mask_path : PathLike
        Where to write the output mask GeoTIFF.
    phrag_shp : PathLike
        Path to Phragmites.geojson.
    fav_shp : PathLike
        Path to FAV.geojson.
    tule_shp : PathLike or None
        Path to Tule.geojson. Pass None to skip Tule burning.
 
    Returns
    -------
    burned : np.ndarray, shape (H, W), dtype uint8
        The burned multiclass mask array.
    """
    image_path       = Path(image_path)
    output_mask_path = Path(output_mask_path)

    with rio.open(image_path) as src:
        transform  = src.transform
        crs        = src.crs
        height     = src.height
        width      = src.width
        data       = src.read()
        valid_mask = data.any(axis=0)

    burned = np.zeros((height, width), dtype=np.uint8)

    # Burn in reverse priority — lower priority first so higher overwrites
    layers: list[tuple[PathLike | None, int, str]] = [
        (tule_shp,  CLASS_MAP["tule"],       "Tule"),
        (fav_shp,   CLASS_MAP["fav"],        "FAV"),
        (phrag_shp, CLASS_MAP["phragmites"], "Phragmites"),
        ]

    for shp_path, value, name in layers:
        if shp_path is None:
            continue
        shp_path = Path(shp_path)
        if not shp_path.exists():
            print(f"  WARNING: {shp_path.name} not found — skipping {name}")
            continue

        gdf = gpd.read_file(shp_path)
        if gdf.crs != crs:
            print(f"  Reprojecting {name} from {gdf.crs} → {crs}")
            gdf = gdf.to_crs(crs)

        geoms = [geom for geom in gdf.geometry if geom is not None]
        if not geoms:
            print(f"  WARNING: {name} shapefile has no valid geometries")
            continue

        layer = rasterize(
            shapes=[(geom, value) for geom in geoms],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        burned[layer == value] = value
        print(f"  {name:12s} ({value}): {(burned == value).sum():>12,} px")

    burned[~valid_mask] = CLASS_MAP["nodata"]

    mask_meta = {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "width":     width,
        "height":    height,
        "count":     1,
        "crs":       crs,
        "transform": transform,
        "nodata":    CLASS_MAP["nodata"],
        }
    with rio.open(output_mask_path, "w", **mask_meta) as dst:
        dst.write(burned[np.newaxis, :, :])

    print(f"\nMask written → {output_mask_path}")
    print(f"  Phragmites : {(burned == 1).sum():>12,} px")
    print(f"  FAV        : {(burned == 2).sum():>12,} px")
    print(f"  Tule       : {(burned == 3).sum():>12,} px")
    print(f"  Background : {(burned == 0).sum():>12,} px")
    print(f"  Nodata     : {(burned == 255).sum():>12,} px")

    return burned


def tile_image_and_mask(
            image_path:         PathLike,
            mask_path:          PathLike,
            output_dir:         PathLike,
            tile_size:          int           = 1024,
            overlap:            int           = 0,
            prefix:             str | None = None,
            skip_empty:         bool          = True,
            min_valid_fraction: float         = 0.3,
            image_subdir:       str           = "images",
            mask_subdir:        str           = "masks",
            mask_suffix:        str           = "_mask",
            ) -> list[TilePair] :
    """
    Tile a full-resolution image and its mask into subtiles.
 
    Image tiles -> output_dir/images/<prefix>_rXXX_cXXX.tif
    Mask tiles  -> output_dir/masks/<prefix>_rXXX_cXXX_mask.tif
 
    For CNN inference, set overlap > 0 (e.g. overlap=128) so that boundary
    pixels appear in multiple tiles and their predictions can be averaged,
    reducing edge artefacts. For classical ML, leave overlap=0.
 
    Parameters
    ----------
    image_path : PathLike
        Path to the full-resolution image GeoTIFF.
    mask_path : PathLike
        Path to the full-resolution mask GeoTIFF.
    output_dir : PathLike
        Root output directory. images/ and masks/ subdirs are created
        automatically.
    tile_size : int, default 1024
        Tile size in pixels (square), not including overlap.
    overlap : int, default 0
        Number of pixels of overlap between adjacent tiles on each side.
        A tile with overlap will have actual pixel dimensions of
        (tile_size + 2*overlap). Set to 0 for classical ML, 64-256 for CNN.
    prefix : str or None
        Filename prefix for output tiles. Defaults to the image stem.
    skip_empty : bool, default True
        Skip tiles where all pixel values are zero across all bands.
    min_valid_fraction : float, default 0.3
        Minimum fraction of non-nodata pixels required to write a tile.
        Default 0.3 captures edge tiles containing Tule.
    image_subdir, mask_subdir : str
        Subdirectories of output_dir to write into. Defaults reproduce the
        original flat layout; pass e.g. "rgb_images/middle" and
        "class_masks/middle" for the site-nested production layout.
    mask_suffix : str, default "_mask"
        Appended to the tile stem for mask filenames.
 
    Returns
    -------
    written_pairs : List[Tuple[Path, Path]]
        List of (image_tile_path, mask_tile_path) for every tile written.
    """
    image_path = Path(image_path)
    mask_path  = Path(mask_path)
    output_dir = Path(output_dir)
 
    img_dir = output_dir / image_subdir
    msk_dir = output_dir / mask_subdir
    img_dir.mkdir(exist_ok=True, parents=True)
    msk_dir.mkdir(exist_ok=True, parents=True)
 
    if prefix is None:
        prefix = image_path.stem
 
    written_pairs: list[TilePair] = []
    stride = tile_size  # non-overlapping stride
 
    with rio.open(image_path) as img_src, \
         rio.open(mask_path)  as msk_src:
 
        n_cols = math.ceil(img_src.width  / stride)
        n_rows = math.ceil(img_src.height / stride)
 
        print(f"\nTiling: {image_path.name}")
        print(f"Grid  : {n_rows} rows x {n_cols} cols = {n_rows * n_cols} total tiles")
        print(f"Tile  : {tile_size}px  overlap: {overlap}px")
 
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                # Core tile origin
                x_off = col_idx * stride
                y_off = row_idx * stride
 
                # Expand by overlap, clamp to image bounds
                x_start = max(0, x_off - overlap)
                y_start = max(0, y_off - overlap)
                x_end   = min(img_src.width,  x_off + stride + overlap)
                y_end   = min(img_src.height, y_off + stride + overlap)
 
                width  = x_end - x_start
                height = y_end - y_start
 
                window    = Window(x_start, y_start, width, height)
                transform = img_src.window_transform(window)
 
                img_data = img_src.read(window=window)
                msk_data = msk_src.read(window=window)
 
                if skip_empty and (img_data.size == 0 or not img_data.any()):
                    continue
 
                valid_px = (msk_data[0] != CLASS_MAP["nodata"]).sum()
                if valid_px / (width * height) < min_valid_fraction:
                    continue
 
                img_profile = img_src.profile.copy()
                img_profile.update({
                    "height": height, "width": width, "transform": transform
                })
                img_out = img_dir / f"{prefix}_r{row_idx:03d}_c{col_idx:03d}.tif"
                with rio.open(img_out, "w", **img_profile) as dst:
                    dst.write(img_data)
 
                msk_profile = msk_src.profile.copy()
                msk_profile.update({
                    "height": height, "width": width, "transform": transform
                })
                msk_out = msk_dir / f"{prefix}_r{row_idx:03d}_c{col_idx:03d}{mask_suffix}.tif"
                with rio.open(msk_out, "w", **msk_profile) as dst:
                    dst.write(msk_data)
 
                written_pairs.append((img_out, msk_out))
 
    print(f"Wrote {len(written_pairs)} image/mask pairs for {image_path.name}")
    return written_pairs


def rebuild_pairs(tile_dir: PathLike) -> list[TilePair] :
    """
    Rebuild (image_path, mask_path) pairs from disk after a kernel restart.

    Scans tile_dir/images/ for .tif files and matches each to its
    corresponding mask in tile_dir/masks/.

    Parameters
    ----------
    tile_dir : PathLike
        Root tile directory containing images/ and masks/ subdirectories.

    Returns
    -------
    pairs : List[Tuple[Path, Path]]
        Sorted list of (image_tile_path, mask_tile_path) tuples.
        Only pairs where both image and mask exist are included.
    """
    tile_dir = Path(tile_dir)
    img_dir  = tile_dir / "images"
    msk_dir  = tile_dir / "masks"

    pairs: list[TilePair] = []
    for img_path in sorted(img_dir.glob("*.tif")):
        mask_path = msk_dir / (img_path.stem + "_mask.tif")
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    print(f"Rebuilt {len(pairs)} image/mask pairs from {tile_dir}")
    return pairs


def compute_dataset_stats(
            pairs:    list[TilePair],
            sample_n: int | None = None,
            ) -> dict[str, float] :
    """
    Compute dataset-wide per-channel statistics across all tile pairs.
 
    Reads raw (unnormalised) pixel values and computes per-channel mean,
    std, 2nd percentile, and 98th percentile. These statistics are used
    for consistent normalisation across all tiles — required for SVM
    (StandardScaler) and CNN (Normalize layer) pipelines.
 
    Call this once on your training tiles and save the result. Pass lo/hi
    to load_tile to ensure all tiles (train and test) are normalised
    consistently.
 
    Parameters
    ----------
    pairs : List[Tuple[Path, Path]]
        List of (image_tile_path, mask_tile_path) tuples to compute
        statistics over. Typically your training pairs only.
    sample_n : int or None
        If set, only reads the first N pairs for speed. A sample of
        20-30 tiles is usually sufficient for stable statistics.
        Default None reads all pairs.
 
    Returns
    -------
    stats : Dict[str, float]
        Dictionary with keys:
            'lo'     : float  — 2nd percentile across all sampled pixels
            'hi'     : float  — 98th percentile across all sampled pixels
            'mean_r' : float  — mean of red channel   (after lo/hi normalisation)
            'mean_g' : float  — mean of green channel
            'mean_b' : float  — mean of blue channel
            'std_r'  : float  — std of red channel
            'std_g'  : float  — std of green channel
            'std_b'  : float  — std of blue channel
 
    Notes
    -----
    mean_*/std_* are computed in [0, 1] space (after lo/hi stretch) so they
    can be passed directly to torchvision.transforms.Normalize when building
    a CNN data pipeline.
 
    Example
    -------
    >>> stats = compute_dataset_stats(train_pairs, sample_n=30)
    >>> # Classical ML / SVM: pass lo/hi to load_tile
    >>> rgb, r, g, b, val, mask = load_tile(img, msk, lo=stats['lo'], hi=stats['hi'])
    >>> # CNN: initialise Normalize layer
    >>> transforms.Normalize(
    ...     mean=[stats['mean_r'], stats['mean_g'], stats['mean_b']],
    ...     std=[stats['std_r'],  stats['std_g'],  stats['std_b']],
    ... )
    """
    targets = pairs[:sample_n] if sample_n else pairs
    print(f"Computing dataset statistics over {len(targets)} tiles...")
 
    # Pass 1 — compute global lo/hi from raw pixel values
    all_vals: list[np.ndarray] = []
    for img_path, _ in targets:
        with rio.open(img_path) as src:
            data = src.read().astype(np.float32)[:3]
        all_vals.append(data.ravel())
 
    flat   = np.concatenate(all_vals)
    lo     = float(np.percentile(flat, 2))
    hi     = float(np.percentile(flat, 98))
    del flat, all_vals
 
    # Pass 2 — compute per-channel mean/std in normalised [0, 1] space
    sums:  dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
    sums2: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
    count: int = 0
 
    for img_path, _ in targets:
        with rio.open(img_path) as src:
            data = src.read().astype(np.float32)[:3]
        data = np.clip((data - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        n     = data.shape[1] * data.shape[2]
        count += n
        for c in range(3):
            sums[c]  += float(data[c].sum())
            sums2[c] += float((data[c] ** 2).sum())
 
    means = {c: sums[c]  / count for c in range(3)}
    stds  = {c: np.sqrt(max(sums2[c] / count - means[c] ** 2, 0.0))
             for c in range(3)}
 
    stats: dict[str, float] = {
        "lo":     lo,
        "hi":     hi,
        "mean_r": means[0],
        "mean_g": means[1],
        "mean_b": means[2],
        "std_r":  stds[0],
        "std_g":  stds[1],
        "std_b":  stds[2],
    }
 
    print(f"  lo={lo:.4f}  hi={hi:.4f}")
    print(f"  mean  R={means[0]:.4f}  G={means[1]:.4f}  B={means[2]:.4f}")
    print(f"  std   R={stds[0]:.4f}   G={stds[1]:.4f}   B={stds[2]:.4f}")
 
    return stats

def class_balance(
            pairs:    list[TilePair],
            sample_n: int | None = None,
            ) -> dict[int, int] :
    """
    Compute class pixel counts across a list of (image, mask) tile pairs.

    Parameters
    ----------
    pairs : List[Tuple[Path, Path]]
        List of (image_tile_path, mask_tile_path) tuples.
    sample_n : int or None
        If set, only reads the first N pairs (useful for a quick check
        without scanning all tiles). Default None reads all pairs.

    Returns
    -------
    counts : Dict[int, int]
        Mapping of class value to total pixel count across all sampled tiles.
        Keys: 0 (background), 1 (Phragmites), 2 (FAV), 3 (Tule), 255 (nodata).
    """
    counts:  Counter        = Counter()
    targets: list[TilePair] = pairs[:sample_n] if sample_n else pairs
 
    for _, mask_path in targets:
        with rio.open(mask_path) as src:
            mask = src.read(1)
        for val in CLASS_MAP.values():
            counts[val] += int((mask == val).sum())
 
    total = sum(counts[v] for v in [0, 1, 2, 3])
    print("Class balance:")
    for val, name in [(0, "Background"), (1, "Phragmites"),
                      (2, "FAV"),        (3, "Tule")]:
        n   = counts[val]
        pct = 100 * n / total if total > 0 else 0.0
        print(f"  {name:12s} ({val}): {n:>12,} px  ({pct:.1f}%)")
    print(f"  Nodata     (255): {counts[255]:>12,} px")
    if sample_n:
        print(f"  (sampled from first {sample_n} of {len(pairs)} pairs)")
 
    return dict(counts)

# ── Layout-aware tile discovery ───────────────────────────────────────────────
# rebuild_pairs (above) assumes the flat tile_dir/images + tile_dir/masks layout
# the notebooks wrote. Production data on disk is nested by site:
#
#     WinterIsland_2025_Drone_Tiles/
#     ├── rgb_images/{middle,south}/  <stem>.tif
#     └── class_masks/{middle,south}/ <stem>_mask.tif
#
# discover_pairs handles both, driven by config, so no data has to be moved.

def discover_pairs(
            tile_dir:   PathLike,
            site:       str | None = None,
            image_subdir: str = "rgb_images",
            mask_subdir:  str = "class_masks",
            mask_suffix:  str = "_mask",
            ) -> list[TilePair] :
    """
    Find (image, mask) tile pairs, supporting site-nested layouts.

    Looks for images in  tile_dir/image_subdir[/site]/*.tif  and pairs each with
    tile_dir/mask_subdir[/site]/<stem><mask_suffix>.tif. If the site
    subdirectory does not exist, falls back to a recursive search and filters by
    `site` appearing in the filename.

    Sidecar files (.aux.xml, :Zone.Identifier) are ignored: the glob only
    matches names ending in .tif.

    Parameters
    ----------
    tile_dir : PathLike
        Root tile directory (e.g. .../WinterIsland_2025_Drone_Tiles).
    site : str or None
        Site key, e.g. "middle" or "south". None = all sites.
    image_subdir, mask_subdir : str
        Directory names under tile_dir holding images and masks.
    mask_suffix : str
        Appended to the image stem to form the mask filename.

    Returns
    -------
    pairs : List[Tuple[Path, Path]]
        Sorted (image_path, mask_path). Only pairs where both files exist.

    Raises
    ------
    FileNotFoundError
        If the image directory is missing, or no pairs are found — with a
        message naming the paths actually checked.
    """
    tile_dir = Path(tile_dir)
    img_root = tile_dir / image_subdir
    msk_root = tile_dir / mask_subdir

    if not img_root.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {img_root}\n"
            f"  Checked under tile_dir={tile_dir}\n"
            f"  Set paths.tile_dir and paths.image_subdir in your config."
        )

    # Prefer the site subdirectory; fall back to recursive + filename filter.
    if site and (img_root / site).is_dir():
        img_paths = sorted((img_root / site).glob("*.tif"))
        msk_dir_for = lambda p: msk_root / site
    else:
        img_paths = sorted(img_root.rglob("*.tif"))
        if site:
            img_paths = [p for p in img_paths if site in p.name]
        # mirror the image's relative subpath on the mask side
        msk_dir_for = lambda p: msk_root / p.parent.relative_to(img_root)

    pairs: list[TilePair] = []
    missing_masks: list[str] = []
    for img_path in img_paths:
        mask_path = msk_dir_for(img_path) / f"{img_path.stem}{mask_suffix}.tif"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            missing_masks.append(mask_path.name)

    if not pairs:
        raise FileNotFoundError(
            f"No image/mask tile pairs found.\n"
            f"  images searched : {img_root}{'/' + site if site else ''}"
            f"  ({len(img_paths)} .tif found)\n"
            f"  masks expected  : {msk_root}{'/' + site if site else ''}"
            f"/<stem>{mask_suffix}.tif\n"
            f"  Check paths.image_subdir / paths.mask_subdir / paths.mask_suffix."
        )

    if missing_masks:
        print(f"  WARNING: {len(missing_masks)} image tile(s) had no matching "
              f"mask and were skipped (first: {missing_masks[0]})")

    print(f"Found {len(pairs)} image/mask pairs"
          f"{f' for site={site}' if site else ''} in {tile_dir}")
    return pairs
"""
vegclassifier/preprocess.py
===========================
Preprocessing stage: burn polygon shapefiles into a multiclass mask aligned to
the orthomosaic, then cut image+mask into training tiles.

This is the stage that turns raw deliverables (an orthomosaic + hand-digitised
polygons) into something `train` can consume. Run it once per site.

Burn priority, from io.burn_multiclass_mask:
    Phragmites (1) > FAV (2) > Tule (3) > Background (0)
Pixels outside the flight extent become 255 (nodata).

Ported from 01_preprocessing.
"""

from pathlib import Path

from .config import PipelineConfig
from .io import burn_multiclass_mask, tile_image_and_mask

# Shapefile basenames looked for in paths.data_dir, by class.
DEFAULT_SHAPEFILES: dict[str, str] = {
    "phragmites": "Phragmites",
    "fav":        "FAV",
    "tule":       "Tule",
}
VECTOR_EXTS = (".geojson", ".shp", ".gpkg", ".json")


def find_shapefile(data_dir: Path, basename: str) -> Path | None:
    """
    Locate a vector file by basename, trying common extensions and
    case-insensitive matching. Returns None if not found.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return None
    for ext in VECTOR_EXTS:
        p = data_dir / f"{basename}{ext}"
        if p.exists():
            return p
    target = basename.lower()
    for p in sorted(data_dir.iterdir()):
        if p.suffix.lower() in VECTOR_EXTS and p.stem.lower() == target:
            return p
    return None


def run_preprocess(cfg: PipelineConfig, site: str,
                   image_name: str | None = None,
                   mask_name: str | None = None,
                   tile_size: int = 1024,
                   overlap: int = 0,
                   min_valid_fraction: float = 0.3,
                   skip_burn: bool = False,
                   verbose: bool = True) -> list:
    """
    Burn a multiclass mask for one site and tile it for training.

    Parameters
    ----------
    cfg : PipelineConfig
        Supplies paths (image_dir, data_dir, tile_dir) and tile layout.
    site : str
        Site key, e.g. "middle". Used to locate <site>_winter2025.tif unless
        image_name is given, and to name the tile subdirectory.
    image_name, mask_name : str or None
        Explicit filenames inside paths.image_dir. Defaults to
        "<site>_winter2025.tif" and "<site>_mask.tif".
    tile_size, overlap, min_valid_fraction :
        Tiling parameters. overlap=0 is correct for the per-pixel classifiers
        here; only CNN workflows need overlap.
    skip_burn : bool
        Reuse an existing mask instead of re-burning from shapefiles.

    Returns
    -------
    written_pairs : list of (image_tile, mask_tile)
    """
    image_dir = Path(cfg.paths.image_dir)
    data_dir  = Path(cfg.paths.data_dir)
    tile_dir  = Path(cfg.paths.tile_dir)

    image_path = image_dir / (image_name or f"{site}_winter2025.tif")
    mask_path  = image_dir / (mask_name  or f"{site}_mask.tif")

    if not image_path.exists():
        raise FileNotFoundError(
            f"Source image not found: {image_path}\n"
            f"  Set paths.image_dir, or pass --image-name explicitly.\n"
            f"  Expected a full-resolution orthomosaic for site '{site}'."
        )

    # ── 1. burn mask ─────────────────────────────────────────────────────────
    if skip_burn:
        if not mask_path.exists():
            raise FileNotFoundError(
                f"--skip-burn given but mask not found: {mask_path}"
            )
        if verbose:
            print(f"Reusing existing mask: {mask_path.name}")
    else:
        shp = {k: find_shapefile(data_dir, v) for k, v in DEFAULT_SHAPEFILES.items()}
        if shp["phragmites"] is None and shp["fav"] is None:
            raise FileNotFoundError(
                f"No Phragmites or FAV vector file found in {data_dir}\n"
                f"  Looked for: "
                + ", ".join(f"{v}{{{'|'.join(VECTOR_EXTS)}}}"
                            for v in DEFAULT_SHAPEFILES.values()) + "\n"
                "  Set paths.data_dir, or pass --skip-burn to reuse an "
                "existing mask."
            )
        if verbose:
            print(f"Burning mask for '{site}' from {data_dir}")
            for k, p in shp.items():
                print(f"  {k:<11s} {p.name if p else 'NOT FOUND — skipped'}")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        # burn_multiclass_mask prints its own per-class pixel summary
        burn_multiclass_mask(
            image_path, mask_path,
            phrag_shp=shp["phragmites"], fav_shp=shp["fav"], tule_shp=shp["tule"],
        )

    # ── 2. tile into the configured layout ───────────────────────────────────
    img_sub = f"{cfg.paths.image_subdir}/{site}"
    msk_sub = f"{cfg.paths.mask_subdir}/{site}"
    if verbose:
        print(f"\nTiling into {tile_dir}/{img_sub} and /{msk_sub}")

    pairs = tile_image_and_mask(
        image_path, mask_path, tile_dir,
        tile_size=tile_size, overlap=overlap,
        skip_empty=True, min_valid_fraction=min_valid_fraction,
        image_subdir=img_sub, mask_subdir=msk_sub,
        mask_suffix=cfg.paths.mask_suffix,
    )
    if verbose:
        print(f"\nDone. {len(pairs)} tile pairs written for site '{site}'.")
        print("  Next: preprocess any remaining sites, then "
              "vegclassifier train --config <config>  (trains on 'middle' by default)")
    return pairs
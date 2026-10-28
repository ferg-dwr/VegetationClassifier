"""
vegclassifier/config.py
=======================
Typed configuration for the pipeline, loaded from YAML. One config file fully
defines an experiment (paths, normalization, features, sampling, model), so a
run is reproducible from the committed YAML alone.

Defaults are locked to 03_B cell 1. Normalization defaults to identity
(lo=0, hi=255), the choice validated in 02_B; it can be overridden per the
`normalization` block.
"""

from dataclasses import asdict, dataclass, field

import yaml

VALID_NORM   = ("identity", "global", "per_tile")
VALID_SCOPES = ("binary", "multiclass", "merged")
VALID_ALGOS  = ("gm", "gmm", "rf", "svm")


@dataclass
class Normalization:
    mode: str = "identity"          # identity | global | per_tile
    lo: float | None = 0.0       # used by identity; for global, filled at train time
    hi: float | None = 255.0


@dataclass
class Features:
    entropy_r: int = 61
    glcm_patch_size: int = 21
    glcm_levels: int = 64
    glcm_distances: list[int] = field(default_factory=lambda: [1, 3, 5])
    nonveg_val_thresh: float = 0.35
    nonveg_sat_thresh: float = 0.35


@dataclass
class Sampling:
    max_px_per_tile: int = 2000
    erode_px: int = 10
    seed: int = 42


@dataclass
class Model:
    algo: str = "rf"                # gm | gmm | rf | svm
    # RandomForest
    rf_n_estimators: int = 300
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 5
    rf_max_features: str = "sqrt"
    rf_class_weight: str = "balanced"
    # SVM
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    svm_max_train: int = 40000
    svm_class_weight: str = "balanced"
    # GMM
    gmm_bg_components: int = 2       # BG gets 2 components, others 1


@dataclass
class Paths:
    tile_dir: str = "data/tiles"        # root holding the tiled imagery
    image_dir: str = "data/imagery"     # full-resolution orthomosaics
    data_dir: str = "data/shapefiles"
    output_dir: str = "models"
    # Tile layout. Defaults match the production layout on disk:
    #   <tile_dir>/rgb_images/<site>/<stem>.tif
    #   <tile_dir>/class_masks/<site>/<stem>_mask.tif
    image_subdir: str = "rgb_images"
    mask_subdir: str = "class_masks"
    mask_suffix: str = "_mask"


@dataclass
class PipelineConfig:
    scope: str = "multiclass"
    paths: Paths = field(default_factory=Paths)
    normalization: Normalization = field(default_factory=Normalization)
    features: Features = field(default_factory=Features)
    sampling: Sampling = field(default_factory=Sampling)
    model: Model = field(default_factory=Model)

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls(
            scope=raw.get("scope", "multiclass"),
            paths=Paths(**raw.get("paths", {})),
            normalization=Normalization(**raw.get("normalization", {})),
            features=Features(**raw.get("features", {})),
            sampling=Sampling(**raw.get("sampling", {})),
            model=Model(**raw.get("model", {})),
        )
        cfg.validate()
        return cfg

    def to_yaml(self, path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, default_flow_style=False)

    # ── validation — loud, specific errors (not tracebacks) ───────────────────
    def validate(self) -> None:
        errs = []
        if self.scope not in VALID_SCOPES:
            errs.append(f"scope '{self.scope}' invalid; must be one of {VALID_SCOPES}")
        if self.normalization.mode not in VALID_NORM:
            errs.append(f"normalization.mode '{self.normalization.mode}' invalid; "
                        f"must be one of {VALID_NORM}")
        if self.model.algo not in VALID_ALGOS:
            errs.append(f"model.algo '{self.model.algo}' invalid; must be one of {VALID_ALGOS}")
        if self.normalization.mode == "identity":
            if self.normalization.lo is None or self.normalization.hi is None:
                errs.append("normalization.mode=identity requires lo and hi (e.g. 0 and 255)")
        if self.sampling.max_px_per_tile <= 0:
            errs.append("sampling.max_px_per_tile must be > 0")
        if self.sampling.erode_px < 0:
            errs.append("sampling.erode_px must be >= 0")
        if errs:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errs))

    # ── helpers consumed by the feature builder ───────────────────────────────
    def feature_dict(self) -> dict:
        return {
            "entropy_r":       self.features.entropy_r,
            "glcm_patch_size": self.features.glcm_patch_size,
            "glcm_levels":     self.features.glcm_levels,
            "glcm_distances":  self.features.glcm_distances,
        }

    def norm_lohi(self):
        """Return (lo, hi) for load_tile. per_tile -> (None, None)."""
        if self.normalization.mode == "per_tile":
            return None, None
        return self.normalization.lo, self.normalization.hi
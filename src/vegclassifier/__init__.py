"""
vegclassifier — Winter Island invasive vegetation classification pipeline.

Foundation layer: io, features, sampling, metrics, config, models.
Orchestration (preprocess/train/predict/evaluate/cli) builds on these.
"""

__version__ = "0.1.0"

from . import (
    config,
    evaluate,
    features,
    io,
    metrics,
    models,
    predict,
    preprocess,
    raster,
    sampling,
    train,
)

__all__ = ["config", "evaluate", "features", "io", "metrics", "models", "predict", "preprocess", "raster", "sampling", "train"]

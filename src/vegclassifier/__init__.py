"""
vegclassifier — Winter Island invasive vegetation classification pipeline.

Foundation layer: io, features, sampling, metrics, config, models.
Orchestration (preprocess/train/predict/evaluate/cli) builds on these.
"""

__version__ = "0.1.0"

from . import io, features, sampling, metrics, config, models, raster, preprocess, train, predict, evaluate

__all__ = ["io", "features", "sampling", "metrics", "config", "models", "raster", "preprocess", "train", "predict", "evaluate"]

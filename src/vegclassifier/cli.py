"""
vegclassifier/cli.py
====================
Command-line entrypoint. Every stage is a subcommand driven by one YAML config,
so a run is reproducible from the committed config alone.

    vegclassifier init-config  configs/rf_multiclass.yaml
    vegclassifier train        --config configs/rf_multiclass.yaml
    vegclassifier predict      --config ... --model ... --image ...
    vegclassifier evaluate     --config ... --model ... --image ... --mask ...

Errors are reported as one clear sentence, not a library traceback.
"""

import argparse
import json
import sys
from pathlib import Path


def _fail(msg: str, code: int = 1):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(code)


def _load_cfg(path):
    from .config import PipelineConfig
    p = Path(path)
    if not p.exists():
        _fail(f"Config file not found: {p}\n"
              f"  Create a starter with: vegclassifier init-config {p}")
    try:
        return PipelineConfig.from_yaml(p)
    except ValueError as e:
        _fail(str(e))
    except Exception as e:
        _fail(f"Could not parse {p}: {e}")


def cmd_init_config(args):
    from .config import PipelineConfig
    out = Path(args.path)
    if out.exists() and not args.force:
        _fail(f"{out} already exists. Use --force to overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)
    PipelineConfig().to_yaml(out)
    print(f"Wrote starter config -> {out}")
    print("Edit paths.tile_dir / paths.output_dir, then run: "
          f"vegclassifier train --config {out}")


def cmd_train(args):
    from .train import run_train
    cfg = _load_cfg(args.config)
    if args.scope:
        cfg.scope = args.scope
    if args.algo:
        cfg.model.algo = args.algo
    cfg.validate()
    try:
        _, path = run_train(cfg, train_site=args.site, out_path=args.out)
    except (FileNotFoundError, ValueError) as e:
        _fail(str(e))
    print(f"\nDone. Model: {path}")


def cmd_predict(args):
    from .predict import run_predict
    model = Path(args.model)
    image = Path(args.image)
    if not model.exists():
        _fail(f"Model not found: {model}. Run 'vegclassifier train' first.")
    if not image.exists():
        _fail(f"Image not found: {image}")
    out_dir = Path(args.out) if args.out else Path("outputs/predictions")
    try:
        written = run_predict(model, image, out_dir, block=args.block,
                              save_probabilities=not args.no_prob)
    except (ValueError, OSError) as e:
        _fail(str(e))
    print("\nDone. Wrote:")
    for k, v in written.items():
        print(f"  {k:<24s} {v}")


def cmd_evaluate(args):
    from .evaluate import run_evaluate
    model, image, mask = Path(args.model), Path(args.image), Path(args.mask)
    for label, p in (("Model", model), ("Image", image), ("Mask", mask)):
        if not p.exists():
            _fail(f"{label} not found: {p}")
    out_dir = Path(args.out) if args.out else Path("outputs/evaluation")
    try:
        res = run_evaluate(model, image, mask, out_dir,
                           erode_px=args.erode, block=args.block,
                           write_error_maps=not args.no_error_maps)
    except (ValueError, OSError) as e:
        _fail(str(e))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  metrics JSON -> {args.json}")
    print("\nDone.")


def build_parser():
    p = argparse.ArgumentParser(
        prog="vegclassifier",
        description="Invasive wetland vegetation classification from RGB drone imagery.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("init-config", help="write a starter YAML config")
    c.add_argument("path")
    c.add_argument("--force", action="store_true")
    c.set_defaults(func=cmd_init_config)

    t = sub.add_parser("train", help="train a model on the training tiles")
    t.add_argument("--config", required=True)
    t.add_argument("--scope", choices=["binary", "multiclass", "merged"])
    t.add_argument("--algo", choices=["gm", "gmm", "rf", "svm"])
    t.add_argument("--site", default="middle", help="tile filename key (default: middle)")
    t.add_argument("--out", help="output .joblib path")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="apply a model to a full-resolution image")
    pr.add_argument("--model", required=True)
    pr.add_argument("--image", required=True)
    pr.add_argument("--out")
    pr.add_argument("--block", type=int, default=1024)
    pr.add_argument("--no-prob", action="store_true", help="skip probability rasters")
    pr.set_defaults(func=cmd_predict)

    e = sub.add_parser("evaluate", help="score a model against ground truth")
    e.add_argument("--model", required=True)
    e.add_argument("--image", required=True)
    e.add_argument("--mask", required=True)
    e.add_argument("--out")
    e.add_argument("--erode", type=int, default=10)
    e.add_argument("--block", type=int, default=1024)
    e.add_argument("--json", help="write metrics to this JSON path")
    e.add_argument("--no-error-maps", action="store_true")
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

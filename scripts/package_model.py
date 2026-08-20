"""Copy the trained model artifact into the installable package tree.

pyproject.toml declares ``shipment_weight/models/*.joblib`` as package data,
but scripts/train_real_data.py always saves to the top-level models/ dir (so
local dev and the repo-relative fallback in
shipment_weight.predict.ShipmentWeightPredictor.load() keep working
regardless of packaging). This script bridges the two: run it before
``python -m build`` so the built wheel/sdist actually contains the model.

There's no separate sidecar file to copy -- category_error_map and
category_error_global_mean are baked into the single joblib bundle dict by
scripts/train_real_data.py (see MODEL_CARD.md's Serving section), so
copying models/model.joblib is the whole job.

This is a deliberate manual step, not an automatic build hook. See
CONTRIBUTING.md's Packaging section for why.

Usage:
    python scripts/package_model.py
    python scripts/package_model.py --model path/to/other_bundle.joblib
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_MODELS_DIR = REPO_ROOT / "src" / "shipment_weight" / "models"
PACKAGED_MODEL_NAME = "model.joblib"  # must match _default_model_path() in predict.py


def package_model(source: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(
            f"No model artifact at {source} -- train one first (scripts/train_real_data.py)."
        )
    PACKAGE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Always named model.joblib in the package tree, regardless of the
    # source filename, since that's the exact name predict.py looks for.
    dest = PACKAGE_MODELS_DIR / PACKAGED_MODEL_NAME
    shutil.copy2(source, dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "models" / "model.joblib"),
        help="Path to the trained model bundle to package (default: models/model.joblib)",
    )
    args = parser.parse_args()

    try:
        dest = package_model(Path(args.model))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Copied {args.model} -> {dest}")
    print("Now run: python -m build")


if __name__ == "__main__":
    main()

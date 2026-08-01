"""Shared, cached data prep for the accuracy experiments.

Reading the two Excel exports takes ~1-2 minutes, and both
``analyze_residuals.py`` and ``experiment_loss_and_interactions.py`` need
the exact same prepared frames. This module runs the unmodified
train_real_data pipeline (load -> clean -> build_features ->
add_derived_features -> time_split -> encode_category_error) once and
caches the result, so repeated experiment runs are instant.

Nothing here changes the pipeline: every stage is imported from
train_real_data / shipment_weight.features and called in the same order
with the same arguments as the production training script.
"""
from __future__ import annotations

import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from shipment_weight.features import add_derived_features

from train_real_data import (
    build_features,
    clean,
    encode_category_error,
    load_data,
    time_split,
)

DEFAULT_SHIPMENTS = "order_shipments_anonymized.xlsx"
DEFAULT_LINES = "order_lines_in_shipment_anonymized.xlsx"


def _default_cache_dir() -> str:
    scratch = os.environ.get("CLAUDE_SCRATCHPAD")
    if scratch:
        return scratch
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "_cache")


def prepare(
    shipments_path: str = DEFAULT_SHIPMENTS,
    lines_path: str = DEFAULT_LINES,
    cache_path: str | None = None,
    refresh: bool = False,
    with_lines: bool = False,
):
    """Return ``(train, test)`` -- or ``(train, test, lines_raw)`` when
    ``with_lines`` -- prepared exactly as scripts/train_real_data.py does.

    ``lines_raw`` is only returned on request because it is the large
    frame; it is cached alongside the split frames so the per-SKU
    diagnostics do not force a second Excel read.
    """
    if cache_path is None:
        cache_path = os.path.join(_default_cache_dir(), "prepared_split.joblib")

    if not refresh and os.path.exists(cache_path):
        cached = joblib.load(cache_path)
        print(f"[prep] loaded cached split from {cache_path}")
        print(f"[prep] train={len(cached['train']):,} rows  test={len(cached['test']):,} rows")
        if with_lines:
            return cached["train"], cached["test"], cached["lines"]
        return cached["train"], cached["test"]

    ships_raw, lines_raw = load_data(shipments_path, lines_path)
    ships, _ = clean(ships_raw)
    df = build_features(ships, lines_raw)
    df = add_derived_features(df)
    train, test = time_split(df)
    train, test = encode_category_error(train, test)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    joblib.dump({"train": train, "test": test, "lines": lines_raw}, cache_path, compress=3)
    print(f"\n[prep] cached prepared split to {cache_path}")

    if with_lines:
        return train, test, lines_raw
    return train, test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build/refresh the experiment data cache.")
    parser.add_argument("--shipments", default=DEFAULT_SHIPMENTS)
    parser.add_argument("--lines", default=DEFAULT_LINES)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    train, test, lines = prepare(args.shipments, args.lines, refresh=args.refresh, with_lines=True)
    print("\nShipment-frame columns:")
    print(sorted(train.columns.tolist()))
    print("\nOrder-line columns:")
    print(sorted(lines.columns.tolist()))
    with pd.option_context("display.width", 200):
        print("\nTrain date range:", train["order_date"].min(), "->", train["order_date"].max())
        print("Test  date range:", test["order_date"].min(), "->", test["order_date"].max())

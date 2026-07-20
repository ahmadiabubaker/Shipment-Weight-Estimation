"""Demo: run the trained model on a 20-shipment sample from June 2026.

Usage:
    python scripts/demo_real_data.py \\
        --shipments order_shipments_anonymized.xlsx \\
        --lines     order_lines_in_shipment_anonymized.xlsx \\
        [--model    models/model.joblib]
        [--seed     42]
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shipment_weight.features import ALL_FEATURES, add_derived_features

# Re-use every pipeline stage from the training script without copy-pasting.
from train_real_data import (
    LBS_TO_OZ,
    build_features,
    clean,
    encode_category_error,
    load_data,
    time_split,
)

SAMPLE_N = 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo predictions on a June 2026 test sample."
    )
    parser.add_argument("--shipments", required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Reproduce the exact same pipeline as training ─────────────────────────
    ships_raw, lines_raw = load_data(args.shipments, args.lines)
    ships, _ = clean(ships_raw)
    df = build_features(ships, lines_raw)
    train, test = time_split(df)
    _, test = encode_category_error(train, test)

    test = add_derived_features(test)

    # ── Load model ────────────────────────────────────────────────────────────
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    model_type = bundle.get("model_type", "unknown")
    model_version = bundle.get("model_version", "unknown")

    # ── Sample 20 rows ────────────────────────────────────────────────────────
    sample = test.sample(n=min(SAMPLE_N, len(test)), random_state=args.seed).reset_index(drop=True)

    preds_oz = pipeline.predict(sample[ALL_FEATURES])

    # ── Build display table ───────────────────────────────────────────────────
    result = pd.DataFrame({
        "shipment_number":        sample["shipment_number"],
        "theoretical_weight_lbs": sample["theoretical_weight_oz"] / LBS_TO_OZ,
        "predicted_weight_lbs":   preds_oz / LBS_TO_OZ,
        "actual_weight_lbs":      sample["actual_weight_oz"] / LBS_TO_OZ,
    })
    result["error_theoretical_lbs"] = result["theoretical_weight_lbs"] - result["actual_weight_lbs"]
    result["error_model_lbs"]        = result["predicted_weight_lbs"]   - result["actual_weight_lbs"]
    result["_abs_model_err"]         = result["error_model_lbs"].abs()

    result = result.sort_values("_abs_model_err").drop(columns="_abs_model_err").reset_index(drop=True)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\nModel : {model_type}  ({model_version})")
    print(f"Sample: {len(result)} shipments from June 2026  (seed={args.seed})\n")

    col_w = {
        "shipment_number":        16,
        "theoretical_weight_lbs": 22,
        "predicted_weight_lbs":   21,
        "actual_weight_lbs":      18,
        "error_theoretical_lbs":  23,
        "error_model_lbs":        16,
    }
    headers = {
        "shipment_number":        "shipment_number",
        "theoretical_weight_lbs": "theoretical_wt_lbs",
        "predicted_weight_lbs":   "predicted_wt_lbs",
        "actual_weight_lbs":      "actual_wt_lbs",
        "error_theoretical_lbs":  "error_theoretical_lbs",
        "error_model_lbs":        "error_model_lbs",
    }

    header_row = "".join(h.rjust(col_w[c]) for c, h in headers.items())
    print(header_row)
    print("-" * len(header_row))

    for _, row in result.iterrows():
        line = (
            f"{str(row['shipment_number']):>{col_w['shipment_number']}}"
            f"{row['theoretical_weight_lbs']:>{col_w['theoretical_weight_lbs']}.3f}"
            f"{row['predicted_weight_lbs']:>{col_w['predicted_weight_lbs']}.3f}"
            f"{row['actual_weight_lbs']:>{col_w['actual_weight_lbs']}.3f}"
            f"{row['error_theoretical_lbs']:>{col_w['error_theoretical_lbs']}.3f}"
            f"{row['error_model_lbs']:>{col_w['error_model_lbs']}.3f}"
        )
        print(line)

    # ── Summary ───────────────────────────────────────────────────────────────
    mae_theoretical = result["error_theoretical_lbs"].abs().mean()
    mae_model       = result["error_model_lbs"].abs().mean()
    print("-" * len(header_row))
    print(
        f"\n  Avg absolute error — theoretical: {mae_theoretical:.3f} lbs  |  "
        f"model: {mae_model:.3f} lbs  "
        f"({'better' if mae_model < mae_theoretical else 'worse'} by "
        f"{abs(mae_theoretical - mae_model):.3f} lbs)"
    )


if __name__ == "__main__":
    main()

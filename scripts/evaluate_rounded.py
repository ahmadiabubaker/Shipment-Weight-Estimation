"""Evaluate predictions after rounding to the nearest whole-pound billing
tier, side by side with the existing continuous-prediction metrics.

Tomas asked us to round predictions to the nearest pound before measuring
accuracy, since delivery carriers round actual shipment weight the same way
before billing/labeling -- the comparison should reflect real-world
billing-tier matching, not raw continuous error.

Rounding rule (verified against real data before implementing):
  - actual_weight_lbs >= 1 lb: 80-90%+ of values are exact whole numbers in
    every weight bucket (99% in the 50-100 lb range) -- strong evidence
    carriers round to the nearest pound here. Predictions >= 1 lb are
    rounded to match.
  - actual_weight_lbs < 1 lb: only ~20% of values land on an exact
    quarter-pound (.25/.5/.75); real examples include 0.51, 0.55, 0.78,
    0.98, which aren't quarter-pound multiples. No quarter-pound rounding
    rule holds here, so predictions < 1 lb are left unrounded. This bucket
    is ~1.3% of the dataset (842 of 66,224 rows) -- reported below, not
    silently dropped.

This is evaluation-only: no model is retrained or modified. Reuses the
exact same load/clean/build_features/time_split/encode_category_error
pipeline as scripts/train_real_data.py (via scripts/_experiment_prep.py)
so the test split and features here are identical to what each model was
actually trained/evaluated on.

Usage:
    python scripts/evaluate_rounded.py
    python scripts/evaluate_rounded.py --shipments ... --lines ... --refresh
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from shipment_weight.evaluate import regression_metrics
from shipment_weight.features_extended import (
    ALL_FEATURES_EXTENDED,
    add_extended_features,
    add_target_encodings,
    finalize_train_fold_stats,
)
from shipment_weight.rounding import round_to_billing_tier

from _experiment_prep import DEFAULT_LINES, DEFAULT_SHIPMENTS, prepare
from train_real_data import LBS_TO_OZ, _hr

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROD_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model.joblib")
EXTENDED_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model_extended.joblib")


def rounded_metrics(actual_lbs: np.ndarray, rounded_pred_lbs: np.ndarray) -> dict[str, float]:
    err = rounded_pred_lbs - actual_lbs
    abs_err = np.abs(err)
    return {
        "mae_lbs": float(np.mean(abs_err)),
        "rmse_lbs": float(np.sqrt(np.mean(err ** 2))),
        "bias_lbs": float(np.mean(err)),
        "within_1lb_pct": float(np.mean(abs_err <= 1.0)) * 100,
        "exact_match_pct": float(np.mean(np.isclose(rounded_pred_lbs, actual_lbs, atol=1e-6))) * 100,
    }


def evaluate_model(
    label: str,
    bundle_path: str,
    X_test: pd.DataFrame,
    actual_oz: pd.Series,
    actual_lbs: pd.Series,
) -> None:
    if not os.path.isfile(bundle_path):
        print(f"  [skip] {label}: no model artifact at {bundle_path}")
        return

    bundle = joblib.load(bundle_path)
    pipeline = bundle["pipeline"]
    preds_oz = pipeline.predict(X_test)
    pred_lbs = preds_oz / LBS_TO_OZ

    cont = regression_metrics(actual_oz, preds_oz)
    rounded, sub_1lb = round_to_billing_tier(pred_lbs)
    rnd = rounded_metrics(actual_lbs.values, rounded)

    _hr(f"{label}\n  model_type={bundle.get('model_type', 'unknown')}  "
        f"model_version={bundle.get('model_version', 'unknown')}")
    print(f"  Test rows                          : {len(X_test):,}")
    print(f"  Sub-1lb predictions (left unrounded): {int(sub_1lb.sum()):,} "
          f"({sub_1lb.mean()*100:.2f}% of test rows)")
    print()
    print(f"  Continuous model MAE                : {cont['mae_oz']/LBS_TO_OZ:>7.3f} lbs   "
          f"(existing metric -- unrounded predictions)")
    print(f"  Rounded-to-billing-tier MAE         : {rnd['mae_lbs']:>7.3f} lbs   "
          f"(same predictions, rounded like a carrier would)")
    print(f"  Rounded-to-billing-tier RMSE        : {rnd['rmse_lbs']:>7.3f} lbs")
    print(f"  Rounded-to-billing-tier bias        : {rnd['bias_lbs']:>+7.3f} lbs")
    print(f"  Rounded-to-billing-tier within-1lb% : {rnd['within_1lb_pct']:>6.1f}%")
    print()
    print(f"  >>> Pre-printed label would exactly match the carrier's charge: "
          f"{rnd['exact_match_pct']:.1f}% of test shipments")


def build_extended_test_frame(
    train: pd.DataFrame, test: pd.DataFrame, lines_raw: pd.DataFrame
) -> pd.DataFrame:
    """Mirror scripts/train_real_data_extended.py's feature order (extended
    features + target encodings computed pre-split on the full chronological
    frame, train-fold stats finalized post-split) starting from the already
    split/category-encoded frames scripts/_experiment_prep.py provides.
    Safe to reorder relative to encode_category_error since the two touch
    disjoint columns."""
    df_full = pd.concat([train, test], ignore_index=True)
    df_full = add_extended_features(df_full, lines_raw)
    df_full = add_target_encodings(df_full)
    train_ext = df_full[df_full["order_date"].dt.month <= 5].reset_index(drop=True)
    test_ext = df_full[df_full["order_date"].dt.month == 6].reset_index(drop=True)
    _, test_ext = finalize_train_fold_stats(train_ext, test_ext)
    return test_ext


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shipments", default=DEFAULT_SHIPMENTS)
    parser.add_argument("--lines", default=DEFAULT_LINES)
    parser.add_argument("--refresh", action="store_true",
                        help="Bypass the cached train/test split from _experiment_prep")
    args = parser.parse_args()

    train, test, lines_raw = prepare(args.shipments, args.lines, refresh=args.refresh, with_lines=True)

    _hr("EVALUATE: PREDICTIONS ROUNDED TO NEAREST BILLING-TIER POUND")
    print("  Rounding predictions >= 1 lb to the nearest whole pound before scoring,")
    print("  since carriers round actual weight the same way before billing (verified")
    print("  against real data -- see module docstring). This is NOT a new/better model;")
    print("  it's the same predictions evaluated through a rounding step that mirrors")
    print("  carrier behavior.")

    bundle = joblib.load(PROD_MODEL_PATH)
    X_test_prod = test[bundle["feature_list"]]
    evaluate_model(
        "PRODUCTION MODEL  (models/model.joblib)  <-- headline numbers",
        PROD_MODEL_PATH, X_test_prod, test["actual_weight_oz"], test["actual_weight_lbs"],
    )

    if os.path.isfile(EXTENDED_MODEL_PATH):
        test_ext = build_extended_test_frame(train, test, lines_raw)
        X_test_ext = test_ext[ALL_FEATURES_EXTENDED]
        evaluate_model(
            "EXTENDED MODEL  (models/model_extended.joblib)  -- comparison only, not production",
            EXTENDED_MODEL_PATH, X_test_ext, test_ext["actual_weight_oz"], test_ext["actual_weight_lbs"],
        )
    else:
        print(f"\n  [skip] extended model: no artifact at {EXTENDED_MODEL_PATH}")

    _hr("DONE")


if __name__ == "__main__":
    main()

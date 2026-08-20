"""Diagnose HistGBT (absolute-error, untuned) against Ridge's known weak
spots, reproducing the exact three breakdowns train_real_data.py's Stage 7
already produces for Ridge -- bias by ship_method, bias by box_name, largest
individual errors -- so the two models are compared like-for-like on WHERE
they're weak, not just on average error.

Context: scripts/evaluate_model_sweep.py already established that HistGBT
beats Ridge on average (0.643 vs 0.709 lbs continuous MAE) and that 4
boosting libraries converge within 0.28 oz of each other after tuning -- the
aggregate model-selection question is effectively settled. Before HistGBT is
even considered for production it needs the same scrutiny Ridge got, because
a lower average can hide the same (or new) trouble spots. Ridge's known weak
spots, per train_real_data.py's Stage 7:
  - FedEx HAZMAT: under-predicts by +10.77 oz on average
  - box_name 30x20x12: 29.2 oz MAE, dominates the largest-errors list
  - error grows with item_count

This script independently recomputes Ridge's own breakdown from the
production artifact on the current June test set (rather than trusting the
numbers above as given) so the comparison is apples-to-apples against
whatever the data currently says, not a stale run.

Deliberately self-contained: does NOT import from evaluate_model_sweep.py,
evaluate_ensemble.py, evaluate_rounded.py, or experiment_loss_and_
interactions.py, so this diagnostic can't be broken by (or break) another
exploratory script. It DOES reuse the actual production building blocks --
shipment_weight.ingest's build_features (via train_real_data.py's load/
clean/time_split/encode_category_error, cached by scripts/_experiment_prep.py)
and shipment_weight.evaluate's existing bias_by_segment /
bias_by_item_count_bucket / largest_errors / regression_metrics functions --
the exact same functions and June test rows Stage 7 already uses for Ridge.

Diagnostic-only: no model artifact is saved. models/, every existing script,
and everything under src/shipment_weight/ are read-only.

Usage:
    python scripts/evaluate_histgbt_diagnostics.py
    python scripts/evaluate_histgbt_diagnostics.py --refresh
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from shipment_weight.evaluate import (
    bias_by_item_count_bucket,
    bias_by_segment,
    largest_errors,
    regression_metrics,
)
from shipment_weight.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

from _experiment_prep import DEFAULT_LINES, DEFAULT_SHIPMENTS, prepare
from train_real_data import LBS_TO_OZ, _hr

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROD_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model.joblib")

# Same untuned absolute-error config validated in scripts/evaluate_model_sweep.py
# (that sweep found tuning bought ~0.00 oz over these defaults, so the
# untuned baseline is what's actually being diagnosed for production fitness).
HGB_KWARGS = dict(
    loss="absolute_error", max_iter=400, learning_rate=0.05,
    max_depth=None, min_samples_leaf=40, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.1, random_state=42,
)

# The specific trouble spots flagged for Ridge -- checked verbatim against
# this June test set's carton_type/ship_method values before writing this.
WATCH_SHIP_METHODS = ["FedEx HAZMAT"]
WATCH_BOX_NAMES = ["30x20x12", "20x14x12"]

SIMILAR_MAE_THRESHOLD_OZ = 1.0  # below this delta, call the two models "similar" on a segment


def build_histgbt_pipeline() -> Pipeline:
    """Local, self-contained pipeline builder -- not imported from any other
    script. HistGBT can't consume a sparse matrix, so the one-hot block is
    densified (sparse_threshold=0.0); StandardScaler on the numeric side is
    a no-op for tree splits (monotonic per-feature transform) but kept so
    this matches exactly what was benchmarked in the model sweep."""
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    pre = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(steps=[("preprocess", pre), ("model", HistGradientBoostingRegressor(**HGB_KWARGS))])


def full_breakdown(label: str, y_test: pd.Series, preds: np.ndarray, test: pd.DataFrame) -> dict:
    """Reproduce train_real_data.py Stage 7's three breakdowns for one
    model's predictions on the June test set."""
    m = regression_metrics(y_test, preds)
    ship_bias = bias_by_segment(y_test, preds, test["ship_method"].reset_index(drop=True), "ship_method")
    box_bias = bias_by_segment(y_test, preds, test["carton_type"].reset_index(drop=True), "box_name")
    item_bias = bias_by_item_count_bucket(y_test, preds, test["item_count"].reset_index(drop=True))

    context_cols = ["theoretical_weight_oz", "carton_type", "ship_method", "item_count", "category_mode"]
    X_ctx = test[ALL_FEATURES].copy().reset_index(drop=True)
    for col in context_cols:
        if col not in X_ctx.columns and col in test.columns:
            X_ctx[col] = test[col].reset_index(drop=True).values
    big_err = largest_errors(X_ctx, y_test.reset_index(drop=True), preds)

    print(f"\n  [{label}]  overall MAE {m['mae_oz']:.2f} oz ({m['mae_oz']/LBS_TO_OZ:.3f} lbs)   "
          f"bias {m['bias_oz']:+.2f} oz   within-1lb {m['within_1lb_pct']:.1f}%")

    print(f"\n  -- Bias by SHIP METHOD [{label}] --")
    print(ship_bias.to_string(index=False))

    print(f"\n  -- Bias by BOX NAME (top 20 by row count) [{label}] --")
    print(box_bias.head(20).to_string(index=False))

    print(f"\n  -- Bias by ITEM COUNT BUCKET [{label}] --")
    print(item_bias.to_string(index=False))

    print(f"\n  -- LARGEST ERRORS (top 20) [{label}] --")
    display_cols = [c for c in [
        "theoretical_weight_oz", "carton_type", "ship_method", "item_count",
        "actual_weight_oz", "predicted_weight_oz", "error_oz", "abs_error_oz",
    ] if c in big_err.columns]
    print(big_err[display_cols].to_string(index=False))

    return {"metrics": m, "ship_bias": ship_bias, "box_bias": box_bias, "item_bias": item_bias, "largest_errors": big_err}


def flag(ridge_mae: float, hgb_mae: float) -> str:
    delta = hgb_mae - ridge_mae
    if abs(delta) < SIMILAR_MAE_THRESHOLD_OZ:
        return "similar"
    return "BETTER (HistGBT)" if delta < 0 else "WORSE (HistGBT)"


def compare_segments(label: str, ridge_tbl: pd.DataFrame, hgb_tbl: pd.DataFrame,
                     seg_col: str, levels: list[str]) -> None:
    print(f"\n  -- {label}: Ridge vs HistGBT --")
    header = f"  {'segment':<22} {'count':>7} {'ridge_bias':>11} {'ridge_mae':>10} {'hgb_bias':>10} {'hgb_mae':>9}   flag"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lvl in levels:
        r = ridge_tbl[ridge_tbl[seg_col] == lvl]
        h = hgb_tbl[hgb_tbl[seg_col] == lvl]
        if r.empty or h.empty:
            print(f"  {lvl:<22}  [not found in one of the tables -- check spelling/level presence]")
            continue
        r, h = r.iloc[0], h.iloc[0]
        f = flag(r["mae_oz"], h["mae_oz"])
        print(f"  {lvl:<22} {int(r['count']):>7} {r['mean_bias_oz']:>+11.2f} {r['mae_oz']:>10.2f} "
              f"{h['mean_bias_oz']:>+10.2f} {h['mae_oz']:>9.2f}   {f}")


def compare_item_count(ridge_tbl: pd.DataFrame, hgb_tbl: pd.DataFrame) -> None:
    print(f"\n  -- ITEM COUNT BUCKET: Ridge vs HistGBT --")
    header = f"  {'bucket':<10} {'count':>7} {'ridge_bias':>11} {'ridge_mae':>10} {'hgb_bias':>10} {'hgb_mae':>9}   flag"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in ridge_tbl.iterrows():
        h = hgb_tbl[hgb_tbl["item_count_bucket"] == r["item_count_bucket"]]
        if h.empty:
            continue
        h = h.iloc[0]
        f = flag(r["mae_oz"], h["mae_oz"])
        print(f"  {str(r['item_count_bucket']):<10} {int(r['count']):>7} {r['mean_bias_oz']:>+11.2f} "
              f"{r['mae_oz']:>10.2f} {h['mean_bias_oz']:>+10.2f} {h['mae_oz']:>9.2f}   {f}")


def find_new_weak_spots(ridge_tbl: pd.DataFrame, hgb_tbl: pd.DataFrame, seg_col: str,
                        min_count: int = 20, worse_by_oz: float = 3.0) -> pd.DataFrame:
    """Segments where HistGBT's MAE is notably worse than Ridge's, restricted
    to segments with enough rows to be meaningful -- the check for NEW
    problems Ridge didn't have, not just missing improvements."""
    merged = ridge_tbl[[seg_col, "count", "mae_oz", "mean_bias_oz"]].merge(
        hgb_tbl[[seg_col, "mae_oz", "mean_bias_oz"]], on=seg_col, suffixes=("_ridge", "_hgb")
    )
    merged["mae_delta_oz"] = merged["mae_oz_hgb"] - merged["mae_oz_ridge"]
    worse = merged[(merged["count"] >= min_count) & (merged["mae_delta_oz"] >= worse_by_oz)]
    return worse.sort_values("mae_delta_oz", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shipments", default=DEFAULT_SHIPMENTS)
    parser.add_argument("--lines", default=DEFAULT_LINES)
    parser.add_argument("--refresh", action="store_true",
                        help="Bypass the cached train/test split from _experiment_prep")
    args = parser.parse_args()

    train, test = prepare(args.shipments, args.lines, refresh=args.refresh)
    X_train, y_train = train[ALL_FEATURES], train[TARGET]
    X_test, y_test = test[ALL_FEATURES], test[TARGET]

    _hr("STEP 1 -- REFIT UNTUNED HISTGBT(absolute_error) ON JAN-MAY, PREDICT JUNE")
    bundle = joblib.load(PROD_MODEL_PATH)
    ridge_preds = bundle["pipeline"].predict(test[bundle["feature_list"]])
    print(f"  Ridge (production, model_version={bundle.get('model_version', 'unknown')}) loaded from {PROD_MODEL_PATH}")

    hgb_pipe = build_histgbt_pipeline()
    hgb_pipe.fit(X_train, y_train)
    hgb_preds = hgb_pipe.predict(X_test)
    print(f"  HistGBT refit on {len(X_train):,} Jan-May rows, predicted on {len(X_test):,} June rows")

    _hr("STEP 2 -- SAME THREE BREAKDOWNS STAGE 7 PRODUCES, FOR EACH MODEL")
    ridge_bd = full_breakdown("RIDGE (production)", y_test, ridge_preds, test)
    hgb_bd = full_breakdown("HISTGBT (untuned, absolute_error)", y_test, hgb_preds, test)

    _hr("STEP 3 -- SIDE-BY-SIDE: KNOWN TROUBLE SPOTS, RIDGE vs HISTGBT")
    compare_segments("SHIP METHOD watch list", ridge_bd["ship_bias"], hgb_bd["ship_bias"],
                     "ship_method", WATCH_SHIP_METHODS)
    compare_segments("BOX NAME watch list", ridge_bd["box_bias"], hgb_bd["box_bias"],
                     "box_name", WATCH_BOX_NAMES)
    compare_item_count(ridge_bd["item_bias"], hgb_bd["item_bias"])

    _hr("STEP 4 -- NEW WEAK SPOTS: SEGMENTS WHERE HISTGBT IS NOTABLY WORSE THAN RIDGE")
    print(f"  (segments with >= 20 test rows where HistGBT's MAE is >= 3.0 oz worse than Ridge's --")
    print(f"   the check for problems Ridge did NOT have, not just missing wins)\n")

    new_ship = find_new_weak_spots(ridge_bd["ship_bias"], hgb_bd["ship_bias"], "ship_method")
    new_box = find_new_weak_spots(ridge_bd["box_bias"], hgb_bd["box_bias"], "box_name")

    if new_ship.empty and new_box.empty:
        print("  None found. No ship_method or box_name with >=20 rows got meaningfully worse")
        print("  under HistGBT than it already was under Ridge.")
    else:
        if not new_ship.empty:
            print("  New/worsened SHIP METHOD segments:")
            print(new_ship.to_string(index=False, formatters={
                "mae_oz_ridge": "{:.2f}".format, "mae_oz_hgb": "{:.2f}".format,
                "mean_bias_oz_ridge": "{:+.2f}".format, "mean_bias_oz_hgb": "{:+.2f}".format,
                "mae_delta_oz": "{:+.2f}".format,
            }))
        if not new_box.empty:
            print("\n  New/worsened BOX NAME segments:")
            print(new_box.to_string(index=False, formatters={
                "mae_oz_ridge": "{:.2f}".format, "mae_oz_hgb": "{:.2f}".format,
                "mean_bias_oz_ridge": "{:+.2f}".format, "mean_bias_oz_hgb": "{:+.2f}".format,
                "mae_delta_oz": "{:+.2f}".format,
            }))

    _hr("STEP 5 -- SUMMARY: DOES HISTGBT FIX, WORSEN, OR LEAVE UNCHANGED EACH KNOWN WEAK SPOT?")
    r_m, h_m = ridge_bd["metrics"], hgb_bd["metrics"]
    print(f"  Overall: Ridge {r_m['mae_oz']:.2f} oz -> HistGBT {h_m['mae_oz']:.2f} oz "
          f"({h_m['mae_oz'] - r_m['mae_oz']:+.2f} oz)\n")

    for lvl in WATCH_SHIP_METHODS + WATCH_BOX_NAMES:
        tbl_r = ridge_bd["ship_bias"] if lvl in WATCH_SHIP_METHODS else ridge_bd["box_bias"]
        tbl_h = hgb_bd["ship_bias"] if lvl in WATCH_SHIP_METHODS else hgb_bd["box_bias"]
        col = "ship_method" if lvl in WATCH_SHIP_METHODS else "box_name"
        r_row = tbl_r[tbl_r[col] == lvl]
        h_row = tbl_h[tbl_h[col] == lvl]
        if r_row.empty or h_row.empty:
            print(f"  {lvl}: not present in one of the tables, skipped.")
            continue
        r_row, h_row = r_row.iloc[0], h_row.iloc[0]
        verdict = flag(r_row["mae_oz"], h_row["mae_oz"])
        print(f"  {lvl:<20} Ridge bias {r_row['mean_bias_oz']:+.2f} oz / MAE {r_row['mae_oz']:.2f} oz  ->  "
              f"HistGBT bias {h_row['mean_bias_oz']:+.2f} oz / MAE {h_row['mae_oz']:.2f} oz   [{verdict}]")

    print("\n  item_count trend (error growing with item_count):")
    for _, r in ridge_bd["item_bias"].iterrows():
        h = hgb_bd["item_bias"][hgb_bd["item_bias"]["item_count_bucket"] == r["item_count_bucket"]]
        if h.empty:
            continue
        h = h.iloc[0]
        print(f"    {str(r['item_count_bucket']):<8} Ridge MAE {r['mae_oz']:.2f} oz  ->  "
              f"HistGBT MAE {h['mae_oz']:.2f} oz   [{flag(r['mae_oz'], h['mae_oz'])}]")

    if new_ship.empty and new_box.empty:
        print("\n  No new weak spots introduced (see Step 4).")
    else:
        print(f"\n  NEW weak spots introduced -- see Step 4 for the segments and magnitudes.")

    _hr("DONE  (diagnostic-only -- no model saved, models/ untouched)")


if __name__ == "__main__":
    main()

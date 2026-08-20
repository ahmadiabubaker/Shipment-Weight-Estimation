"""Does imputing missing box tare weight fix the monthly "drift" and reduce
model error -- or was the drift an artifact all along?

Context: scripts/analyze_residuals.py's monthly drift table showed the raw
residual (actual - theoretical) climbing from +12.8 oz (Jan) to +18.9 oz
(June). scripts/experiment_box_geometry.py separately found that box tare
weight (theoretical_empty_box_weight_lbs) stopped being recorded for a
growing share of shipments starting ~April, reaching 100% missing by June --
but concluded that using a MISSING-TARE INDICATOR as a model feature is a
fake win (it's a disguised timestamp, constant across the whole test fold).

This script asks a different, complementary question: instead of telling
the model "tare is missing" (which just teaches it to identify late
months), what if we IMPUTE a reasonable tare value for those rows, so
theoretical_weight_oz stops being systematically understated? Checked here:

  1. Does the monthly drift actually disappear once rows are split by
     tare-missing status? (it does -- see the conversation this script
     follows from: within non-missing rows, residual is flat/improving;
     within missing rows it's stable; the AGGREGATE drift is pure mix-shift
     as missing-tare rows grow from 0% to 100% of the data.)
  2. How much is theoretical_weight_oz actually understated for
     missing-tare rows? (train-fold-only median tare per carton_type,
     looked up and imputed -- same leakage-safe pattern as
     category_error_map.)
  3. Does the imputed theoretical weight reduce the RAW baseline's bias/MAE?
  4. Does refitting production HistGBT on the imputed feature set (theoretical_
     weight_oz, weight_per_item_oz, and category_avg_weight_error_oz all
     recomputed from the imputed value) beat the current production model
     on June test?

Evaluation-only / exploratory: nothing in models/ is changed, no training
script is touched, no model is selected or saved.

Usage:
    python scripts/evaluate_tare_imputation.py
    python scripts/evaluate_tare_imputation.py --refresh
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
from shipment_weight.features import ALL_FEATURES, TARGET
from shipment_weight.ingest import apply_category_map
from shipment_weight.rounding import round_to_billing_tier

from _experiment_prep import DEFAULT_LINES, DEFAULT_SHIPMENTS, prepare
from evaluate_rounded import rounded_metrics
from train_real_data import LBS_TO_OZ, _hr, category_error_map, make_histgbt_pipeline

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROD_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model.joblib")


def impute_tare(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train-fold-only median tare per carton_type, from rows where tare
    was actually recorded (> 0). Applied to rows where tare == 0 in BOTH
    train and test -- unseen/never-recorded carton_types fall back to the
    global median tare among recorded rows, same pattern as
    apply_category_map/category_error_map."""
    recorded = train[train["theoretical_empty_box_weight_lbs"] > 0]
    tare_by_box = recorded.groupby("carton_type")["theoretical_empty_box_weight_lbs"].median().to_dict()
    global_tare = float(recorded["theoretical_empty_box_weight_lbs"].median())

    stats = {"n_boxes_with_known_tare": len(tare_by_box), "global_median_tare_lbs": global_tare}

    def _fix(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = df["theoretical_empty_box_weight_lbs"] <= 0
        imputed_tare = df["carton_type"].map(tare_by_box).fillna(global_tare)
        df["theoretical_empty_box_weight_lbs_v2"] = np.where(
            missing, imputed_tare, df["theoretical_empty_box_weight_lbs"]
        )
        df["was_tare_imputed"] = missing.astype(int)
        df["total_theoretical_shipment_weight_lbs_v2"] = (
            df["theoretical_cargo_weight_lbs"] + df["theoretical_empty_box_weight_lbs_v2"]
        )
        df["theoretical_weight_oz_v2"] = df["total_theoretical_shipment_weight_lbs_v2"] * LBS_TO_OZ
        df["weight_per_item_oz_v2"] = df["theoretical_weight_oz_v2"] / df["item_count"].clip(lower=1)
        return df

    return _fix(train), _fix(test), stats


def rebuild_features_v2(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overwrites theoretical_weight_oz / weight_per_item_oz /
    category_avg_weight_error_oz IN PLACE (under their original ALL_FEATURES
    names) with the tare-imputed versions, so the standard
    make_histgbt_pipeline()/ALL_FEATURES can be reused unchanged. Category
    encoding recomputed against the corrected residual so it stays
    consistent with the corrected theoretical weight (train-fold only, same
    leakage discipline as train_real_data.encode_category_error)."""
    train = train.copy()
    test = test.copy()

    train_resid_v2 = train["actual_weight_oz"] - train["theoretical_weight_oz_v2"]
    cat_map_v2 = (
        pd.DataFrame({"cat": train["category_mode"], "err": train_resid_v2})
        .groupby("cat")["err"].mean().to_dict()
    )
    global_mean_v2 = float(train_resid_v2.mean())

    for df, resid in [(train, train_resid_v2), (test, None)]:
        df["theoretical_weight_oz"] = df["theoretical_weight_oz_v2"]
        df["weight_per_item_oz"] = df["weight_per_item_oz_v2"]
    train["category_avg_weight_error_oz"] = apply_category_map(train["category_mode"], cat_map_v2, global_mean_v2)
    test["category_avg_weight_error_oz"] = apply_category_map(test["category_mode"], cat_map_v2, global_mean_v2)

    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shipments", default=DEFAULT_SHIPMENTS)
    parser.add_argument("--lines", default=DEFAULT_LINES)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    train, test = prepare(args.shipments, args.lines, refresh=args.refresh)
    actual_oz, actual_lbs = test["actual_weight_oz"], test["actual_weight_lbs"]

    _hr("STEP 1 -- IMPUTE MISSING TARE (train-fold-only median per carton_type)")
    train, test, stats = impute_tare(train, test)
    print(f"  Boxes with a known (recorded) tare in train: {stats['n_boxes_with_known_tare']}")
    print(f"  Global fallback tare (train median, recorded rows only): {stats['global_median_tare_lbs']:.2f} lbs")
    n_imputed_test = int(test["was_tare_imputed"].sum())
    print(f"  June test rows with imputed tare: {n_imputed_test:,} ({n_imputed_test/len(test)*100:.1f}%)")

    understatement_oz = (test["theoretical_weight_oz_v2"] - test["theoretical_weight_oz"])[test["was_tare_imputed"] == 1]
    print(f"  Median theoretical-weight understatement on imputed June rows: "
          f"{understatement_oz.median():.2f} oz ({understatement_oz.median()/LBS_TO_OZ:.3f} lbs)")

    _hr("STEP 2 -- DOES THE RAW THEORETICAL BASELINE IMPROVE?")
    base_orig = regression_metrics(actual_oz, test["theoretical_weight_oz"].values)
    base_v2 = regression_metrics(actual_oz, test["theoretical_weight_oz_v2"].values)
    print(f"  theoretical_weight_oz (original) : MAE {base_orig['mae_oz']:.2f} oz  bias {base_orig['bias_oz']:+.2f} oz")
    print(f"  theoretical_weight_oz_v2 (fixed) : MAE {base_v2['mae_oz']:.2f} oz  bias {base_v2['bias_oz']:+.2f} oz")
    print(f"  Bias reduction: {base_orig['bias_oz'] - base_v2['bias_oz']:+.2f} oz "
          f"({(1 - abs(base_v2['bias_oz'])/abs(base_orig['bias_oz']))*100:.0f}% of the baseline's systematic "
          f"underestimate removed)")

    _hr("STEP 3 -- DOES THE MONTHLY DRIFT DISAPPEAR?  (whole dataset, actual - theoretical_v2)")
    all_rows = pd.concat([train, test], ignore_index=True)
    all_rows["month"] = pd.to_datetime(all_rows["order_date"]).dt.to_period("M").astype(str)
    all_rows["resid_v2_oz"] = all_rows["actual_weight_oz"] - all_rows["theoretical_weight_oz_v2"]
    drift_v2 = all_rows.groupby("month")["resid_v2_oz"].mean()
    print("  Raw (actual - theoretical_v2) residual by month:")
    print(drift_v2.to_string(float_format="{:+.2f}".format))

    _hr("STEP 4 -- REFIT PRODUCTION HISTGBT ON THE TARE-IMPUTED FEATURE SET")
    train_v2, test_v2 = rebuild_features_v2(train, test)

    orig_pipe = make_histgbt_pipeline()
    orig_pipe.fit(train[ALL_FEATURES], train[TARGET])
    orig_preds = orig_pipe.predict(test[ALL_FEATURES])

    v2_pipe = make_histgbt_pipeline()
    v2_pipe.fit(train_v2[ALL_FEATURES], train_v2[TARGET])
    v2_preds = v2_pipe.predict(test_v2[ALL_FEATURES])

    def report(label: str, preds_oz: np.ndarray) -> None:
        cont = regression_metrics(actual_oz, preds_oz)
        rounded, _ = round_to_billing_tier(preds_oz / LBS_TO_OZ)
        rnd = rounded_metrics(actual_lbs.values, rounded)
        print(f"  {label:<32} continuous MAE {cont['mae_oz']/LBS_TO_OZ:>6.3f} lbs, bias {cont['bias_oz']:>+6.2f} oz  |  "
              f"rounded MAE {rnd['mae_lbs']:>6.3f} lbs, within-1lb {rnd['within_1lb_pct']:>5.1f}%, "
              f"exact-match {rnd['exact_match_pct']:>5.1f}%")

    report("histgbt_production (original)", orig_preds)
    report("histgbt_tare_imputed", v2_preds)

    # Bonus: does the fix help most exactly where we'd expect -- the rows
    # that actually had their tare imputed?
    imputed_mask = test_v2["was_tare_imputed"].values == 1
    if imputed_mask.sum() > 0:
        orig_mae_imputed = np.abs(orig_preds[imputed_mask] - actual_oz.values[imputed_mask]).mean()
        v2_mae_imputed = np.abs(v2_preds[imputed_mask] - actual_oz.values[imputed_mask]).mean()
        print(f"\n  On just the {imputed_mask.sum():,} June rows with imputed tare:")
        print(f"    histgbt_production (original) MAE : {orig_mae_imputed:.2f} oz")
        print(f"    histgbt_tare_imputed MAE          : {v2_mae_imputed:.2f} oz  "
              f"({v2_mae_imputed - orig_mae_imputed:+.2f} oz)")

    _hr("DONE  (exploratory only -- no model selected, nothing saved to models/)")


if __name__ == "__main__":
    main()

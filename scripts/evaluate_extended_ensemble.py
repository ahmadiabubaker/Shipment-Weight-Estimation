"""Can HistGBT (production) and the extended-feature Ridge model be
combined -- and should the extended model be trusted at all yet?

Context: scripts/evaluate_ensemble.py already checked blending production
HistGBT with *base* Ridge (same ALL_FEATURES as HistGBT) and found their
residuals correlate at 0.884 -- too similar to gain much. That check never
covered models/model_extended.joblib, which is trained on a different,
larger feature set (ALL_FEATURES_EXTENDED: distinct_sku_count,
weight_density_lbs_per_in3, top_category_by_weight, fill_ratio_winsorized,
is_overfilled, box_name_target_enc_oz, ship_method_target_enc_oz). A model
trained on different inputs could plausibly make less-correlated errors --
worth checking on its own rather than assuming the base-Ridge result
transfers.

Also: the extended model is labeled "experimental" in its own bundle and
has never been through the diagnostic pass HistGBT got before production
promotion (bias by ship_method/box_name/item_count -- see
scripts/evaluate_histgbt_diagnostics.py). Blending an unvetted model into
anything would skip that scrutiny, so this script runs it first.

Steps:
  1. Predictions from production HistGBT (base features) and the extended
     Ridge model (extended features) on the same June test set.
  2. Diagnostic breakdown of the extended model alone -- bias by
     ship_method, box_name, item_count bucket -- same shape as the checks
     HistGBT already passed, so any hidden weak spot is visible before
     trusting this model for anything.
  3. Residual correlation between HistGBT and extended-Ridge on June test.
  4. Blend-weight sweep (0.3/0.5/0.7 HistGBT/extended-Ridge) on a
     train-fold-only validation split (Jan-Apr fit, May validate; category
     encoding and extended-feature train-fold stats all recomputed from
     Jan-Apr only -- June never touched during tuning).
  5. Continuous + rounded-to-billing-tier metrics (reusing
     shipment_weight.rounding and evaluate_rounded.rounded_metrics) for
     HistGBT alone, extended-Ridge alone, and the best blend, on June test.

Evaluation-only / exploratory: nothing in models/ is changed, no training
script is touched, no model is selected or saved.

Usage:
    python scripts/evaluate_extended_ensemble.py
    python scripts/evaluate_extended_ensemble.py --refresh
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from shipment_weight.evaluate import bias_by_item_count_bucket, bias_by_segment, regression_metrics
from shipment_weight.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from shipment_weight.features_extended import (
    ALL_FEATURES_EXTENDED,
    add_extended_features,
    add_target_encodings,
    finalize_train_fold_stats,
)
from shipment_weight.ingest import apply_category_map
from shipment_weight.rounding import round_to_billing_tier

from _experiment_prep import DEFAULT_LINES, DEFAULT_SHIPMENTS, prepare
from evaluate_rounded import rounded_metrics
from train_real_data import LBS_TO_OZ, _hr, category_error_map, make_histgbt_pipeline
from train_real_data_extended import make_pipeline_extended

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROD_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model.joblib")
EXTENDED_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model_extended.joblib")

BLEND_WEIGHTS = (0.3, 0.5, 0.7)  # fraction assigned to HistGBT; extended-Ridge gets the rest
SIMILAR_MAE_THRESHOLD_OZ = 1.0


def make_validation_split(train: pd.DataFrame):
    """Leakage-safe Jan-Apr fit / May validate split -- category encoding
    recomputed from Jan-Apr only, same pattern as evaluate_ensemble.py."""
    subtrain = train[train["order_date"].dt.month <= 4].copy()
    val = train[train["order_date"].dt.month == 5].copy()
    cmap, gmean = category_error_map(subtrain)
    subtrain["category_avg_weight_error_oz"] = apply_category_map(subtrain["category_mode"], cmap, gmean)
    val["category_avg_weight_error_oz"] = apply_category_map(val["category_mode"], cmap, gmean)
    return subtrain, val


def build_extended_split(fit_df: pd.DataFrame, score_df: pd.DataFrame, lines_raw: pd.DataFrame):
    """Add extended features + target encodings over fit_df+score_df's own
    combined date range, then finalize train-fold-only stats (winsorize
    threshold, density fallback) from fit_df alone. Mirrors
    train_real_data_extended.py's ordering exactly -- works identically for
    the real June split (fit=Jan-May, score=June) and the validation split
    (fit=Jan-Apr, score=May); target encoding never sees dates beyond
    score_df's own range either way, so June is never touched when this is
    called for validation."""
    fit_df = fit_df.copy()
    score_df = score_df.copy()
    fit_df["_split"] = "fit"
    score_df["_split"] = "score"
    df_full = pd.concat([fit_df, score_df], ignore_index=True)
    df_full = add_extended_features(df_full, lines_raw)
    df_full = add_target_encodings(df_full)
    fit_ext = df_full[df_full["_split"] == "fit"].drop(columns="_split").reset_index(drop=True)
    score_ext = df_full[df_full["_split"] == "score"].drop(columns="_split").reset_index(drop=True)
    fit_ext, score_ext = finalize_train_fold_stats(fit_ext, score_ext)
    return fit_ext, score_ext


def flag(a_mae: float, b_mae: float) -> str:
    delta = b_mae - a_mae
    if abs(delta) < SIMILAR_MAE_THRESHOLD_OZ:
        return "similar"
    return "extended-Ridge BETTER" if delta < 0 else "extended-Ridge WORSE"


def diagnose_extended_model(y_test: pd.Series, preds_oz: np.ndarray, test: pd.DataFrame) -> None:
    m = regression_metrics(y_test, preds_oz)
    print(f"\n  [extended-Ridge]  overall MAE {m['mae_oz']:.2f} oz ({m['mae_oz']/LBS_TO_OZ:.3f} lbs)   "
          f"bias {m['bias_oz']:+.2f} oz   within-1lb {m['within_1lb_pct']:.1f}%")

    ship_bias = bias_by_segment(y_test, preds_oz, test["ship_method"].reset_index(drop=True), "ship_method")
    box_bias = bias_by_segment(y_test, preds_oz, test["carton_type"].reset_index(drop=True), "box_name")
    item_bias = bias_by_item_count_bucket(y_test, preds_oz, test["item_count"].reset_index(drop=True))

    print("\n  -- Bias by SHIP METHOD [extended-Ridge] --")
    print(ship_bias.to_string(index=False))
    print("\n  -- Bias by BOX NAME (top 15 by row count) [extended-Ridge] --")
    print(box_bias.head(15).to_string(index=False))
    print("\n  -- Bias by ITEM COUNT BUCKET [extended-Ridge] --")
    print(item_bias.to_string(index=False))

    # The two known HistGBT/Ridge trouble spots, for a direct look at
    # whether the extended model inherits, fixes, or worsens them.
    for lvl in ["FedEx HAZMAT"]:
        row = ship_bias[ship_bias["ship_method"] == lvl]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n  {lvl}: bias {r['mean_bias_oz']:+.2f} oz, MAE {r['mae_oz']:.2f} oz "
                  f"(HistGBT was -2.31 oz / 8.15 oz; Ridge base was +10.77 oz / 25.83 oz)")
    for lvl in ["30x20x12", "20x14x12"]:
        row = box_bias[box_bias["box_name"] == lvl]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {lvl}: bias {r['mean_bias_oz']:+.2f} oz, MAE {r['mae_oz']:.2f} oz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shipments", default=DEFAULT_SHIPMENTS)
    parser.add_argument("--lines", default=DEFAULT_LINES)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    train, test, lines_raw = prepare(args.shipments, args.lines, refresh=args.refresh, with_lines=True)
    actual_oz, actual_lbs = test["actual_weight_oz"], test["actual_weight_lbs"]

    _hr("STEP 1 -- PREDICTIONS ON JUNE TEST")
    hgb_bundle = joblib.load(PROD_MODEL_PATH)
    hgb_preds = hgb_bundle["pipeline"].predict(test[hgb_bundle["feature_list"]])
    print(f"  HistGBT (production, {hgb_bundle.get('model_version')}): {len(hgb_preds):,} predictions")

    _, test_ext = build_extended_split(train, test, lines_raw)
    ext_bundle = joblib.load(EXTENDED_MODEL_PATH)
    ext_preds = ext_bundle["pipeline"].predict(test_ext[ALL_FEATURES_EXTENDED])
    print(f"  Extended Ridge ({ext_bundle.get('model_version')}): {len(ext_preds):,} predictions")

    _hr("STEP 2 -- DIAGNOSTIC PASS ON THE EXTENDED MODEL (never done before)")
    diagnose_extended_model(actual_oz, ext_preds, test)

    _hr("STEP 3 -- HOW CORRELATED ARE HISTGBT AND EXTENDED-RIDGE'S ERRORS?")
    hgb_resid = hgb_preds - actual_oz.values
    ext_resid = ext_preds - actual_oz.values
    corr = float(np.corrcoef(hgb_resid, ext_resid)[0, 1])
    print(f"  Pearson correlation of residuals (HistGBT vs extended-Ridge): {corr:.3f}")
    print(f"  (for reference: base Ridge vs HistGBT was 0.884, per evaluate_ensemble.py)")
    if corr > 0.85:
        print("  -> Still highly correlated: the extended features didn't decouple the errors")
        print("     enough to expect a meaningful ensembling gain.")
    elif corr > 0.6:
        print("  -> Moderately correlated: somewhat less overlap than the base-Ridge pairing.")
    else:
        print("  -> Notably less correlated than the base-Ridge pairing -- this combination")
        print("     is more promising for ensembling than the one already tested.")

    _hr("STEP 4 -- BLEND-WEIGHT SWEEP (train-fold validation only; June untouched)")
    subtrain, val = make_validation_split(train)
    subtrain_ext, val_ext = build_extended_split(subtrain, val, lines_raw)

    hgb_val_pipe = make_histgbt_pipeline()
    hgb_val_pipe.fit(subtrain[ALL_FEATURES], subtrain[TARGET])
    hgb_val_preds = hgb_val_pipe.predict(val[ALL_FEATURES])

    ext_val_pipe = make_pipeline_extended()
    ext_val_pipe.fit(subtrain_ext[ALL_FEATURES_EXTENDED], subtrain_ext[TARGET])
    ext_val_preds = ext_val_pipe.predict(val_ext[ALL_FEATURES_EXTENDED])

    val_actual = val[TARGET].values
    rows, best_w, best_mae = [], None, float("inf")
    for w in BLEND_WEIGHTS:
        blend = w * hgb_val_preds + (1 - w) * ext_val_preds
        mae_oz = float(np.mean(np.abs(blend - val_actual)))
        rows.append({"histgbt_weight": w, "extended_ridge_weight": 1 - w, "val_mae_oz": mae_oz})
        if mae_oz < best_mae:
            best_mae, best_w = mae_oz, w
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "histgbt_weight": "{:.1f}".format, "extended_ridge_weight": "{:.1f}".format,
        "val_mae_oz": "{:.2f}".format,
    }))
    print(f"\n  Best blend weight on validation: histgbt={best_w:.1f} / extended_ridge={1-best_w:.1f}")

    blend_preds = best_w * hgb_preds + (1 - best_w) * ext_preds

    _hr("STEP 5 -- JUNE RESULTS: HISTGBT ALONE vs EXTENDED-RIDGE ALONE vs BLEND")
    def report(label: str, preds_oz: np.ndarray) -> None:
        cont = regression_metrics(actual_oz, preds_oz)
        rounded, _ = round_to_billing_tier(preds_oz / LBS_TO_OZ)
        rnd = rounded_metrics(actual_lbs.values, rounded)
        print(f"  {label:<28} continuous MAE {cont['mae_oz']/LBS_TO_OZ:>6.3f} lbs  "
              f"(within-1lb {cont['within_1lb_pct']:>5.1f}%)   |   "
              f"rounded MAE {rnd['mae_lbs']:>6.3f} lbs  (within-1lb {rnd['within_1lb_pct']:>5.1f}%, "
              f"exact-match {rnd['exact_match_pct']:>5.1f}%)")

    report("histgbt_alone", hgb_preds)
    report("extended_ridge_alone", ext_preds)
    report(f"blend_{best_w:.1f}_{1-best_w:.1f}", blend_preds)

    _hr("STEP 6 -- VERDICT")
    print(f"  Residual correlation: {corr:.3f} (vs 0.884 for the already-tested base-Ridge pairing)")
    print("  This script is diagnostic/exploratory only -- it does not select or save a model,")
    print("  and the extended model remains 'experimental' regardless of these numbers unless")
    print("  someone deliberately promotes it (and updates its bundle with feature_list /")
    print("  category_error_map the way models/model.joblib has, which it currently lacks).")

    _hr("DONE  (exploratory only -- no model selected, nothing saved to models/)")


if __name__ == "__main__":
    main()

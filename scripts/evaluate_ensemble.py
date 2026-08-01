"""Evaluate a Ridge + HistGBT(absolute-error) ENSEMBLE.

Terminology note: Tomas asked about combining multiple models ("boosting")
to improve accuracy. True boosting -- sequential error-correction, each new
model fit on the previous one's residuals -- is already what
HistGradientBoostingRegressor does internally. What's actually useful here
is ENSEMBLING: blending predictions from two different model *families*,
since different model types tend to make different, less-correlated
errors, and averaging over that disagreement can reduce variance. This
script blends:
  - Ridge (linear, production model -- models/model.joblib)
  - HistGBT with loss="absolute_error" (refit here with the exact
    hyperparameters from scripts/experiment_loss_and_interactions.py's
    "histgbt_absolute_error" candidate, Experiment 3)

Steps:
  1. Predictions from both models on the same June test set.
  2. Correlation of their residuals on that test set -- the diagnostic for
     whether ensembling is likely to help at all (high correlation ->
     the models are wrong about the same rows -> little to gain).
  3. A blend-weight sweep (Ridge weight 0.3 / 0.5 / 0.7) chosen on a
     TRAIN-FOLD validation split only (May, held out from a Jan-Apr
     sub-train; category encoding recomputed from Jan-Apr only, same
     leakage-safe pattern as train_real_data.encode_category_error) --
     June test is never touched by weight selection. The winning weight is
     then applied once to real June test predictions.
  4. Continuous MAE/within-1lb and rounded-to-billing-tier metrics (reusing
     evaluate_rounded.py's rounding logic) for Ridge alone, HistGBT alone,
     and the blend, side by side.
  5. The real cost of shipping an ensemble, printed explicitly so it's part
     of the decision, not just the accuracy delta.

Evaluation-only / exploratory: nothing in models/ is changed, no training
script is touched, and this script does not pick or save a final model.
HistGBT is fit fresh in-memory each run purely for comparison.

Usage:
    python scripts/evaluate_ensemble.py
    python scripts/evaluate_ensemble.py --shipments ... --lines ... --refresh
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

from shipment_weight.evaluate import regression_metrics
from shipment_weight.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from shipment_weight.ingest import apply_category_map
from shipment_weight.train import make_pipeline

from _experiment_prep import DEFAULT_LINES, DEFAULT_SHIPMENTS, prepare
from evaluate_rounded import round_to_billing_tier, rounded_metrics
from experiment_loss_and_interactions import pipeline_for
from train_real_data import LBS_TO_OZ, _hr, category_error_map

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROD_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model.joblib")

# Kept in sync with scripts/experiment_loss_and_interactions.py's Experiment 3
# "histgbt_absolute_error" candidate -- same hyperparameters, same feature set.
HGB_KWARGS = dict(
    loss="absolute_error", max_iter=400, learning_rate=0.05,
    max_depth=None, min_samples_leaf=40, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.1, random_state=42,
)
BLEND_WEIGHTS = (0.3, 0.5, 0.7)  # fraction assigned to Ridge; HistGBT gets the rest


def fit_histgbt(X_train: pd.DataFrame, y_train: pd.Series):
    pipe = pipeline_for(HistGradientBoostingRegressor(**HGB_KWARGS),
                        NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True)
    pipe.fit(X_train, y_train)
    return pipe


def select_blend_weight(train: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Sweep BLEND_WEIGHTS on a train-fold-only validation split (May, held
    out from a Jan-Apr sub-train) and return (best_weight, sweep_table).
    Never touches June test."""
    subtrain = train[train["order_date"].dt.month <= 4].copy()
    val = train[train["order_date"].dt.month == 5].copy()

    cmap, gmean = category_error_map(subtrain)
    subtrain["category_avg_weight_error_oz"] = apply_category_map(subtrain["category_mode"], cmap, gmean)
    val["category_avg_weight_error_oz"] = apply_category_map(val["category_mode"], cmap, gmean)

    X_sub, y_sub = subtrain[ALL_FEATURES], subtrain[TARGET]
    X_val, y_val = val[ALL_FEATURES], val[TARGET]

    ridge_val = make_pipeline_ridge().fit(X_sub, y_sub)
    hgb_val = fit_histgbt(X_sub, y_sub)
    ridge_val_preds = ridge_val.predict(X_val)
    hgb_val_preds = hgb_val.predict(X_val)

    rows = []
    best_w, best_mae = None, float("inf")
    for w in BLEND_WEIGHTS:
        blend = w * ridge_val_preds + (1 - w) * hgb_val_preds
        mae_oz = float(np.mean(np.abs(blend - y_val.values)))
        rows.append({"ridge_weight": w, "histgbt_weight": 1 - w,
                     "val_mae_oz": mae_oz, "val_mae_lbs": mae_oz / LBS_TO_OZ})
        if mae_oz < best_mae:
            best_mae, best_w = mae_oz, w

    return best_w, pd.DataFrame(rows)


def make_pipeline_ridge():
    from sklearn.linear_model import Ridge
    return make_pipeline(Ridge(alpha=1.0))


def report_candidate(label: str, actual_oz: pd.Series, actual_lbs: pd.Series, preds_oz: np.ndarray) -> dict:
    cont = regression_metrics(actual_oz, preds_oz)
    pred_lbs = preds_oz / LBS_TO_OZ
    rounded, sub_1lb = round_to_billing_tier(pred_lbs)
    rnd = rounded_metrics(actual_lbs.values, rounded)
    row = {
        "model": label,
        "continuous_mae_lbs": cont["mae_oz"] / LBS_TO_OZ,
        "continuous_within_1lb_pct": cont["within_1lb_pct"],
        "rounded_mae_lbs": rnd["mae_lbs"],
        "rounded_within_1lb_pct": rnd["within_1lb_pct"],
        "rounded_exact_match_pct": rnd["exact_match_pct"],
        "sub_1lb_rows": int(sub_1lb.sum()),
    }
    print(f"  {label:<22} continuous MAE {row['continuous_mae_lbs']:>6.3f} lbs  "
          f"(within-1lb {row['continuous_within_1lb_pct']:>5.1f}%)   |   "
          f"rounded MAE {row['rounded_mae_lbs']:>6.3f} lbs  "
          f"(within-1lb {row['rounded_within_1lb_pct']:>5.1f}%, "
          f"exact-match {row['rounded_exact_match_pct']:>5.1f}%)")
    return row


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
    actual_oz, actual_lbs = test["actual_weight_oz"], test["actual_weight_lbs"]

    _hr("STEP 1 -- PREDICTIONS FROM BOTH MODELS ON JUNE TEST")
    bundle = joblib.load(PROD_MODEL_PATH)
    ridge_preds = bundle["pipeline"].predict(test[bundle["feature_list"]])
    print(f"  Ridge (production, model_version={bundle.get('model_version', 'unknown')}): "
          f"{len(ridge_preds):,} predictions")

    print("  Fitting histgbt_absolute_error on the full Jan-May train fold "
          "(same hyperparameters as scripts/experiment_loss_and_interactions.py)...")
    hgb_full = fit_histgbt(X_train, y_train)
    hgb_preds = hgb_full.predict(X_test)
    print(f"  HistGBT (absolute_error, refit here): {len(hgb_preds):,} predictions")

    _hr("STEP 2 -- HOW CORRELATED ARE THEIR ERRORS?  (June test residuals)")
    ridge_resid = ridge_preds - actual_oz.values
    hgb_resid = hgb_preds - actual_oz.values
    corr = float(np.corrcoef(ridge_resid, hgb_resid)[0, 1])
    print(f"  Pearson correlation of residuals (Ridge vs HistGBT): {corr:.3f}")
    if corr > 0.85:
        print("  -> Highly correlated: the two models tend to be wrong on the same rows.")
        print("     Ensembling is unlikely to help much beyond noise reduction.")
    elif corr > 0.6:
        print("  -> Moderately correlated: some shared error, some independent error.")
        print("     A blend may help a little.")
    else:
        print("  -> Not strongly correlated: the models disagree on which rows are hard.")
        print("     This is the situation where ensembling tends to pay off.")

    _hr("STEP 3 -- BLEND-WEIGHT SWEEP  (train-fold validation only, May held out; June untouched)")
    best_w, sweep = select_blend_weight(train)
    print(sweep.to_string(index=False, formatters={
        "ridge_weight": "{:.1f}".format, "histgbt_weight": "{:.1f}".format,
        "val_mae_oz": "{:.2f}".format, "val_mae_lbs": "{:.3f}".format,
    }))
    print(f"\n  Best blend weight on validation: ridge={best_w:.1f} / histgbt={1-best_w:.1f}")

    blend_preds = best_w * ridge_preds + (1 - best_w) * hgb_preds

    _hr("STEP 4 -- JUNE TEST RESULTS: RIDGE ALONE vs HISTGBT ALONE vs BLEND")
    report_candidate("ridge_alone", actual_oz, actual_lbs, ridge_preds)
    report_candidate("histgbt_alone", actual_oz, actual_lbs, hgb_preds)
    report_candidate(f"blend_{best_w:.1f}_{1-best_w:.1f}", actual_oz, actual_lbs, blend_preds)

    _hr("STEP 5 -- COST OF SHIPPING AN ENSEMBLE")
    print("  An ensemble is not free even when it wins on MAE:")
    print("    - Two models to train, version, and monitor for drift instead of one --")
    print("      double the retraining pipeline, double the artifacts to keep in sync.")
    print("    - Less interpretable: a single blended number no longer has one clean")
    print("      'raise theoretical weight by X' story the way Ridge's coefficients do,")
    print("      which matters for explaining a prediction to warehouse ops or Tomas.")
    print("    - Slower/heavier to serve: two model loads and two predict() calls per")
    print("      shipment instead of one.")
    print("  Any MAE gain above should be weighed against this maintenance cost before")
    print("  deciding whether to pursue it further -- this script does not make that call.")

    _hr("DONE  (exploratory only -- no model selected, nothing saved to models/)")


if __name__ == "__main__":
    main()

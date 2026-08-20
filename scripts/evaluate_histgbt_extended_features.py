"""Can HistGBT + extended features, or a finer HistGBT/extended-Ridge
blend, push rounded within-1lb% past 90%?

Context: scripts/evaluate_extended_ensemble.py found HistGBT (production)
at 89.4% rounded within-1lb, and a coarse 0.7/0.3 HistGBT/extended-Ridge
blend at 89.9% -- close to 90% but not there. That script also found
HistGBT's residuals correlate with extended-Ridge's at 0.886, suggesting
HistGBT's trees may already implicitly capture much of what the
hand-engineered extended features (distinct_sku_count,
weight_density_lbs_per_in3, top_category_by_weight, fill_ratio_winsorized,
is_overfilled, box_name_target_enc_oz, ship_method_target_enc_oz) give
Ridge explicitly -- untested claim, checked directly here by training
HistGBT itself on those features instead of just blending with a model
that has them.

Two independent levers tried:
  1. HistGBT trained directly on ALL_FEATURES_EXTENDED (never tried
     before -- the extended features have only ever been evaluated
     against Ridge). Includes a leave-one-in ablation so it's visible
     WHICH extended feature (if any) actually moves HistGBT, not just
     whether the bundle of all seven does.
  2. A finer blend-weight sweep (0.05 increments, not just 0.3/0.5/0.7)
     between production HistGBT and extended-Ridge, in case the coarse
     sweep in evaluate_extended_ensemble.py missed a better weight.

All tuning (ablation feature selection, blend weight) is chosen on a
train-fold-only validation split (Jan-Apr fit / May validate); June test
is only used for final scoring, exactly once per candidate.

Evaluation-only / exploratory: nothing in models/ is changed, no training
script is touched, no model is selected or saved.

Usage:
    python scripts/evaluate_histgbt_extended_features.py
    python scripts/evaluate_histgbt_extended_features.py --refresh
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

from shipment_weight.evaluate import regression_metrics
from shipment_weight.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from shipment_weight.features_extended import (
    ALL_FEATURES_EXTENDED,
    CATEGORICAL_FEATURES_EXTENDED,
    EXTENDED_CATEGORICAL_FEATURES,
    EXTENDED_NUMERIC_FEATURES,
    NUMERIC_FEATURES_EXTENDED,
)
from shipment_weight.rounding import round_to_billing_tier

from _experiment_prep import DEFAULT_LINES, DEFAULT_SHIPMENTS, prepare
from evaluate_extended_ensemble import build_extended_split, make_validation_split
from evaluate_rounded import rounded_metrics
from train_real_data import HISTGBT_KWARGS, LBS_TO_OZ, _hr, make_histgbt_pipeline

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROD_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model.joblib")
EXTENDED_MODEL_PATH = os.path.join(REPO_ROOT, "models", "model_extended.joblib")

TARGET_WITHIN_1LB_PCT = 90.0


def make_histgbt_pipeline_for(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Same HISTGBT_KWARGS/densification as train_real_data.make_histgbt_pipeline,
    but over an arbitrary feature list -- needed for the extended feature set
    and for the leave-one-in ablation, neither of which is ALL_FEATURES."""
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    pre = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        sparse_threshold=0.0,
    )
    return Pipeline(steps=[("preprocess", pre), ("model", HistGradientBoostingRegressor(**HISTGBT_KWARGS))])


def report(label: str, actual_oz: pd.Series, actual_lbs: pd.Series, preds_oz: np.ndarray, rows: list) -> None:
    cont = regression_metrics(actual_oz, preds_oz)
    rounded, _ = round_to_billing_tier(preds_oz / LBS_TO_OZ)
    rnd = rounded_metrics(actual_lbs.values, rounded)
    hit = rnd["within_1lb_pct"] >= TARGET_WITHIN_1LB_PCT
    marker = "  <-- >= 90%!" if hit else ""
    print(f"  {label:<32} continuous MAE {cont['mae_oz']/LBS_TO_OZ:>6.3f} lbs  |  "
          f"rounded MAE {rnd['mae_lbs']:>6.3f} lbs, within-1lb {rnd['within_1lb_pct']:>5.1f}%, "
          f"exact-match {rnd['exact_match_pct']:>5.1f}%{marker}")
    rows.append({"label": label, "rounded_within_1lb_pct": rnd["within_1lb_pct"],
                 "rounded_mae_lbs": rnd["mae_lbs"], "exact_match_pct": rnd["exact_match_pct"]})


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
    results: list[dict] = []

    _hr("BASELINES  (June test)")
    hgb_bundle = joblib.load(PROD_MODEL_PATH)
    hgb_base_preds = hgb_bundle["pipeline"].predict(test[hgb_bundle["feature_list"]])
    report("histgbt_production (base features)", actual_oz, actual_lbs, hgb_base_preds, results)

    _, test_ext = build_extended_split(train, test, lines_raw)
    ext_bundle = joblib.load(EXTENDED_MODEL_PATH)
    ext_preds = ext_bundle["pipeline"].predict(test_ext[ALL_FEATURES_EXTENDED])
    report("extended_ridge (extended features)", actual_oz, actual_lbs, ext_preds, results)

    # ── Lever 1: HistGBT trained directly on the extended feature set ───────
    _hr("LEVER 1 -- HISTGBT TRAINED ON ALL_FEATURES_EXTENDED")
    train_ext, _ = build_extended_split(train, test, lines_raw)
    X_train_ext, y_train_ext = train_ext[ALL_FEATURES_EXTENDED], train_ext[TARGET]

    hgb_ext_pipe = make_histgbt_pipeline_for(NUMERIC_FEATURES_EXTENDED, CATEGORICAL_FEATURES_EXTENDED)
    hgb_ext_pipe.fit(X_train_ext, y_train_ext)
    hgb_ext_preds = hgb_ext_pipe.predict(test_ext[ALL_FEATURES_EXTENDED])
    report("histgbt_ALL_extended_features", actual_oz, actual_lbs, hgb_ext_preds, results)

    print("\n  Leave-one-in ablation (train-fold validation: fit Jan-Apr, validate May) --")
    print("  which single extended feature, if any, actually helps HistGBT:\n")
    subtrain, val = make_validation_split(train)
    subtrain_ext, val_ext = build_extended_split(subtrain, val, lines_raw)

    base_pipe = make_histgbt_pipeline()
    base_pipe.fit(subtrain[ALL_FEATURES], subtrain[TARGET])
    base_val_mae = float(np.mean(np.abs(base_pipe.predict(val[ALL_FEATURES]) - val[TARGET].values)))
    print(f"  {'feature':<32} {'val MAE oz':>11} {'delta vs base':>14}")
    print(f"  {'(base, no extended features)':<32} {base_val_mae:>11.2f} {0.0:>+14.2f}")

    ablation_rows = []
    for feat in EXTENDED_NUMERIC_FEATURES:
        numeric = NUMERIC_FEATURES + [feat]
        pipe = make_histgbt_pipeline_for(numeric, CATEGORICAL_FEATURES)
        pipe.fit(subtrain_ext[numeric + CATEGORICAL_FEATURES], subtrain_ext[TARGET])
        mae = float(np.mean(np.abs(pipe.predict(val_ext[numeric + CATEGORICAL_FEATURES]) - val_ext[TARGET].values)))
        ablation_rows.append((feat, mae, mae - base_val_mae))
    for feat in EXTENDED_CATEGORICAL_FEATURES:
        categorical = CATEGORICAL_FEATURES + [feat]
        pipe = make_histgbt_pipeline_for(NUMERIC_FEATURES, categorical)
        pipe.fit(subtrain_ext[NUMERIC_FEATURES + categorical], subtrain_ext[TARGET])
        mae = float(np.mean(np.abs(pipe.predict(val_ext[NUMERIC_FEATURES + categorical]) - val_ext[TARGET].values)))
        ablation_rows.append((feat, mae, mae - base_val_mae))

    for feat, mae, delta in sorted(ablation_rows, key=lambda r: r[2]):
        marker = "  <- helps" if delta < -0.1 else ("  <- hurts" if delta > 0.1 else "")
        print(f"  {feat:<32} {mae:>11.2f} {delta:>+14.2f}{marker}")

    best_feat, best_feat_mae, best_feat_delta = min(ablation_rows, key=lambda r: r[2])

    # ── Lever 2: finer blend-weight sweep ────────────────────────────────────
    _hr("LEVER 2 -- FINER BLEND-WEIGHT SWEEP (0.05 increments, train-fold validation)")
    hgb_val_pipe = make_histgbt_pipeline()
    hgb_val_pipe.fit(subtrain[ALL_FEATURES], subtrain[TARGET])
    hgb_val_preds = hgb_val_pipe.predict(val[ALL_FEATURES])

    from train_real_data_extended import make_pipeline_extended
    ext_val_pipe = make_pipeline_extended()
    ext_val_pipe.fit(subtrain_ext[ALL_FEATURES_EXTENDED], subtrain_ext[TARGET])
    ext_val_preds = ext_val_pipe.predict(val_ext[ALL_FEATURES_EXTENDED])

    val_actual = val[TARGET].values
    best_w, best_mae = None, float("inf")
    for w in np.arange(0.05, 1.0, 0.05):
        blend = w * hgb_val_preds + (1 - w) * ext_val_preds
        mae = float(np.mean(np.abs(blend - val_actual)))
        if mae < best_mae:
            best_mae, best_w = mae, w
    print(f"  Best weight on validation: histgbt={best_w:.2f} / extended_ridge={1-best_w:.2f}  (val MAE {best_mae:.2f} oz)")
    blend_preds = best_w * hgb_base_preds + (1 - best_w) * ext_preds
    report(f"blend_{best_w:.2f}_{1-best_w:.2f}_fine", actual_oz, actual_lbs, blend_preds, results)

    # ── Combine both levers: extended-feature HistGBT + extended Ridge ──────
    _hr("COMBINED -- BEST OF BOTH LEVERS")
    print(f"  (Ablation found '{best_feat}' as the best single extended feature for HistGBT,")
    print(f"   delta {best_feat_delta:+.2f} oz on validation -- ", end="")
    if best_feat_delta < -0.1:
        print("adding it to production HistGBT's feature set is a real candidate.)")
        numeric = NUMERIC_FEATURES + ([best_feat] if best_feat in EXTENDED_NUMERIC_FEATURES else [])
        categorical = CATEGORICAL_FEATURES + ([best_feat] if best_feat in EXTENDED_CATEGORICAL_FEATURES else [])
        pipe = make_histgbt_pipeline_for(numeric, categorical)
        pipe.fit(train_ext[numeric + categorical], train_ext[TARGET])
        best_feat_preds = pipe.predict(test_ext[numeric + categorical])
        report(f"histgbt_+{best_feat}", actual_oz, actual_lbs, best_feat_preds, results)
        blend2 = best_w * best_feat_preds + (1 - best_w) * ext_preds
        report(f"blend_(histgbt+{best_feat})_extRidge", actual_oz, actual_lbs, blend2, results)
    else:
        print("no single feature helped enough to bother combining further.)")

    _hr("SUMMARY vs 90% ROUNDED WITHIN-1LB TARGET")
    summary = pd.DataFrame(results).sort_values("rounded_within_1lb_pct", ascending=False)
    print(summary.to_string(index=False, formatters={
        "rounded_within_1lb_pct": "{:.2f}".format, "rounded_mae_lbs": "{:.3f}".format,
        "exact_match_pct": "{:.1f}".format,
    }))
    best_row = summary.iloc[0]
    if best_row["rounded_within_1lb_pct"] >= TARGET_WITHIN_1LB_PCT:
        print(f"\n  >= 90% reached: {best_row['label']} at {best_row['rounded_within_1lb_pct']:.2f}%")
    else:
        gap = TARGET_WITHIN_1LB_PCT - best_row["rounded_within_1lb_pct"]
        print(f"\n  Best candidate ({best_row['label']}) reaches {best_row['rounded_within_1lb_pct']:.2f}%, "
              f"{gap:.2f} points short of 90%. Given the ~6.53 oz repeat-shipment noise floor")
        print(f"  (scripts/check_noise_floor.py) and how little the model-sweep/ensemble/extended-feature")
        print(f"  levers have moved this number so far, 90% likely requires new information -- e.g. a real")
        print(f"  fix for the 30x20x12 box or the extreme-item-count tail -- not another model/feature swap.")

    _hr("DONE  (exploratory only -- no model selected, nothing saved to models/)")


if __name__ == "__main__":
    main()

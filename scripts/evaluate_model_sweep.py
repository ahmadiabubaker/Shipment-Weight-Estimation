"""Broader model sweep: tuned HistGBT, LightGBM/XGBoost/CatBoost (if
installed), and a linear sanity check -- all against the noise-floor ceiling.

Context: HistGBT with absolute-error loss beats production Ridge (~10.3 oz
vs 11.34 oz June MAE), and ensembling the two barely helps (residual
correlation 0.884, see scripts/evaluate_ensemble.py). Tomas wants other/
better models considered generally. The constraint that frames all of it:
scripts/check_noise_floor.py estimates an irreducible repeat-shipment noise
floor of ~6.5 oz, so only ~4 oz of real headroom exists above current error.
The question is whether a meaningfully better model exists, not whether
another Ridge->HistGBT-sized jump does (it can't -- there isn't enough
headroom).

What runs:
  1. HistGBT(absolute_error) tuning sweep -- learning_rate, max_iter (via
     early stopping cap), max_leaf_nodes, l2_regularization -- using
     train-fold-only validation: fit on Jan-Apr, validate on May, June never
     touched during tuning (same discipline as the ensemble blend-weight
     sweep). Best config is then refit on full Jan-May and scored on June.
  2. LightGBM (objective="mae"), lightly tuned the same way -- IF installed.
  3. XGBoost (objective="reg:absoluteerror"), lightly tuned -- IF installed.
  4. CatBoost (loss_function="MAE") with NATIVE categorical handling for
     carton_type / ship_method / category_mode instead of one-hot dummies --
     IF installed. Native categorical handling is also the candidate fix for
     the sparse-category overfitting seen in Ridge (one rare category got a
     -68.9 oz coefficient).
     Missing libraries are FLAGGED in the output, not silently skipped --
     adding the dependency is a separate decision.
  5. ElasticNet/Lasso quick sanity check: confirm linear alternatives don't
     beat Ridge/HistGBT given theoretical_weight_oz's dominance.
  6. One comparison table (continuous + rounded-to-billing-tier metrics,
     reusing evaluate_rounded.py's rounding) and an explicit
     distance-to-noise-floor column so it's visible whether further
     sweeping is worth anything.

Evaluation-only / exploratory: nothing in models/ is changed, no training
script is touched, no new production model is selected or saved.

Usage:
    python scripts/evaluate_model_sweep.py
    python scripts/evaluate_model_sweep.py --refresh
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge

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

# Repeat-shipment irreducible noise floor from scripts/check_noise_floor.py
# (pooled 6.53 oz; the weight-mix-adjusted figure there is close enough that
# the verdict section calls the estimate representative). An oracle that knew
# each SKU-set's true mean weight would still average this much error.
NOISE_FLOOR_OZ = 6.53

# Baseline HistGBT config carried over from experiment_loss_and_interactions.py
# Experiment 3 -- what "untuned" means in this sweep.
HGB_BASELINE = dict(
    loss="absolute_error", max_iter=400, learning_rate=0.05,
    max_depth=None, min_samples_leaf=40, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.1, random_state=42,
)


# ── Train-fold validation split (Jan-Apr fit / May validate) ────────────────

def make_validation_split(train: pd.DataFrame):
    """Same leakage discipline as evaluate_ensemble.select_blend_weight:
    category encoding recomputed from the Jan-Apr sub-train only."""
    subtrain = train[train["order_date"].dt.month <= 4].copy()
    val = train[train["order_date"].dt.month == 5].copy()
    cmap, gmean = category_error_map(subtrain)
    subtrain["category_avg_weight_error_oz"] = apply_category_map(subtrain["category_mode"], cmap, gmean)
    val["category_avg_weight_error_oz"] = apply_category_map(val["category_mode"], cmap, gmean)
    return subtrain, val


def val_mae(pipe, subtrain, val) -> tuple[float, float]:
    t0 = time.time()
    pipe.fit(subtrain[ALL_FEATURES], subtrain[TARGET])
    fit_s = time.time() - t0
    preds = pipe.predict(val[ALL_FEATURES])
    return float(np.mean(np.abs(preds - val[TARGET].values))), fit_s


# ── Reporting ────────────────────────────────────────────────────────────────

def score_on_june(label: str, tuning: str, preds_oz: np.ndarray, test: pd.DataFrame,
                  fit_s: float, rows: list) -> None:
    cont = regression_metrics(test[TARGET], preds_oz)
    rounded, _ = round_to_billing_tier(preds_oz / LBS_TO_OZ)
    rnd = rounded_metrics(test["actual_weight_lbs"].values, rounded)
    rows.append({
        "model": label,
        "tuning": tuning,
        "mae_oz": cont["mae_oz"],
        "mae_lbs": cont["mae_oz"] / LBS_TO_OZ,
        "within_1lb_pct": cont["within_1lb_pct"],
        "rounded_mae_lbs": rnd["mae_lbs"],
        "rounded_within_1lb_pct": rnd["within_1lb_pct"],
        "exact_match_pct": rnd["exact_match_pct"],
        "oz_above_floor": cont["mae_oz"] - NOISE_FLOOR_OZ,
        "fit_s": fit_s,
    })


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shipments", default=DEFAULT_SHIPMENTS)
    parser.add_argument("--lines", default=DEFAULT_LINES)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    train, test = prepare(args.shipments, args.lines, refresh=args.refresh)
    X_train, y_train = train[ALL_FEATURES], train[TARGET]
    X_test = test[ALL_FEATURES]
    subtrain, val = make_validation_split(train)
    print(f"  Tuning split: fit Jan-Apr ({len(subtrain):,} rows), validate May "
          f"({len(val):,} rows). June test ({len(test):,} rows) untouched until final scoring.")

    results: list[dict] = []

    # ── Reference points: production Ridge + baseline HistGBT ───────────────
    _hr("REFERENCE -- PRODUCTION RIDGE AND UNTUNED HISTGBT ON JUNE")
    bundle = joblib.load(PROD_MODEL_PATH)
    t0 = time.time()
    ridge_preds = bundle["pipeline"].predict(test[bundle["feature_list"]])
    score_on_june("ridge_production", "none (saved artifact)", ridge_preds, test, float("nan"), results)

    hgb_base = pipeline_for(HistGradientBoostingRegressor(**HGB_BASELINE),
                            NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True)
    t0 = time.time()
    hgb_base.fit(X_train, y_train)
    base_fit_s = time.time() - t0
    score_on_june("histgbt_abs_err_baseline", "defaults from loss experiment",
                  hgb_base.predict(X_test), test, base_fit_s, results)
    print(f"  ridge_production June MAE       : {results[0]['mae_oz']:.2f} oz")
    print(f"  histgbt_baseline June MAE       : {results[1]['mae_oz']:.2f} oz  (fit {base_fit_s:.0f}s)")
    print(f"  noise floor (check_noise_floor) : {NOISE_FLOOR_OZ:.2f} oz")

    # ── 1. HistGBT tuning sweep (train-fold validation only) ────────────────
    _hr("1 -- HISTGBT TUNING SWEEP  (fit Jan-Apr, validate May)")
    print("  Stage B: learning_rate x max_leaf_nodes grid (max_iter capped at 1000,")
    print("  early stopping picks the effective iteration count; l2 fixed at 1.0).\n")

    def hgb(**overrides):
        cfg = dict(HGB_BASELINE)
        cfg.update(overrides)
        return pipeline_for(HistGradientBoostingRegressor(**cfg),
                            NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True)

    sweep_rows = []
    base_val_mae, _ = val_mae(hgb(), subtrain, val)
    sweep_rows.append(("baseline (lr=0.05, leaves=31, iter<=400, l2=1.0)", base_val_mae))
    print(f"  {'config':<48} {'May MAE oz':>11}")
    print(f"  {'baseline (lr=0.05, leaves=31, iter<=400, l2=1.0)':<48} {base_val_mae:>11.2f}")

    best_cfg = {}
    best_val = base_val_mae
    for lr in (0.05, 0.1):
        for leaves in (31, 63, 127):
            cfg = dict(learning_rate=lr, max_leaf_nodes=leaves, max_iter=1000)
            mae, fit_s = val_mae(hgb(**cfg), subtrain, val)
            label = f"lr={lr}, leaves={leaves}, iter<=1000, l2=1.0"
            sweep_rows.append((label, mae))
            marker = ""
            if mae < best_val:
                best_val, best_cfg = mae, cfg
                marker = "  <- best so far"
            print(f"  {label:<48} {mae:>11.2f}{marker}")

    print("\n  Stage C: l2_regularization sweep around the Stage B winner.\n")
    for l2 in (0.0, 0.1, 10.0):
        cfg = dict(best_cfg or {}, l2_regularization=l2)
        mae, fit_s = val_mae(hgb(**cfg), subtrain, val)
        label = f"best + l2={l2}"
        marker = ""
        if mae < best_val:
            best_val, best_cfg = mae, cfg
            marker = "  <- best so far"
        print(f"  {label:<48} {mae:>11.2f}{marker}")

    tuned_desc = ", ".join(f"{k}={v}" for k, v in best_cfg.items()) or "baseline config already best"
    print(f"\n  Winning config on May validation: {tuned_desc}  (May MAE {best_val:.2f} oz)")

    print("  Refitting winner on full Jan-May, scoring June...")
    tuned_pipe = hgb(**best_cfg)
    t0 = time.time()
    tuned_pipe.fit(X_train, y_train)
    tuned_fit_s = time.time() - t0
    score_on_june("histgbt_abs_err_TUNED", f"val-sweep: {tuned_desc}",
                  tuned_pipe.predict(X_test), test, tuned_fit_s, results)
    print(f"  histgbt_TUNED June MAE: {results[-1]['mae_oz']:.2f} oz")

    # ── 2-4. LightGBM / XGBoost / CatBoost (if installed) ───────────────────
    _hr("2-4 -- LIGHTGBM / XGBOOST / CATBOOST  (skipped loudly if not installed)")
    missing: list[str] = []

    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
        missing.append("lightgbm")
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
        missing.append("xgboost")
    try:
        import catboost as cb
    except ImportError:
        cb = None
        missing.append("catboost")

    if lgb is not None:
        print("  LightGBM: small sweep (learning_rate x num_leaves), objective='mae'")
        best_lgb_val, best_lgb_params = float("inf"), {}
        for lr in (0.05, 0.1):
            for leaves in (31, 63, 127):
                params = dict(objective="mae", n_estimators=1000, learning_rate=lr,
                              num_leaves=leaves, min_child_samples=40, reg_lambda=1.0,
                              random_state=42, verbosity=-1)
                pipe = pipeline_for(lgb.LGBMRegressor(**params),
                                    NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True)
                mae, _ = val_mae(pipe, subtrain, val)
                print(f"    lr={lr}, leaves={leaves}: May MAE {mae:.2f} oz")
                if mae < best_lgb_val:
                    best_lgb_val, best_lgb_params = mae, params
        pipe = pipeline_for(lgb.LGBMRegressor(**best_lgb_params),
                            NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        score_on_june("lightgbm_mae", f"val-sweep: lr={best_lgb_params['learning_rate']}, "
                      f"leaves={best_lgb_params['num_leaves']}",
                      pipe.predict(X_test), test, time.time() - t0, results)

    if xgb is not None:
        print("  XGBoost: small sweep (learning_rate x max_depth), objective='reg:absoluteerror'")
        best_xgb_val, best_xgb_params = float("inf"), {}
        for lr in (0.05, 0.1):
            for depth in (6, 8):
                params = dict(objective="reg:absoluteerror", n_estimators=600,
                              learning_rate=lr, max_depth=depth, min_child_weight=40,
                              reg_lambda=1.0, random_state=42, verbosity=0)
                pipe = pipeline_for(xgb.XGBRegressor(**params),
                                    NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True)
                mae, _ = val_mae(pipe, subtrain, val)
                print(f"    lr={lr}, depth={depth}: May MAE {mae:.2f} oz")
                if mae < best_xgb_val:
                    best_xgb_val, best_xgb_params = mae, params
        pipe = pipeline_for(xgb.XGBRegressor(**best_xgb_params),
                            NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        score_on_june("xgboost_mae", f"val-sweep: lr={best_xgb_params['learning_rate']}, "
                      f"depth={best_xgb_params['max_depth']}",
                      pipe.predict(X_test), test, time.time() - t0, results)

    if cb is not None:
        print("  CatBoost: loss_function='MAE' with NATIVE categorical handling for")
        print("  carton_type / ship_method / category_mode (no one-hot dummies) --")
        print("  the candidate fix for sparse-category overfitting (-68.9 oz Ridge coef).")
        cat_cols = ["carton_type", "ship_method", "category_mode"]
        num_cols = [c for c in NUMERIC_FEATURES]

        def cb_frames(df):
            X = df[num_cols + cat_cols].copy()
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
            X[cat_cols] = X[cat_cols].astype(str)
            return X

        best_cb_val, best_cb_params = float("inf"), {}
        for lr in (0.05, 0.1):
            for depth in (6, 8):
                model = cb.CatBoostRegressor(
                    loss_function="MAE", iterations=800, learning_rate=lr, depth=depth,
                    l2_leaf_reg=3.0, random_seed=42, verbose=False,
                    cat_features=cat_cols,
                )
                t0 = time.time()
                model.fit(cb_frames(subtrain), subtrain[TARGET])
                preds = model.predict(cb_frames(val))
                mae = float(np.mean(np.abs(preds - val[TARGET].values)))
                print(f"    lr={lr}, depth={depth}: May MAE {mae:.2f} oz")
                if mae < best_cb_val:
                    best_cb_val, best_cb_params = mae, dict(learning_rate=lr, depth=depth)
        model = cb.CatBoostRegressor(
            loss_function="MAE", iterations=800, l2_leaf_reg=3.0, random_seed=42,
            verbose=False, cat_features=cat_cols, **best_cb_params,
        )
        t0 = time.time()
        model.fit(cb_frames(train), y_train)
        score_on_june("catboost_mae_native_cats",
                      f"val-sweep: lr={best_cb_params['learning_rate']}, depth={best_cb_params['depth']}",
                      model.predict(cb_frames(test)), test, time.time() - t0, results)

    if missing:
        print()
        for lib in missing:
            print(f"  [NOT INSTALLED] {lib} -- skipped. Install it and rerun this script to")
            print(f"                  include it; adding the dependency is your call.")

    # ── 5. Linear sanity check ──────────────────────────────────────────────
    _hr("5 -- LINEAR SANITY CHECK  (ElasticNet / Lasso; quick, not tuned hard)")
    print("  Expectation: neither beats Ridge -- theoretical_weight_oz dominates and")
    print("  Ridge already handles the mild collinearity. This is a confirmation, not a search.\n")
    for label, est in [
        ("elasticnet_a0.1_l1r0.5", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ("elasticnet_a1.0_l1r0.5", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)),
        ("lasso_a0.1", Lasso(alpha=0.1, max_iter=5000)),
    ]:
        mae, _ = val_mae(make_pipeline(est), subtrain, val)
        print(f"  {label:<28} May MAE {mae:.2f} oz")
    # Only the best linear variant gets June scoring (they're a sanity check,
    # not candidates): pick by May MAE.
    linear_candidates = {
        "elasticnet_a0.1_l1r0.5": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "lasso_a0.1": Lasso(alpha=0.1, max_iter=5000),
    }
    best_lin_label, best_lin_mae, best_lin_est = None, float("inf"), None
    for label, est in linear_candidates.items():
        mae, _ = val_mae(make_pipeline(est), subtrain, val)
        if mae < best_lin_mae:
            best_lin_label, best_lin_mae, best_lin_est = label, mae, est
    pipe = make_pipeline(best_lin_est)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    score_on_june(best_lin_label, "quick sanity check (best of small linear set)",
                  pipe.predict(X_test), test, time.time() - t0, results)

    # ── 6. Comparison table + noise-floor framing ───────────────────────────
    _hr("6 -- COMPARISON TABLE  (June test; rounded metrics per evaluate_rounded.py)")
    table = pd.DataFrame(results).sort_values("mae_oz")
    print(table.to_string(index=False, formatters={
        "mae_oz": "{:.2f}".format, "mae_lbs": "{:.3f}".format,
        "within_1lb_pct": "{:.1f}".format, "rounded_mae_lbs": "{:.3f}".format,
        "rounded_within_1lb_pct": "{:.1f}".format, "exact_match_pct": "{:.1f}".format,
        "oz_above_floor": "{:+.2f}".format, "fit_s": "{:.0f}".format,
    }))

    best = table.iloc[0]
    print(f"\n  Noise floor (irreducible, from check_noise_floor.py): {NOISE_FLOOR_OZ:.2f} oz")
    print(f"  Best model here: {best['model']} at {best['mae_oz']:.2f} oz "
          f"= {best['oz_above_floor']:.2f} oz above the floor.")
    print(f"  Of the ~{results[1]['mae_oz'] - NOISE_FLOOR_OZ:.1f} oz of headroom the untuned")
    print(f"  HistGBT had, this sweep recovered "
          f"{results[1]['mae_oz'] - best['mae_oz']:.2f} oz.")
    print("  Anything still on the table is within a couple of ounces of the practical")
    print("  ceiling -- judge further sweeping (or new dependencies) against that, not")
    print("  against the Ridge->HistGBT jump, which cannot happen again.")

    _hr("DONE  (exploratory only -- no model selected, nothing saved to models/)")


if __name__ == "__main__":
    main()

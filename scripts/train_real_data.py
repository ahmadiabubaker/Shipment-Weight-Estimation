"""Retrain shipment-weight model on real warehouse data (Jan–Jun 2026).

Pipeline stages (all in one run):
  1. Load   — read both Excel files
  2. Clean  — drop bad ship methods, non-positive actual weights, extreme outliers
  3. Feats  — join order lines → per-shipment aggregates → model columns
  4. Split  — time-based: train = Jan–May 2026, test = June 2026
  5. Encode — compute category_avg_weight_error_oz from training fold only
  6. Train  — linear/ridge/random_forest/gradient_boosted_trees (MODEL_CANDIDATES)
              plus histgbt_absolute_error (own dense pipeline, see
              make_histgbt_pipeline) vs theoretical-weight baseline
  7. Eval   — MAE/RMSE/bias overall; sliced by box_name, ship_method, item_count
  8. Save   — bundle → models/model.joblib  (version v0.5.0-histgbt)

Usage:
    python scripts/train_real_data.py \\
        --shipments path/to/order_shipments_anonymized.xlsx \\
        --lines     path/to/order_lines_in_shipment_anonymized.xlsx \\
        [--out      models/model.joblib]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shipment_weight.evaluate import (
    bias_by_item_count_bucket,
    bias_by_segment,
    compare_to_baseline,
    largest_errors,
    regression_metrics,
)
from shipment_weight.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, add_derived_features
from shipment_weight.ingest import LBS_TO_OZ, apply_category_map
from shipment_weight.ingest import build_features as _build_features_shared
from shipment_weight.train import MODEL_CANDIDATES, make_pipeline

MODEL_VERSION = "v0.5.0-histgbt"
# histgbt_absolute_error beat ridge on both average error (0.643 vs 0.709 lbs
# MAE, 84.4% vs 81.2% within-1lb) and on a full diagnostic pass over Ridge's
# known weak spots (scripts/evaluate_histgbt_diagnostics.py): it largely
# fixes the FedEx HAZMAT bias (+10.77 -> -2.31 oz) and improves low-item-count
# error, at the cost of one small new regression (box 26x20x8, 49 rows/0.5%
# of test) and no fix for the 30x20x12 box, which remains the dominant error
# source for both models. See MODEL_CARD.md "Model Selection" for the full
# story and scripts/evaluate_model_sweep.py for why no other library (tuned
# HistGBT, LightGBM, XGBoost, CatBoost) beat this untuned baseline -- 4
# boosting libraries converged within 0.28 oz of each other and ~3.8 oz of
# the ~6.5 oz repeat-shipment noise floor (scripts/check_noise_floor.py).
PREFERRED_MODEL = "histgbt_absolute_error"

# Same untuned config that won scripts/evaluate_model_sweep.py's tuning sweep
# (tuning bought ~0.00 oz over these defaults, so they're kept as-is here).
HISTGBT_KWARGS = dict(
    loss="absolute_error", max_iter=400, learning_rate=0.05,
    max_depth=None, min_samples_leaf=40, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.1, random_state=42,
)


def make_histgbt_pipeline() -> Pipeline:
    """Separate from shipment_weight.train.make_pipeline (used by the other
    MODEL_CANDIDATES) because HistGradientBoostingRegressor cannot consume a
    sparse matrix, unlike linear/ridge/random_forest/gradient_boosted_trees --
    same NUMERIC_FEATURES/CATEGORICAL_FEATURES from shipment_weight.features
    and the same X_train/y_train built from ingest.py either way, just a
    densified ColumnTransformer for this one estimator (sparse_threshold=0.0
    forces the one-hot block dense; harmless for tree splits, just costs a
    little more memory for ~9-12 dummy columns)."""
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", HistGradientBoostingRegressor(**HISTGBT_KWARGS))])


# Catches: "Pick Up At Medusa", "Pick Up at Medusa", "Pickup At Medusa",
#          "CanceledItem", "Ship Outside System", "Ship Outside System Int"
# \s? handles "Pick Up" vs "Pickup"; case=False on str.contains handles capitalisation
BAD_METHOD_RE = r"Pick\s?Up\s+At\s+Medusa|CanceledItem|Ship Outside System|LTL"


# ── helpers ───────────────────────────────────────────────────────────────────

def _hr(title: str) -> None:
    print(f"\n{'='*62}\n{title}\n{'='*62}")


# ── Stage 1: Load ─────────────────────────────────────────────────────────────

def load_data(shipments_path: str, lines_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _hr("STAGE 1 — LOAD")
    ships = pd.read_excel(shipments_path, engine="openpyxl")
    print(f"  Shipments loaded  : {len(ships):>10,} rows")
    lines = pd.read_excel(lines_path, engine="openpyxl")
    print(f"  Order lines loaded: {len(lines):>10,} rows")
    return ships, lines


# ── Stage 2: Clean ────────────────────────────────────────────────────────────

def clean(ships_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop unusable rows; return cleaned df and a row-count exclusion log."""
    _hr("STAGE 2 — CLEAN")
    log: dict[str, int | float] = {"initial": len(ships_raw)}
    df = ships_raw.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # 1. Null actual weight
    before = len(df)
    m = df["actual_weight_lbs"].isnull()
    df = df[~m]
    log["dropped_null_actual_weight"] = int(m.sum())
    print(f"  1. Null actual_weight_lbs     : dropped {m.sum():>6,}  ({before:,} → {len(df):,})")

    # 2. Non-positive actual weight
    before = len(df)
    m = df["actual_weight_lbs"] <= 0
    df = df[~m]
    log["dropped_nonpositive_actual_weight"] = int(m.sum())
    print(f"  2. Non-positive actual weight : dropped {m.sum():>6,}  ({before:,} → {len(df):,})")

    # 3. Bad ship methods (exact + keyword variants, case-insensitive)
    before = len(df)
    ltl_in_raw = int(ships_raw["ship_method"].str.contains(r"LTL", case=False, na=False).sum())
    print(f"  LTL rows in raw dataset      : {ltl_in_raw:,}  ({ltl_in_raw / log['initial'] * 100:.2f}% of {log['initial']:,} total)")
    m = df["ship_method"].str.contains(BAD_METHOD_RE, case=False, na=False)
    log["dropped_bad_ship_method"] = int(m.sum())
    if m.sum():
        print(f"  3. Bad ship methods           : dropped {m.sum():>6,}  ({before:,} → {before - m.sum():,})")
        print(df.loc[m, "ship_method"].value_counts().to_string())
    else:
        print(f"  3. Bad ship methods           : dropped      0  ({before:,} → {before:,})")
    df = df[~m]

    # 4. Outliers: actual > 3× theoretical AND actual > 50 lbs.
    #    Catches data errors without discarding legitimately heavy shipments.
    before = len(df)
    m = (
        (df["actual_weight_lbs"] > 3 * df["total_theoretical_shipment_weight_lbs"])
        & (df["actual_weight_lbs"] > 50)
    )
    log["dropped_outliers"] = int(m.sum())
    if m.sum():
        outliers = (
            df[m][["shipment_number", "ship_method", "box_name",
                   "total_theoretical_shipment_weight_lbs", "actual_weight_lbs"]]
            .rename(columns={"total_theoretical_shipment_weight_lbs": "theoretical_lbs"})
            .sort_values("actual_weight_lbs", ascending=False)
        )
        print(f"\n  4. Outliers (actual > 3× theoretical AND > 50 lbs): dropped {m.sum():,}  ({before:,} → {before - m.sum():,})")
        print(outliers.to_string(index=False))
    else:
        print(f"\n  4. Outliers (actual > 3× theoretical AND > 50 lbs): dropped      0  ({before:,} → {before:,})")
    df = df[~m]

    log["final"] = len(df)

    _hr("Exclusion summary")
    for k, v in log.items():
        if isinstance(v, int):
            print(f"  {k:<42}: {v:>8,}")
        else:
            print(f"  {k:<42}: {v:>8.2f}")
    retained_pct = log["final"] / log["initial"] * 100
    print(f"\n  Retained {log['final']:,} of {log['initial']:,} rows  ({retained_pct:.1f}%)")
    return df, log


# ── Stage 3: Feature engineering ──────────────────────────────────────────────
#
# The actual column-building logic (aggregate_lines, build_features) now
# lives in shipment_weight.ingest so shipment_weight.predict can reuse it
# unchanged for live inference -- no second, simplified feature path. This
# wrapper only adds the operator-facing diagnostic prints this script has
# always produced around that shared logic.

def build_features(ships: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    _hr("STAGE 3 — FEATURE ENGINEERING")

    dup_lines = int(lines.duplicated(subset=["shipment_number", "item_id"]).sum())
    if dup_lines:
        print(f"  Deduplicated {dup_lines:,} duplicate (shipment_number, item_id) rows in order lines")

    line_shipment_ids = set(lines["shipment_number"].dropna().unique())
    print(f"  Order-line aggregates: {len(line_shipment_ids):,} unique shipment_numbers")
    no_lines = int((~ships["shipment_number"].isin(line_shipment_ids)).sum())
    print(f"  Shipments with no matching order lines: {no_lines:,}  (item_count filled with 1)")

    null_box = int(ships["box_name"].isnull().sum())

    df = _build_features_shared(ships, lines)

    if null_box:
        print(f"  Filled {null_box:,} null box_name values with dimension strings")
    print(f"  Feature dataframe shape: {df.shape}")
    return df


# ── Stage 4: Time split ───────────────────────────────────────────────────────

def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _hr("STAGE 4 — TIME SPLIT  (train = Jan–May  |  test = June)")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    train = df[df["order_date"].dt.month <= 5].reset_index(drop=True)
    test = df[df["order_date"].dt.month == 6].reset_index(drop=True)
    print(f"  Train rows (Jan–May) : {len(train):>10,}")
    print(f"  Test rows  (June)    : {len(test):>10,}")
    null_dates = df["order_date"].isnull().sum()
    if null_dates:
        print(f"  Warning: {null_dates:,} rows have unparseable order_date — excluded from both sets")
    return train, test


# ── Stage 5: Category error encoding ─────────────────────────────────────────

def category_error_map(train: pd.DataFrame) -> tuple[dict, float]:
    """Category -> mean training residual (oz), plus the global fallback
    mean for categories unseen in train. Factored out of
    encode_category_error so Stage 8 can persist the map itself into the
    saved model bundle -- shipment_weight.predict applies it the same way
    (via apply_category_map) at serving time, using apply_category_map so
    callers never have to supply category_avg_weight_error_oz by hand.
    """
    train_error = train["actual_weight_oz"] - train["theoretical_weight_oz"]
    cat_map = (
        pd.DataFrame({"cat": train["category_mode"], "err": train_error})
        .groupby("cat")["err"]
        .mean()
    )
    return cat_map.to_dict(), float(train_error.mean())


def encode_category_error(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target-encode category_mode with mean training error (oz).

    Computed on train only to prevent leakage into the June test set.
    Unknown categories in test fall back to the global training mean.
    """
    _hr("STAGE 5 — CATEGORY ERROR ENCODING  (train-fold only)")
    cat_map, global_mean = category_error_map(train)
    print(f"  Unique categories in train: {len(cat_map):,}")
    print(f"  Global mean error (train) : {global_mean:.2f} oz  ({global_mean/LBS_TO_OZ:.3f} lbs)")
    n_unseen = test["category_mode"].isin(cat_map.keys()).eq(False).sum()
    print(f"  Test categories unseen in train (→ global mean): {n_unseen:,} rows")

    train = train.copy()
    test = test.copy()
    train["category_avg_weight_error_oz"] = apply_category_map(train["category_mode"], cat_map, global_mean)
    test["category_avg_weight_error_oz"] = apply_category_map(test["category_mode"], cat_map, global_mean)
    return train, test


# ── Stage 6-7: Train + Evaluate ───────────────────────────────────────────────

def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_path: str,
    cat_error_map: dict,
    cat_error_global_mean: float,
) -> None:
    # Derive fill_ratio, weight_per_item_oz, etc.
    train = add_derived_features(train)
    test = add_derived_features(test)

    X_train = train[ALL_FEATURES]
    y_train = train[TARGET]
    X_test = test[ALL_FEATURES]
    y_test = test[TARGET]
    theoretical_test_oz = test["theoretical_weight_oz"]

    # ── Model comparison ──────────────────────────────────────────────────────
    _hr("STAGE 6 — MODEL COMPARISON  (test = June 2026)")
    header = (f"{'Model':<32} {'MAE oz':>8} {'MAE lbs':>8} {'RMSE oz':>8} {'Bias oz':>9} "
              f"{'≤0.5oz%':>9} {'≤0.3lb%':>9} {'≤0.5lb%':>9} {'≤1.0lb%':>9}")
    print(header)
    print("-" * len(header))

    base = regression_metrics(y_test, theoretical_test_oz.values)
    print(f"  {'theoretical_baseline':<30} {base['mae_oz']:>8.2f} {base['mae_oz']/LBS_TO_OZ:>8.3f} "
          f"{base['rmse_oz']:>8.2f} {base['bias_oz']:>9.2f} {base['within_0_5oz_pct']:>8.1f}% "
          f"{base['within_0_3lb_pct']:>8.1f}% {base['within_0_5lb_pct']:>8.1f}% {base['within_1lb_pct']:>8.1f}%")

    best_pipeline, best_preds, best_name, best_mae = None, None, "", float("inf")
    all_results: dict[str, tuple] = {}

    for name, estimator in MODEL_CANDIDATES.items():
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        m = regression_metrics(y_test, preds)
        all_results[name] = (pipe, preds, m)
        marker = ""
        if m["mae_oz"] < best_mae:
            best_mae = m["mae_oz"]
            best_pipeline, best_preds, best_name = pipe, preds, name
            marker = " ◀ best"
        print(f"  {name:<30} {m['mae_oz']:>8.2f} {m['mae_oz']/LBS_TO_OZ:>8.3f} "
              f"{m['rmse_oz']:>8.2f} {m['bias_oz']:>9.2f} {m['within_0_5oz_pct']:>8.1f}% "
              f"{m['within_0_3lb_pct']:>8.1f}% {m['within_0_5lb_pct']:>8.1f}% {m['within_1lb_pct']:>8.1f}%{marker}")

    # histgbt_absolute_error isn't part of MODEL_CANDIDATES (see
    # make_histgbt_pipeline's docstring for why it needs its own dense
    # pipeline) but goes through the exact same X_train/y_train/X_test built
    # from ingest.py/features.py above, and is scored/compared the same way.
    histgbt_pipeline = make_histgbt_pipeline()
    histgbt_pipeline.fit(X_train, y_train)
    histgbt_preds = histgbt_pipeline.predict(X_test)
    histgbt_m = regression_metrics(y_test, histgbt_preds)
    all_results["histgbt_absolute_error"] = (histgbt_pipeline, histgbt_preds, histgbt_m)
    marker = ""
    if histgbt_m["mae_oz"] < best_mae:
        best_mae = histgbt_m["mae_oz"]
        best_pipeline, best_preds, best_name = histgbt_pipeline, histgbt_preds, "histgbt_absolute_error"
        marker = " ◀ best"
    print(f"  {'histgbt_absolute_error':<30} {histgbt_m['mae_oz']:>8.2f} {histgbt_m['mae_oz']/LBS_TO_OZ:>8.3f} "
          f"{histgbt_m['rmse_oz']:>8.2f} {histgbt_m['bias_oz']:>9.2f} {histgbt_m['within_0_5oz_pct']:>8.1f}% "
          f"{histgbt_m['within_0_3lb_pct']:>8.1f}% {histgbt_m['within_0_5lb_pct']:>8.1f}% {histgbt_m['within_1lb_pct']:>8.1f}%{marker}")

    # ── Preferred-model override for save + detailed eval ────────────────────
    if PREFERRED_MODEL in all_results:
        if best_name != PREFERRED_MODEL:
            print(f"\n  Preferred model override: saving '{PREFERRED_MODEL}' "
                  f"(MAE {all_results[PREFERRED_MODEL][2]['mae_oz']:.2f} oz) "
                  f"instead of best-by-MAE '{best_name}' ({best_mae:.2f} oz)")
        best_name = PREFERRED_MODEL
        best_pipeline, best_preds, _ = all_results[PREFERRED_MODEL]

    # ── Threshold sanity check ────────────────────────────────────────────────
    gbt_name = next(
        (n for n in all_results if "gbt" in n.lower() or "gradient" in n.lower()),
        best_name,
    )
    _, gbt_preds, _ = all_results[gbt_name]
    abs_errs = np.abs(gbt_preds - y_test.values)
    print(f"\n  Threshold sanity check [{gbt_name}]  (n={len(abs_errs):,} test rows):")
    print(f"    threshold in evaluate.py        : 0.5 oz  (abs_error <= 0.5)")
    print(f"    % rows with abs_error <= 0.5 oz : {(abs_errs <= 0.5).mean()*100:.2f}%")
    print(f"    % rows with abs_error <= 1.0 oz : {(abs_errs <= 1.0).mean()*100:.2f}%")
    print(f"    % rows with abs_error <= 2.0 oz : {(abs_errs <= 2.0).mean()*100:.2f}%")
    print(f"    % rows with abs_error <= 16.0 oz: {(abs_errs <= 16.0).mean()*100:.2f}%")
    print(f"    within_0_5oz_pct from metrics() : {all_results[gbt_name][2]['within_0_5oz_pct']:.2f}%")
    print(f"    within_1lb_pct    from metrics() : {all_results[gbt_name][2]['within_1lb_pct']:.2f}%")

    # ── Detailed eval of best model ───────────────────────────────────────────
    best_m = all_results[best_name][2]
    _hr(f"STAGE 7 — DETAILED EVALUATION  [{best_name}]")
    print(f"  MAE  : {best_m['mae_oz']:>8.2f} oz   ({best_m['mae_oz']/LBS_TO_OZ:.3f} lbs)")
    print(f"  RMSE : {best_m['rmse_oz']:>8.2f} oz   ({best_m['rmse_oz']/LBS_TO_OZ:.3f} lbs)")
    print(f"  Bias : {best_m['bias_oz']:>8.2f} oz   ({best_m['bias_oz']/LBS_TO_OZ:.3f} lbs)")
    print(f"  Within 2 oz: {best_m['within_2oz_pct']:.1f}%")
    print(f"\n  Baseline (theoretical weight) bias: {base['bias_oz']:.2f} oz  "
          f"(+{base['bias_oz']/LBS_TO_OZ:.3f} lbs — model reduces this to "
          f"{best_m['bias_oz']:.2f} oz)")

    print(f"\n  — Bias by SHIP METHOD —")
    ship_bias = bias_by_segment(
        y_test, best_preds, test["ship_method"].reset_index(drop=True), "ship_method"
    )
    print(ship_bias.to_string(index=False))

    print(f"\n  — Bias by BOX NAME (top 20 by volume) —")
    box_bias = bias_by_segment(
        y_test, best_preds, test["carton_type"].reset_index(drop=True), "box_name"
    )
    print(box_bias.head(20).to_string(index=False))

    print(f"\n  — Bias by ITEM COUNT BUCKET —")
    item_bias = bias_by_item_count_bucket(
        y_test, best_preds, test["item_count"].reset_index(drop=True)
    )
    print(item_bias.to_string(index=False))

    print(f"\n  — LARGEST ERRORS (top 20, post-outlier-removal) —")
    context_cols = ["theoretical_weight_oz", "carton_type", "ship_method",
                    "item_count", "category_mode"]
    X_test_ctx = X_test.copy().reset_index(drop=True)
    for col in context_cols:
        if col not in X_test_ctx.columns and col in test.columns:
            X_test_ctx[col] = test[col].values
    big_err = largest_errors(X_test_ctx, y_test.reset_index(drop=True), best_preds)
    display_cols = [c for c in [
        "theoretical_weight_oz", "carton_type", "ship_method", "item_count",
        "actual_weight_oz", "predicted_weight_oz", "error_oz", "abs_error_oz",
    ] if c in big_err.columns]
    print(big_err[display_cols].to_string(index=False))

    # ── Stage 8: Save model ───────────────────────────────────────────────────
    _hr("STAGE 8 — SAVE MODEL ARTIFACT")
    residuals = y_test.values - best_preds
    bundle = {
        "pipeline": best_pipeline,
        "model_type": best_name,
        "model_version": MODEL_VERSION,
        "residual_std": float(residuals.std()),
        "trained_on_rows": len(X_train),
        "feature_list": list(ALL_FEATURES),
        # category_avg_weight_error_oz's train-fold map, baked into the
        # artifact so shipment_weight.predict can apply it at serving time
        # (via apply_category_map) instead of requiring callers to supply
        # this value themselves -- it was never exposed anywhere before.
        "category_error_map": cat_error_map,
        "category_error_global_mean": cat_error_global_mean,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"  Saved to      : {out_path}")
    print(f"  model_version : {MODEL_VERSION}")
    print(f"  model_type    : {best_name}")
    print(f"  training rows : {len(X_train):,}")
    print(f"  residual_std  : {residuals.std():.2f} oz  ({residuals.std()/LBS_TO_OZ:.3f} lbs)")
    print(f"  category_error_map: {len(cat_error_map):,} categories  (+global fallback mean)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full retraining pipeline on real warehouse data."
    )
    parser.add_argument("--shipments", required=True,
                        help="Path to order_shipments_anonymized.xlsx")
    parser.add_argument("--lines", required=True,
                        help="Path to order_lines_in_shipment_anonymized.xlsx")
    parser.add_argument("--out", default="models/model.joblib",
                        help="Output path for model bundle (default: models/model.joblib)")
    args = parser.parse_args()

    ships_raw, lines_raw = load_data(args.shipments, args.lines)
    ships, _ = clean(ships_raw)
    df = build_features(ships, lines_raw)
    train, test = time_split(df)
    cat_map, cat_global_mean = category_error_map(train)
    train, test = encode_category_error(train, test)
    train_and_evaluate(train, test, args.out, cat_map, cat_global_mean)

    _hr("DONE")


if __name__ == "__main__":
    main()

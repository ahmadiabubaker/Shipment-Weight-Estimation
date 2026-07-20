"""Retrain shipment-weight model on real warehouse data (Jan–Jun 2026).

Pipeline stages (all in one run):
  1. Load   — read both Excel files
  2. Clean  — drop bad ship methods, non-positive actual weights, extreme outliers
  3. Feats  — join order lines → per-shipment aggregates → model columns
  4. Split  — time-based: train = Jan–May 2026, test = June 2026
  5. Encode — compute category_avg_weight_error_oz from training fold only
  6. Train  — all 4 model candidates vs theoretical-weight baseline
  7. Eval   — MAE/RMSE/bias overall; sliced by box_name, ship_method, item_count
  8. Save   — GBT bundle → models/model.joblib  (version v0.2.0-real-data)

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shipment_weight.evaluate import (
    bias_by_item_count_bucket,
    bias_by_segment,
    compare_to_baseline,
    largest_errors,
    regression_metrics,
)
from shipment_weight.features import ALL_FEATURES, TARGET, add_derived_features
from shipment_weight.train import MODEL_CANDIDATES, make_pipeline

MODEL_VERSION = "v0.4.0-real-data"
PREFERRED_MODEL = "ridge"  # Ridge outperforms GBT on this near-linear calibration problem
LBS_TO_OZ = 16.0

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

def _safe_mode(s: pd.Series) -> str:
    m = s.dropna().mode()
    return str(m.iloc[0]) if len(m) > 0 else "unknown"


def aggregate_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse order-line rows to one row per shipment_number.

    2,153 duplicate (shipment_number, item_id) pairs exist in the raw data
    (audit finding). These are treated as double-entry errors; we keep the
    first occurrence so quantities are not double-counted.
    """
    lines = lines.copy()
    before = len(lines)
    lines = lines.drop_duplicates(subset=["shipment_number", "item_id"], keep="first")
    dropped = before - len(lines)
    if dropped:
        print(f"  Deduplicated {dropped:,} duplicate (shipment_number, item_id) rows in order lines")
    lines["item_volume_in3"] = (
        lines["item_width_inches"].fillna(0)
        * lines["item_length_inches"].fillna(0)
        * lines["item_height_inches"].fillna(0)
        * lines["item_quantity"].fillna(1)
    )
    lines["missing_wt"] = lines["theoretical_item_weight_lbs"].isnull().astype(int)

    return (
        lines.groupby("shipment_number")
        .agg(
            item_count=("item_quantity", "sum"),
            distinct_sku_count=("item_id", "nunique"),
            total_item_volume_in3=("item_volume_in3", "sum"),
            category_mode=("category", _safe_mode),
            item_categories=("category", lambda s: ",".join(sorted(s.dropna().unique()))),
            num_missing_catalog_weights=("missing_wt", "sum"),
        )
        .reset_index()
    )


def build_features(ships: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    _hr("STAGE 3 — FEATURE ENGINEERING")
    agg = aggregate_lines(lines)
    print(f"  Order-line aggregates: {len(agg):,} unique shipment_numbers")

    df = ships.merge(agg, on="shipment_number", how="left")
    no_lines = df["item_count"].isnull().sum()
    print(f"  Shipments with no matching order lines: {no_lines:,}  (item_count filled with 1)")
    df["item_count"] = df["item_count"].fillna(1).clip(lower=1)
    df["total_item_volume_in3"] = df["total_item_volume_in3"].fillna(0)
    df["num_missing_catalog_weights"] = df["num_missing_catalog_weights"].fillna(0)
    df["item_categories"] = df["item_categories"].fillna("unknown")
    df["category_mode"] = df["category_mode"].fillna("unknown")

    # Unit conversions lbs → oz
    df["theoretical_weight_oz"] = df["total_theoretical_shipment_weight_lbs"] * LBS_TO_OZ
    df["actual_weight_oz"] = df["actual_weight_lbs"] * LBS_TO_OZ

    # Fill null box_name with "{length}x{width}x{height}" before any carton features
    null_box = df["box_name"].isnull()
    if null_box.any():
        df.loc[null_box, "box_name"] = (
            df.loc[null_box, "box_length"].fillna(0).round(0).astype(int).astype(str) + "x" +
            df.loc[null_box, "box_width"].fillna(0).round(0).astype(int).astype(str) + "x" +
            df.loc[null_box, "box_height"].fillna(0).round(0).astype(int).astype(str)
        )
        print(f"  Filled {null_box.sum():,} null box_name values with dimension strings")

    # Carton identity and box volume (void_volume_in3 derived later in add_derived_features)
    df["carton_type"] = df["box_name"].fillna("UNKNOWN_BOX")
    df["box_volume_in3"] = df["box_length"] * df["box_width"] * df["box_height"]

    # packing_material is not captured in real data;
    # OneHotEncoder(handle_unknown='ignore') will produce all-zero encoding
    df["packing_material"] = "unknown"

    # category_avg_weight_error_oz is filled after the time split (train-only encoding)
    df["category_avg_weight_error_oz"] = np.nan

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

def encode_category_error(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target-encode category_mode with mean training error (oz).

    Computed on train only to prevent leakage into the June test set.
    Unknown categories in test fall back to the global training mean.
    """
    _hr("STAGE 5 — CATEGORY ERROR ENCODING  (train-fold only)")
    train_error = train["actual_weight_oz"] - train["theoretical_weight_oz"]
    cat_map = (
        pd.DataFrame({"cat": train["category_mode"], "err": train_error})
        .groupby("cat")["err"]
        .mean()
    )
    global_mean = float(train_error.mean())
    print(f"  Unique categories in train: {len(cat_map):,}")
    print(f"  Global mean error (train) : {global_mean:.2f} oz  ({global_mean/LBS_TO_OZ:.3f} lbs)")
    n_unseen = test["category_mode"].isin(cat_map.index).eq(False).sum()
    print(f"  Test categories unseen in train (→ global mean): {n_unseen:,} rows")

    train = train.copy()
    test = test.copy()
    train["category_avg_weight_error_oz"] = train["category_mode"].map(cat_map).fillna(global_mean)
    test["category_avg_weight_error_oz"] = test["category_mode"].map(cat_map).fillna(global_mean)
    return train, test


# ── Stage 6-7: Train + Evaluate ───────────────────────────────────────────────

def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_path: str,
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
    header = f"{'Model':<32} {'MAE oz':>8} {'MAE lbs':>8} {'RMSE oz':>8} {'Bias oz':>9} {'≤0.5oz%':>9} {'≤1.0lb%':>9}"
    print(header)
    print("-" * len(header))

    base = regression_metrics(y_test, theoretical_test_oz.values)
    print(f"  {'theoretical_baseline':<30} {base['mae_oz']:>8.2f} {base['mae_oz']/LBS_TO_OZ:>8.3f} "
          f"{base['rmse_oz']:>8.2f} {base['bias_oz']:>9.2f} {base['within_0_5oz_pct']:>8.1f}% {base['within_1lb_pct']:>8.1f}%")

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
              f"{m['rmse_oz']:>8.2f} {m['bias_oz']:>9.2f} {m['within_0_5oz_pct']:>8.1f}% {m['within_1lb_pct']:>8.1f}%{marker}")

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
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"  Saved to      : {out_path}")
    print(f"  model_version : {MODEL_VERSION}")
    print(f"  model_type    : {best_name}")
    print(f"  training rows : {len(X_train):,}")
    print(f"  residual_std  : {residuals.std():.2f} oz  ({residuals.std()/LBS_TO_OZ:.3f} lbs)")


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
    train, test = encode_category_error(train, test)
    train_and_evaluate(train, test, args.out)

    _hr("DONE")


if __name__ == "__main__":
    main()

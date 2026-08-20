"""Box-geometry features -- and the data regime change they uncovered.

WHAT THIS STARTED AS
    ``total_theoretical_shipment_weight_lbs`` decomposes exactly into
    ``theoretical_cargo_weight_lbs + theoretical_empty_box_weight_lbs``
    (100% of rows), 28.4% of rows carry a ZERO recorded tare, and per-box
    mean residual correlates with box surface area at r = 0.818. That
    suggested continuous box-shape features would beat the carton_type
    one-hot, because they generalise to boxes with little or no history.

WHAT IT ACTUALLY FOUND
    Adding tare features appears to cut MAE from 9.99 to 8.90 oz. That
    number is NOT REAL. ``box_tare_oz`` has std 6.70 in train and std
    0.00 in test; ``box_tare_is_missing`` is 0.15 on average in train and
    exactly 1.0 for every test row. The features are constant across the
    whole test fold, so they cannot be discriminating anything within it.

    They help only because they are a near-perfect PROXY FOR TIME:

        month           1     2     3     4      5      6(test)
        % backfilled  0.0   0.0   0.0  30.1   88.2    100.0
        % tare == 0   0.0   0.0   0.0   0.0   75.4    100.0

    box_name and the box tare stopped being populated partway through the
    dataset. Since the raw residual also drifts upward over time
    (Jan +12.8 oz -> Jun +19.0 oz), "tare is missing" lets the model
    identify recent rows and apply the larger recent correction. It is a
    disguised timestamp, and it would invert the moment the warehouse
    fixes the data gap -- new rows would read tare-present and be scored
    with the stale January calibration.

    The honest test is the shape-only ablation below: surface area, max
    dimension and length+girth, which ARE populated in both folds, move
    MAE by +0.02 oz. Box geometry does not help.

    The wider consequence is that the June test fold is not exchangeable
    with the training data: 100% of its carton_type values are
    dimension-string backfills versus 0% in Jan-Mar. Any feature
    correlated with the collection regime will look good on this split
    and fail in production.

Usage:
    python scripts/experiment_box_geometry.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from shipment_weight.evaluate import regression_metrics
from shipment_weight.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

from _experiment_prep import prepare
from experiment_loss_and_interactions import LBS_TO_OZ, pipeline_for
from train_real_data import _hr

SHAPE_FEATURES = ["box_surface_in2", "box_max_dim", "box_length_plus_girth"]
TARE_FEATURES = ["box_tare_oz", "box_tare_is_missing"]

HGB_KWARGS = dict(loss="absolute_error", max_iter=400, learning_rate=0.05,
                  min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.1, random_state=42)


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Box-shape columns, all derivable at prediction time from the same
    box_length/width/height the pipeline already reads."""
    df = df.copy()
    L = df["box_length"].fillna(0)
    W = df["box_width"].fillna(0)
    H = df["box_height"].fillna(0)

    df["box_surface_in2"] = 2 * (L * W + L * H + W * H)
    df["box_tare_oz"] = df["theoretical_empty_box_weight_lbs"].fillna(0) * LBS_TO_OZ
    df["box_tare_is_missing"] = (df["box_tare_oz"] <= 0).astype(int)

    # Length + girth = longest side + 2*(the other two), the dimension
    # combination carriers actually price on.
    dims = pd.concat([L, W, H], axis=1)
    df["box_max_dim"] = dims.max(axis=1)
    df["box_length_plus_girth"] = df["box_max_dim"] + 2 * (dims.sum(axis=1) - df["box_max_dim"])
    return df


def backfill_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose carton_type was reconstructed from dimensions because
    box_name was null (ingest.build_features does this fill), and rows with
    no recorded box tare."""
    df = df.copy()
    dimstr = (
        df["box_length"].fillna(0).round(0).astype(int).astype(str) + "x"
        + df["box_width"].fillna(0).round(0).astype(int).astype(str) + "x"
        + df["box_height"].fillna(0).round(0).astype(int).astype(str)
    )
    df["is_backfilled_box_name"] = (df["carton_type"].astype(str) == dimstr).astype(int)
    df["has_zero_tare"] = (df["theoretical_empty_box_weight_lbs"] == 0).astype(int)
    df["month"] = pd.to_datetime(df["order_date"]).dt.month
    return df


def residual_hgb(tr, te, extra, y_tr, theo_tr, theo_te):
    numeric = NUMERIC_FEATURES + extra
    cols = numeric + CATEGORICAL_FEATURES
    pipe = pipeline_for(HistGradientBoostingRegressor(**HGB_KWARGS),
                        numeric, CATEGORICAL_FEATURES, dense=True)
    pipe.fit(tr[cols], y_tr.values - theo_tr)
    return pipe.predict(te[cols]) + theo_te


def main() -> None:
    parser = argparse.ArgumentParser(description="Box geometry / regime-change diagnostic.")
    parser.add_argument("--shipments", default="order_shipments_anonymized.xlsx")
    parser.add_argument("--lines", default="order_lines_in_shipment_anonymized.xlsx")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    train, test = prepare(args.shipments, args.lines, refresh=args.refresh)
    train, test = add_geometry(train), add_geometry(test)
    tr_f, te_f = backfill_flags(train), backfill_flags(test)

    y_train, y_test = train[TARGET], test[TARGET]
    theo_train = train["theoretical_weight_oz"].values
    theo_test = test["theoretical_weight_oz"].values

    # ── 1. The regime change ─────────────────────────────────────────────────
    _hr("1 — DATA COLLECTION REGIME CHANGE")
    print("  Share of rows whose box_name was null (so carton_type is a")
    print("  dimension-string backfill), and whose box tare is zero, by month:\n")
    combined = pd.concat([tr_f, te_f], ignore_index=True)
    by_month = combined.groupby("month")[["is_backfilled_box_name", "has_zero_tare"]].mean() * 100
    by_month["n"] = combined.groupby("month").size()
    by_month["mean_resid_oz"] = combined.groupby("month").apply(
        lambda d: (d["actual_weight_oz"] - d["theoretical_weight_oz"]).mean(), include_groups=False
    )
    print(by_month.to_string(formatters={
        "is_backfilled_box_name": "{:.1f}%".format,
        "has_zero_tare": "{:.1f}%".format,
        "mean_resid_oz": "{:+.2f}".format,
    }))
    print("\n  Train is mostly the OLD regime; the June test fold is 100% the NEW one.")

    # ── 2. Degeneracy of the tare features ───────────────────────────────────
    _hr("2 — ARE THE TARE FEATURES DEGENERATE IN TEST?")
    print(f"  {'feature':<26} {'train mean':>11} {'train std':>11} {'test mean':>11} {'test std':>11}")
    for c in TARE_FEATURES + SHAPE_FEATURES:
        print(f"  {c:<26} {train[c].mean():>11.2f} {train[c].std():>11.2f} "
              f"{test[c].mean():>11.2f} {test[c].std():>11.2f}")
    print("\n  A feature with ZERO variance across the whole test fold cannot")
    print("  discriminate between test rows. It can only shift them all equally.")

    # ── 3. The ablation that settles it ──────────────────────────────────────
    _hr("3 — SHAPE-ONLY vs TARE-ONLY ABLATION  (HistGBT, absolute error, residual framing)")
    print("  Shape features are populated in BOTH folds. Tare features are not.\n")
    print(f"  {'variant':<32} {'MAE oz':>8} {'RMSE oz':>9} {'bias oz':>9} "
          f"{'<=0.3lb':>8} {'<=0.5lb':>8} {'<=1lb':>7} {'verdict':>12}")
    results = {}
    for label, extra in [
        ("no geometry (baseline)", []),
        ("shape only  (valid)", SHAPE_FEATURES),
        ("tare only   (degenerate)", TARE_FEATURES),
        ("shape + tare", SHAPE_FEATURES + TARE_FEATURES),
    ]:
        preds = residual_hgb(train, test, extra, y_train, theo_train, theo_test)
        m = regression_metrics(y_test, preds)
        results[label] = m
        verdict = "TRUSTWORTHY" if "degenerate" not in label and "tare" not in label else "ARTEFACT"
        print(f"  {label:<32} {m['mae_oz']:>8.2f} {m['rmse_oz']:>9.2f} {m['bias_oz']:>+9.2f} "
              f"{m['within_0_3lb_pct']:>7.1f}% {m['within_0_5lb_pct']:>7.1f}% "
              f"{m['within_1lb_pct']:>6.1f}% {verdict:>12}")

    shape_delta = results["shape only  (valid)"]["mae_oz"] - results["no geometry (baseline)"]["mae_oz"]
    tare_delta = results["tare only   (degenerate)"]["mae_oz"] - results["no geometry (baseline)"]["mae_oz"]
    print(f"\n  Shape features move MAE by {shape_delta:+.2f} oz -> box geometry does NOT help.")
    print(f"  Tare features move MAE by  {tare_delta:+.2f} oz -> but only as a time proxy.")

    # ── 4. Proving the time-proxy mechanism ──────────────────────────────────
    _hr("4 — PROOF THAT box_tare_is_missing IS ACTING AS A TIMESTAMP")
    print("  If the flag were carrying box physics, it would still help when the")
    print("  model can already see time directly. Adding an explicit day-index")
    print("  makes the flag redundant -- which is exactly what happens.\n")

    tr2, te2 = train.copy(), test.copy()
    origin = pd.Timestamp(pd.to_datetime(train["order_date"]).min())
    tr2["day_index"] = (pd.to_datetime(tr2["order_date"]) - origin).dt.days
    te2["day_index"] = (pd.to_datetime(te2["order_date"]) - origin).dt.days

    print(f"  {'variant':<40} {'MAE oz':>8} {'bias oz':>9}")
    for label, extra in [
        ("day_index only", ["day_index"]),
        ("day_index + tare flags", ["day_index"] + TARE_FEATURES),
    ]:
        preds = residual_hgb(tr2, te2, extra, y_train, theo_train, theo_test)
        m = regression_metrics(y_test, preds)
        print(f"  {label:<40} {m['mae_oz']:>8.2f} {m['bias_oz']:>+9.2f}")
    print("\n  (day_index also extrapolates outside its training range, so it is not")
    print("   shippable either -- it is here purely to demonstrate the mechanism.)")

    # ── 5. Correlation between the flag and time ─────────────────────────────
    _hr("5 — HOW STRONG IS THE FLAG/TIME CORRELATION?")
    c = combined.copy()
    c["day_index"] = (pd.to_datetime(c["order_date"]) - origin).dt.days
    for col in ["is_backfilled_box_name", "has_zero_tare"]:
        r = c[col].corr(c["day_index"])
        print(f"  Pearson r({col}, day_index) = {r:.3f}")
    print("\n  A correlation this high means the flag and the calendar are")
    print("  nearly the same variable on this dataset.")

    _hr("VERDICT")
    print("  1. Box geometry (surface area, dims) does NOT improve accuracy:")
    print(f"     {shape_delta:+.2f} oz. Drop the idea.")
    print("  2. The apparent tare-feature win is an artefact of a data-collection")
    print("     regime change and must NOT be shipped.")
    print("  3. The real, transferable finding is that the residual drifts with")
    print("     time and the test fold sits entirely in a different collection")
    print("     regime -- handle that explicitly (recency weighting / refresh")
    print("     cadence), not through a feature that happens to encode the date.")

    os.makedirs(args.outdir, exist_ok=True)
    out = pd.DataFrame([{"variant": k, **v} for k, v in results.items()])
    out.to_csv(os.path.join(args.outdir, "experiment_box_geometry.csv"), index=False)
    by_month.to_csv(os.path.join(args.outdir, "regime_change_by_month.csv"))
    print(f"\n  Written to {args.outdir}/experiment_box_geometry.csv and regime_change_by_month.csv")

    _hr("DONE")


if __name__ == "__main__":
    main()

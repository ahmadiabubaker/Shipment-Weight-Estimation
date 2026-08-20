"""Residual segmentation for the production Ridge model.

Answers three questions before any further feature work is committed to:

  A. WHERE is the error?   RMSE (19.98 oz) is ~1.8x MAE (11.32 oz), which
     means error is tail-concentrated. This quantifies the tail and
     attributes it to segments by *share of total absolute error*, not by
     segment MAE -- a segment with terrible MAE but 40 rows is not worth
     a feature.

  B. HOW MUCH of it is reducible?  Estimates an irreducible noise floor
     from repeat shipments: groups of shipments with an identical SKU/qty
     set in an identical box still disagree on actual weight, and no
     feature can explain that spread. Model MAE minus noise floor is the
     real headroom.

  C. IS PER-SKU CORRECTION (idea #1) WORTH IT?  Tests it directly and
     cheaply on single-SKU shipments: learn one mean residual per item_id
     on the train fold, apply to the June test fold, and see how much MAE
     it removes -- plus what fraction of test rows such a map can even
     cover. This is the go/no-go signal for building the full sparse
     per-SKU weight solver.

Read-only: fits the existing pipeline, writes nothing to models/.

Usage:
    python scripts/analyze_residuals.py            # uses cached split
    python scripts/analyze_residuals.py --refresh  # re-read the Excel files
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from shipment_weight.evaluate import regression_metrics
from shipment_weight.features import ALL_FEATURES, TARGET
from shipment_weight.train import make_pipeline

from _experiment_prep import prepare
from train_real_data import LBS_TO_OZ, _hr

MIN_SEGMENT_ROWS = 30  # below this a segment's MAE is noise, not signal


# ── Segment attribution ──────────────────────────────────────────────────────

def error_contribution(
    seg: pd.Series, abs_err: np.ndarray, signed_err: np.ndarray, name: str, top_n: int = 15
) -> pd.DataFrame:
    """Per-segment error table sorted by SHARE OF TOTAL ABSOLUTE ERROR.

    ``err_share_pct`` is the decision column: it is the fraction of the
    model's entire test-set absolute error that this segment accounts
    for. ``lift`` = err_share_pct / row_share_pct, so >1 means the
    segment carries more error than its size warrants.
    """
    total_abs = abs_err.sum()
    n_total = len(abs_err)
    frame = pd.DataFrame({"seg": np.asarray(seg), "abs_err": abs_err, "err": signed_err})
    g = frame.groupby("seg", observed=True).agg(
        n=("abs_err", "size"),
        mae_oz=("abs_err", "mean"),
        bias_oz=("err", "mean"),
        std_oz=("err", "std"),
        sum_abs_err=("abs_err", "sum"),
    )
    g["row_share_pct"] = g["n"] / n_total * 100
    g["err_share_pct"] = g["sum_abs_err"] / total_abs * 100
    g["lift"] = g["err_share_pct"] / g["row_share_pct"].replace(0, np.nan)
    g = g.reset_index().rename(columns={"seg": name})
    g = g.sort_values("err_share_pct", ascending=False)
    cols = [name, "n", "row_share_pct", "mae_oz", "bias_oz", "std_oz", "err_share_pct", "lift"]
    return g[cols].head(top_n)


def print_table(df: pd.DataFrame) -> None:
    fmt = {
        "row_share_pct": "{:.1f}".format,
        "mae_oz": "{:.2f}".format,
        "bias_oz": "{:+.2f}".format,
        "std_oz": "{:.2f}".format,
        "err_share_pct": "{:.1f}".format,
        "lift": "{:.2f}".format,
    }
    print(df.to_string(index=False, formatters={k: v for k, v in fmt.items() if k in df.columns}))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Segment the Ridge model's test residuals.")
    parser.add_argument("--shipments", default="order_shipments_anonymized.xlsx")
    parser.add_argument("--lines", default="order_lines_in_shipment_anonymized.xlsx")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    train, test, lines = prepare(args.shipments, args.lines, refresh=args.refresh, with_lines=True)

    X_train, y_train = train[ALL_FEATURES], train[TARGET]
    X_test, y_test = test[ALL_FEATURES], test[TARGET]

    pipe = make_pipeline(Ridge(alpha=1.0))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    m = regression_metrics(y_test, preds)
    base_m = regression_metrics(y_test, test["theoretical_weight_oz"].values)

    signed_err = preds - y_test.values           # +ve = model over-predicts
    abs_err = np.abs(signed_err)
    total_abs = abs_err.sum()

    _hr("REFERENCE — production Ridge on June test fold")
    print(f"  rows           : {len(test):,}")
    print(f"  MAE            : {m['mae_oz']:.2f} oz  ({m['mae_oz']/LBS_TO_OZ:.3f} lbs)")
    print(f"  RMSE           : {m['rmse_oz']:.2f} oz  ({m['rmse_oz']/LBS_TO_OZ:.3f} lbs)")
    print(f"  Bias           : {m['bias_oz']:+.2f} oz")
    print(f"  RMSE/MAE ratio : {m['rmse_oz']/m['mae_oz']:.2f}   (1.25 = Gaussian; higher = heavy tail)")
    print(f"  within 0.3 lb  : {m['within_0_3lb_pct']:.1f}%")
    print(f"  within 0.5 lb  : {m['within_0_5lb_pct']:.1f}%")
    print(f"  within 1 lb    : {m['within_1lb_pct']:.1f}%")
    print(f"  theoretical baseline MAE: {base_m['mae_oz']:.2f} oz")

    # ── A. Error concentration ───────────────────────────────────────────────
    _hr("A1 — ERROR CONCENTRATION  (how tail-heavy is the error?)")
    order = np.argsort(-abs_err)
    cum = np.cumsum(abs_err[order]) / total_abs * 100
    print(f"  {'worst X% of rows':<20} {'rows':>7} {'% of total abs error':>22} {'MAE in slice':>14}")
    for pct in (1, 2, 5, 10, 20, 25, 50):
        k = max(1, int(len(abs_err) * pct / 100))
        print(f"  {'top ' + str(pct) + '%':<20} {k:>7,} {cum[k-1]:>21.1f}% {abs_err[order][:k].mean():>13.1f} oz")

    err_1lb = abs_err[abs_err > 16.0].sum() / total_abs * 100
    err_2lb = abs_err[abs_err > 32.0].sum() / total_abs * 100
    print(f"\n  Rows with abs error > 1 lb : {(abs_err > 16).sum():,} ({(abs_err > 16).mean()*100:.1f}% of rows) "
          f"-> {err_1lb:.1f}% of total error")
    print(f"  Rows with abs error > 2 lb : {(abs_err > 32).sum():,} ({(abs_err > 32).mean()*100:.1f}% of rows) "
          f"-> {err_2lb:.1f}% of total error")

    # ── A2. Segment attribution ──────────────────────────────────────────────
    _hr("A2 — ERROR BY SEGMENT  (sorted by share of total absolute error)")

    seg_defs: list[tuple[str, pd.Series]] = []

    seg_defs.append(("carton_type", test["carton_type"].astype(str).reset_index(drop=True)))
    seg_defs.append(("ship_method", test["ship_method"].astype(str).reset_index(drop=True)))

    actual_lbs = (y_test.values / LBS_TO_OZ)
    weight_band = pd.cut(
        actual_lbs,
        bins=[-np.inf, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["<1lb", "1-2lb", "2-5lb", "5-10lb", "10-20lb", "20-50lb", "50lb+"],
    )
    seg_defs.append(("actual_weight_band", pd.Series(weight_band)))

    item_bucket = pd.cut(
        test["item_count"].reset_index(drop=True),
        bins=[0, 1, 2, 5, 9, 1e9],
        labels=["1", "2", "3-5", "6-9", "10+"],
    )
    seg_defs.append(("item_count_bucket", pd.Series(item_bucket)))

    sku_bucket = pd.cut(
        test["distinct_sku_count"].reset_index(drop=True),
        bins=[0, 1, 2, 5, 1e9],
        labels=["1 sku", "2 skus", "3-5 skus", "6+ skus"],
    )
    seg_defs.append(("distinct_sku_bucket", pd.Series(sku_bucket)))

    overfilled = (test["total_item_volume_in3"] > test["box_volume_in3"]).map(
        {True: "overfilled", False: "fits"}
    ).reset_index(drop=True)
    seg_defs.append(("is_overfilled", overfilled))

    seg_defs.append(("category_mode", test["category_mode"].astype(str).reset_index(drop=True)))
    seg_defs.append(("destination_zone", test["destination_zone"].astype(str).reset_index(drop=True)))

    saved: dict[str, pd.DataFrame] = {}
    for name, seg in seg_defs:
        print(f"\n  — {name} —")
        tbl = error_contribution(seg, abs_err, signed_err, name)
        print_table(tbl)
        saved[name] = tbl

    # ── A3. Where the top-decile errors live ─────────────────────────────────
    _hr("A3 — PROFILE OF THE WORST 10% OF ROWS")
    k = int(len(abs_err) * 0.10)
    worst_idx = order[:k]
    rest_idx = order[k:]
    worst = test.iloc[worst_idx]
    rest = test.iloc[rest_idx]
    print(f"  Worst {k:,} rows carry {cum[k-1]:.1f}% of total error (MAE {abs_err[worst_idx].mean():.1f} oz)\n")
    print(f"  {'attribute':<34} {'worst 10%':>14} {'other 90%':>14}")
    comparisons = [
        ("mean actual_weight_lbs", worst["actual_weight_lbs"].mean(), rest["actual_weight_lbs"].mean()),
        ("mean theoretical_weight_lbs", worst["theoretical_weight_oz"].mean() / LBS_TO_OZ,
         rest["theoretical_weight_oz"].mean() / LBS_TO_OZ),
        ("mean item_count", worst["item_count"].mean(), rest["item_count"].mean()),
        ("mean distinct_sku_count", worst["distinct_sku_count"].mean(), rest["distinct_sku_count"].mean()),
        ("mean num_categories", worst["num_categories"].mean(), rest["num_categories"].mean()),
        ("mean fill_ratio", worst["fill_ratio"].mean(), rest["fill_ratio"].mean()),
        ("pct overfilled", (worst["total_item_volume_in3"] > worst["box_volume_in3"]).mean() * 100,
         (rest["total_item_volume_in3"] > rest["box_volume_in3"]).mean() * 100),
        ("pct zero total_item_volume", (worst["total_item_volume_in3"] == 0).mean() * 100,
         (rest["total_item_volume_in3"] == 0).mean() * 100),
        ("mean actual/theoretical ratio",
         (worst["actual_weight_oz"] / worst["theoretical_weight_oz"].replace(0, np.nan)).median(),
         (rest["actual_weight_oz"] / rest["theoretical_weight_oz"].replace(0, np.nan)).median()),
    ]
    for label, a, b in comparisons:
        print(f"  {label:<34} {a:>14.2f} {b:>14.2f}")

    print(f"\n  Direction of the worst errors:")
    w_signed = signed_err[worst_idx]
    print(f"    model UNDER-predicts (actual heavier): {(w_signed < 0).mean()*100:.1f}% of worst rows")
    print(f"    model OVER-predicts  (actual lighter): {(w_signed > 0).mean()*100:.1f}% of worst rows")

    # ── B. Irreducible noise floor ───────────────────────────────────────────
    _hr("B — IRREDUCIBLE NOISE FLOOR  (repeat-shipment disagreement)")
    print("  Shipments with an IDENTICAL sku/qty set in an IDENTICAL box should weigh")
    print("  the same. Whatever spread remains is label/process noise that no feature")
    print("  can explain, and it bounds how far MAE can possibly fall.\n")

    ln = lines.drop_duplicates(subset=["shipment_number", "item_id"], keep="first").copy()
    ln["item_quantity"] = ln["item_quantity"].fillna(1)
    ln = ln.sort_values(["shipment_number", "item_id"])
    ln["pair"] = ln["item_id"].astype(str) + "x" + ln["item_quantity"].astype(int).astype(str)
    sig = ln.groupby("shipment_number")["pair"].agg("|".join).rename("sku_signature").reset_index()

    all_rows = pd.concat([train, test], ignore_index=True)
    all_rows = all_rows.merge(sig, on="shipment_number", how="left")
    all_rows["group_key"] = all_rows["sku_signature"].astype(str) + "||" + all_rows["carton_type"].astype(str)

    grp = all_rows.groupby("group_key")["actual_weight_oz"].agg(["size", "mean", "std"])
    repeats = grp[grp["size"] >= 3].dropna(subset=["std"])
    n_rows_in_repeats = int(repeats["size"].sum())
    print(f"  Repeat groups (same sku-set + same box, n>=3): {len(repeats):,} groups "
          f"covering {n_rows_in_repeats:,} shipments ({n_rows_in_repeats/len(all_rows)*100:.1f}% of data)")

    if len(repeats) > 0:
        # Within-group MAD around the group mean = the noise floor an oracle
        # model (perfect knowledge of contents + box) would still incur.
        merged = all_rows.merge(
            repeats[["mean"]].rename(columns={"mean": "group_mean_oz"}),
            left_on="group_key", right_index=True, how="inner",
        )
        within_dev = (merged["actual_weight_oz"] - merged["group_mean_oz"]).abs()
        # Small-sample correction: with n per group, deviation from the sample
        # mean understates true spread by sqrt((n-1)/n).
        sizes = merged["group_key"].map(repeats["size"])
        corrected = within_dev * np.sqrt(sizes / (sizes - 1).clip(lower=1))
        print(f"  Oracle MAE within repeat groups (noise floor): {corrected.mean():.2f} oz "
              f"({corrected.mean()/LBS_TO_OZ:.3f} lbs)")
        print(f"  Median within-group std                      : {repeats['std'].median():.2f} oz")
        print(f"\n  Current model MAE : {m['mae_oz']:.2f} oz")
        print(f"  Noise floor       : {corrected.mean():.2f} oz")
        headroom = m["mae_oz"] - corrected.mean()
        print(f"  => HEADROOM       : {headroom:.2f} oz ({headroom/LBS_TO_OZ:.3f} lbs), "
              f"i.e. at most {headroom/m['mae_oz']*100:.0f}% further MAE reduction is achievable")

        # How does the model actually do on those same repeat rows?
        test_with_key = test.merge(sig, on="shipment_number", how="left")
        test_with_key["group_key"] = (
            test_with_key["sku_signature"].astype(str) + "||" + test_with_key["carton_type"].astype(str)
        )
        in_repeat = test_with_key["group_key"].isin(repeats.index).values
        if in_repeat.sum() > MIN_SEGMENT_ROWS:
            print(f"\n  On the {in_repeat.sum():,} TEST rows that fall in a repeat group:")
            print(f"    model MAE  : {abs_err[in_repeat].mean():.2f} oz")
            print(f"    model MAE on the other {(~in_repeat).sum():,} rows: {abs_err[~in_repeat].mean():.2f} oz")

    # ── C. Is per-SKU correction worth building? ─────────────────────────────
    _hr("C — PER-SKU CORRECTION PROBE  (go/no-go for idea #1)")
    print("  Learns ONE mean per-unit residual per item_id from single-SKU TRAIN")
    print("  shipments, then applies it to the June test fold. If catalog weights")
    print("  are systematically wrong per item, this alone should cut MAE sharply.\n")

    single = ln.groupby("shipment_number").filter(lambda g: len(g) == 1)
    single = single[["shipment_number", "item_id", "item_quantity"]]

    tr_s = train.merge(single, on="shipment_number", how="inner")
    te_s = test.merge(single, on="shipment_number", how="inner")
    print(f"  Single-SKU shipments: train={len(tr_s):,}  test={len(te_s):,} "
          f"({len(te_s)/len(test)*100:.1f}% of test rows)")

    tr_s = tr_s.assign(
        resid_oz=tr_s["actual_weight_oz"] - tr_s["theoretical_weight_oz"],
    )
    tr_s["resid_per_unit_oz"] = tr_s["resid_oz"] / tr_s["item_quantity"].clip(lower=1)

    for min_n in (3, 5, 10):
        stats = tr_s.groupby("item_id")["resid_per_unit_oz"].agg(["size", "mean"])
        keep = stats[stats["size"] >= min_n]
        corr = te_s["item_id"].map(keep["mean"])
        covered = corr.notna()
        if covered.sum() < MIN_SEGMENT_ROWS:
            print(f"  min_n={min_n}: too few covered test rows ({covered.sum()}) to judge")
            continue

        sub = te_s[covered]
        adj = sub["theoretical_weight_oz"] + corr[covered] * sub["item_quantity"].clip(lower=1)
        mae_theo = (sub["actual_weight_oz"] - sub["theoretical_weight_oz"]).abs().mean()
        mae_sku = (sub["actual_weight_oz"] - adj).abs().mean()

        # Ridge's MAE on the very same rows, for a like-for-like comparison.
        ridge_sub = test.set_index("shipment_number").loc[sub["shipment_number"]]
        ridge_pred_sub = pipe.predict(ridge_sub[ALL_FEATURES])
        mae_ridge = np.abs(ridge_pred_sub - ridge_sub[TARGET].values).mean()

        print(f"\n  min_n={min_n:<3} items with a learned correction: {len(keep):,}")
        print(f"    test rows covered            : {covered.sum():,} "
              f"({covered.sum()/len(test)*100:.1f}% of ALL test rows)")
        print(f"    theoretical-only MAE (these rows) : {mae_theo:>7.2f} oz")
        print(f"    production Ridge MAE (these rows) : {mae_ridge:>7.2f} oz")
        print(f"    theoretical + per-SKU corr MAE    : {mae_sku:>7.2f} oz   "
              f"({mae_sku - mae_ridge:+.2f} oz vs Ridge)")

    # ── C2. Stability of a per-SKU correction ────────────────────────────────
    print("\n  Per-item residual stability (are item corrections a real constant,")
    print("  or just noise?) — computed on single-SKU train shipments with n>=5:")
    stats5 = tr_s.groupby("item_id")["resid_per_unit_oz"].agg(["size", "mean", "std"])
    stats5 = stats5[stats5["size"] >= 5].dropna()
    if len(stats5) > 0:
        between_var = stats5["mean"].var()
        within_var = (stats5["std"] ** 2).mean()
        icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else float("nan")
        print(f"    items examined                    : {len(stats5):,}")
        print(f"    between-item variance of residual : {between_var:>10.1f} oz^2")
        print(f"    mean within-item variance         : {within_var:>10.1f} oz^2")
        print(f"    => intraclass correlation (ICC)   : {icc:.3f}")
        print(f"       ICC near 1 = item identity explains the residual (build #1);")
        print(f"       ICC near 0 = residual is per-shipment noise (skip #1).")
        print(f"    median |mean per-unit residual|   : {stats5['mean'].abs().median():.2f} oz")
        print(f"    median within-item std            : {stats5['std'].median():.2f} oz")

    # ── D. Drift ─────────────────────────────────────────────────────────────
    _hr("D — MONTHLY DRIFT  (does the residual move over time?)")
    all_rows["month"] = pd.to_datetime(all_rows["order_date"]).dt.to_period("M").astype(str)
    all_rows["resid_oz"] = all_rows["actual_weight_oz"] - all_rows["theoretical_weight_oz"]
    drift = all_rows.groupby("month")["resid_oz"].agg(
        n="size", mean_resid_oz="mean", median_resid_oz="median", std_oz="std"
    ).reset_index()
    print("  Raw (actual - theoretical) residual by month, whole dataset:")
    print(drift.to_string(index=False, formatters={
        "mean_resid_oz": "{:+.2f}".format,
        "median_resid_oz": "{:+.2f}".format,
        "std_oz": "{:.2f}".format,
    }))

    test_month_err = pd.DataFrame({
        "week": pd.to_datetime(test["order_date"]).dt.isocalendar().week.values,
        "abs_err": abs_err,
        "err": signed_err,
    }).groupby("week").agg(n=("abs_err", "size"), mae_oz=("abs_err", "mean"), bias_oz=("err", "mean"))
    print("\n  Model error by week within the June test fold (drift within horizon):")
    print(test_month_err.to_string(formatters={"mae_oz": "{:.2f}".format, "bias_oz": "{:+.2f}".format}))

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    for name, tbl in saved.items():
        tbl.to_csv(os.path.join(args.outdir, f"residuals_by_{name}.csv"), index=False)
    print(f"\n  Segment tables written to {args.outdir}/residuals_by_*.csv")

    _hr("DONE")


if __name__ == "__main__":
    main()

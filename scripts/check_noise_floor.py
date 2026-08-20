"""Sanity-check the repeat-shipment noise floor from analyze_residuals.py.

The headroom claim ("MAE can fall at most ~4.8 oz further") rests on a
noise floor estimated from the 3.3% of shipments that belong to a repeat
group -- same sku/qty set, same box, seen 3+ times. That subset is not a
random sample, so this script checks whether it is representative before
the number is trusted:

  1. How do repeat-group shipments differ from the rest (weight, item
     count, box mix)?
  2. Does the floor hold up when re-estimated within weight bands, so a
     weight-mix difference cannot explain it?
  3. What does the floor look like as a FRACTION of shipment weight,
     which is the scale-free version of the same question?

Usage:
    python scripts/check_noise_floor.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from _experiment_prep import prepare
from train_real_data import LBS_TO_OZ, _hr

MIN_GROUP_SIZE = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the repeat-shipment noise floor.")
    parser.add_argument("--shipments", default="order_shipments_anonymized.xlsx")
    parser.add_argument("--lines", default="order_lines_in_shipment_anonymized.xlsx")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    train, test, lines = prepare(args.shipments, args.lines, refresh=args.refresh, with_lines=True)
    all_rows = pd.concat([train, test], ignore_index=True)

    ln = lines.drop_duplicates(subset=["shipment_number", "item_id"], keep="first").copy()
    ln["item_quantity"] = ln["item_quantity"].fillna(1)
    ln = ln.sort_values(["shipment_number", "item_id"])
    ln["pair"] = ln["item_id"].astype(str) + "x" + ln["item_quantity"].astype(int).astype(str)
    sig = ln.groupby("shipment_number")["pair"].agg("|".join).rename("sku_signature").reset_index()

    all_rows = all_rows.merge(sig, on="shipment_number", how="left")
    all_rows["group_key"] = all_rows["sku_signature"].astype(str) + "||" + all_rows["carton_type"].astype(str)

    sizes = all_rows.groupby("group_key")["actual_weight_oz"].transform("size")
    all_rows["in_repeat"] = sizes >= MIN_GROUP_SIZE

    rep = all_rows[all_rows["in_repeat"]]
    oth = all_rows[~all_rows["in_repeat"]]

    _hr("1 — ARE REPEAT-GROUP SHIPMENTS REPRESENTATIVE?")
    print(f"  repeat-group rows: {len(rep):,} ({len(rep)/len(all_rows)*100:.1f}%)   "
          f"other rows: {len(oth):,}\n")
    print(f"  {'attribute':<32} {'repeat groups':>15} {'everything else':>17}")
    for label, col in [
        ("mean actual_weight_lbs", "actual_weight_lbs"),
        ("median actual_weight_lbs", "actual_weight_lbs"),
        ("mean item_count", "item_count"),
        ("mean distinct_sku_count", "distinct_sku_count"),
        ("mean num_categories", "num_categories"),
    ]:
        fn = np.median if label.startswith("median") else np.mean
        print(f"  {label:<32} {fn(rep[col]):>15.2f} {fn(oth[col]):>17.2f}")

    print(f"\n  Top boxes among repeat groups vs overall (share of rows):")
    rep_mix = rep["carton_type"].value_counts(normalize=True).head(8) * 100
    all_mix = all_rows["carton_type"].value_counts(normalize=True) * 100
    print(f"  {'box':<16} {'repeat %':>10} {'overall %':>11}")
    for box, pct in rep_mix.items():
        print(f"  {str(box):<16} {pct:>10.1f} {all_mix.get(box, 0.0):>11.1f}")

    _hr("2 — NOISE FLOOR RE-ESTIMATED WITHIN WEIGHT BANDS")
    print("  If the headline floor were an artefact of repeat groups being lighter or")
    print("  heavier than average, the per-band floors would disagree with it.\n")

    grp = all_rows.groupby("group_key")["actual_weight_oz"].agg(["size", "mean"])
    grp = grp[grp["size"] >= MIN_GROUP_SIZE]
    merged = all_rows.merge(
        grp.rename(columns={"mean": "group_mean_oz", "size": "group_size"}),
        left_on="group_key", right_index=True, how="inner",
    )
    # Deviation from the group mean understates spread by sqrt((n-1)/n).
    merged["dev_oz"] = (merged["actual_weight_oz"] - merged["group_mean_oz"]).abs() * np.sqrt(
        merged["group_size"] / (merged["group_size"] - 1).clip(lower=1)
    )

    merged["band"] = pd.cut(
        merged["actual_weight_lbs"],
        bins=[-np.inf, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["<1lb", "1-2lb", "2-5lb", "5-10lb", "10-20lb", "20-50lb", "50lb+"],
    )
    band = merged.groupby("band", observed=True).agg(
        n=("dev_oz", "size"),
        floor_oz=("dev_oz", "mean"),
        mean_weight_lbs=("actual_weight_lbs", "mean"),
    ).reset_index()
    band["floor_pct_of_weight"] = band["floor_oz"] / (band["mean_weight_lbs"] * LBS_TO_OZ) * 100

    print(band.to_string(index=False, formatters={
        "floor_oz": "{:.2f}".format,
        "mean_weight_lbs": "{:.2f}".format,
        "floor_pct_of_weight": "{:.2f}".format,
    }))

    overall_floor = merged["dev_oz"].mean()
    print(f"\n  Pooled floor (as reported by analyze_residuals): {overall_floor:.2f} oz")

    # Reweight the per-band floors by the FULL dataset's weight distribution,
    # which removes any weight-mix difference between repeat groups and the rest.
    full_band = pd.cut(
        all_rows["actual_weight_lbs"],
        bins=[-np.inf, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["<1lb", "1-2lb", "2-5lb", "5-10lb", "10-20lb", "20-50lb", "50lb+"],
    )
    full_share = full_band.value_counts(normalize=True)
    band_lookup = band.set_index("band")["floor_oz"]
    common = [b for b in band_lookup.index if b in full_share.index]
    reweighted = sum(band_lookup[b] * full_share[b] for b in common) / sum(full_share[b] for b in common)
    print(f"  Weight-mix-adjusted floor (per-band floors reweighted to the full")
    print(f"  dataset's weight distribution)                 : {reweighted:.2f} oz")

    _hr("3 — VERDICT")
    print(f"  Reported floor        : {overall_floor:.2f} oz")
    print(f"  Mix-adjusted floor    : {reweighted:.2f} oz")
    drift = abs(reweighted - overall_floor)
    if drift < 1.0:
        print(f"  Difference is {drift:.2f} oz -- the weight-mix skew of repeat groups does")
        print(f"  NOT materially bias the floor, so the headroom estimate stands.")
    else:
        print(f"  Difference is {drift:.2f} oz -- repeat groups ARE skewed enough to matter;")
        print(f"  prefer the mix-adjusted figure when quoting headroom.")

    _hr("DONE")


if __name__ == "__main__":
    main()

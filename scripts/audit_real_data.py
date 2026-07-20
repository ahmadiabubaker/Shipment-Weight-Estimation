"""Data audit for real warehouse shipment data.

Checks and documents (with row counts) every data quality issue before
retraining.  Output is plain stdout — pipe to a file to keep a record.

Checks performed:
  1. Null rates per column (both files)
  2. Weight consistency: theoretical_cargo + theoretical_empty_box == total
  3. Duplicate shipment_numbers in shipments file
  4. Duplicate (shipment_number, item_id) pairs in order lines
  5. Shipments missing from / orphaned in order lines
  6. actual_weight_lbs distribution + non-positive values
  7. Extreme outlier investigation (top 20 by actual_weight_lbs)
  8. ship_method value counts + flagged exclusions
  9. box_name null analysis and dimension fallback availability
 10. category distribution in order lines

Usage:
    python scripts/audit_real_data.py \\
        --shipments path/to/order_shipments_anonymized.xlsx \\
        --lines     path/to/order_lines_in_shipment_anonymized.xlsx
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

WEIGHT_TOLERANCE_LBS = 0.01

# Exact strings known to be bad + keyword patterns for variants
BAD_METHOD_EXACT = {"Pick Up At Medusa", "CanceledItem", "Ship Outside System"}
BAD_METHOD_KEYWORDS = ["pick up", "canceled", "outside system"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _hr(title: str = "") -> None:
    line = "=" * 62
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(line)


def _load(shipments_path: str, lines_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _hr("1. LOADING FILES")
    print(f"  shipments : {shipments_path}")
    ships = pd.read_excel(shipments_path, engine="openpyxl")
    print(f"    {len(ships):>10,} rows  |  {ships.shape[1]} columns")

    print(f"  order lines: {lines_path}")
    lines = pd.read_excel(lines_path, engine="openpyxl")
    print(f"    {len(lines):>10,} rows  |  {lines.shape[1]} columns")
    return ships, lines


# ── individual checks ─────────────────────────────────────────────────────────

def check_nulls(df: pd.DataFrame, name: str) -> None:
    _hr(f"NULL RATES — {name}")
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)
    report = pd.DataFrame({"null_count": null_counts, "null_%": null_pcts})
    report = report[report["null_count"] > 0].sort_values("null_%", ascending=False)
    if report.empty:
        print("  No nulls found.")
    else:
        print(report.to_string())


def check_weight_consistency(ships: pd.DataFrame) -> None:
    _hr("WEIGHT CONSISTENCY  (cargo + empty_box == total)")
    expected = (
        ships["theoretical_cargo_weight_lbs"].fillna(0)
        + ships["theoretical_empty_box_weight_lbs"].fillna(0)
    )
    diff = (ships["total_theoretical_shipment_weight_lbs"] - expected).abs()
    inconsistent = diff > WEIGHT_TOLERANCE_LBS
    n = int(inconsistent.sum())
    print(f"  tolerance : {WEIGHT_TOLERANCE_LBS} lbs")
    print(f"  mismatch  : {n:,} rows  ({n / len(ships) * 100:.2f}%)")
    if n > 0:
        print(f"  max diff  : {diff.max():.4f} lbs")
        print(f"  median diff (bad rows): {diff[inconsistent].median():.4f} lbs")
        print(f"\n  Sample mismatched rows (up to 5):")
        sample = ships.loc[inconsistent, [
            "shipment_number",
            "theoretical_cargo_weight_lbs",
            "theoretical_empty_box_weight_lbs",
            "total_theoretical_shipment_weight_lbs",
        ]].head(5)
        print(sample.to_string(index=False))


def check_duplicates(ships: pd.DataFrame, lines: pd.DataFrame) -> None:
    _hr("DUPLICATE CHECKS")
    dup_ships = int(ships["shipment_number"].duplicated().sum())
    print(f"  Duplicate shipment_number in shipments file : {dup_ships:,}")

    dup_lines = int(lines.duplicated(subset=["shipment_number", "item_id"]).sum())
    print(f"  Duplicate (shipment_number, item_id) in lines: {dup_lines:,}")

    ship_ids = set(ships["shipment_number"].dropna())
    line_ids = set(lines["shipment_number"].dropna())
    print(f"  Shipments with no matching order lines : {len(ship_ids - line_ids):,}")
    print(f"  Order-line shipment_numbers not in shipments: {len(line_ids - ship_ids):,}")


def check_actual_weight(ships: pd.DataFrame) -> None:
    _hr("ACTUAL WEIGHT CHECKS")
    null_n = int(ships["actual_weight_lbs"].isnull().sum())
    print(f"  null actual_weight_lbs   : {null_n:,}")
    nonpos_n = int((ships["actual_weight_lbs"] <= 0).sum())
    print(f"  actual_weight_lbs <= 0   : {nonpos_n:,}")

    s = ships["actual_weight_lbs"].dropna()
    print(f"\n  Distribution (non-null, n={len(s):,}):")
    pcts = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
    for p in pcts:
        print(f"    p{int(p*100):>3}  = {s.quantile(p):>12.2f} lbs")
    print(f"    max   = {s.max():>12.2f} lbs")
    print(f"    mean  = {s.mean():>12.2f} lbs")
    print(f"    std   = {s.std():>12.2f} lbs")


def check_outliers(ships: pd.DataFrame) -> None:
    _hr("OUTLIER INVESTIGATION — top 20 by actual_weight_lbs")
    cols = [
        "shipment_number", "order_date", "ship_method", "box_name",
        "actual_weight_lbs", "total_theoretical_shipment_weight_lbs",
    ]
    available = [c for c in cols if c in ships.columns]
    top = ships.nlargest(20, "actual_weight_lbs")[available]
    print(top.to_string(index=False))

    # Flag rows where actual >> theoretical (ratio > 3x)
    ratio = ships["actual_weight_lbs"] / ships["total_theoretical_shipment_weight_lbs"].replace(0, np.nan)
    extreme = (ratio > 3).sum()
    print(f"\n  Rows where actual > 3× theoretical : {extreme:,}")
    print(f"  Rows where actual > 5× theoretical : {int((ratio > 5).sum()):,}")
    print(f"  Rows where actual > 10× theoretical: {int((ratio > 10).sum()):,}")


def check_ship_methods(ships: pd.DataFrame) -> None:
    _hr("SHIP METHOD VALUE COUNTS")
    vc = ships["ship_method"].value_counts(dropna=False)
    print(vc.to_string())

    exact_mask = ships["ship_method"].isin(BAD_METHOD_EXACT)
    kw_mask = ships["ship_method"].str.contains(
        "|".join(BAD_METHOD_KEYWORDS), case=False, na=False
    )
    all_bad = exact_mask | kw_mask
    print(f"\n  Rows flagged for exclusion (known bad ship methods): {int(all_bad.sum()):,}")
    if all_bad.sum() > 0:
        print("  Breakdown:")
        print(ships.loc[all_bad, "ship_method"].value_counts().to_string())


def check_box_name(ships: pd.DataFrame) -> None:
    _hr("BOX NAME ANALYSIS")
    null_n = int(ships["box_name"].isnull().sum())
    pct = null_n / len(ships) * 100
    print(f"  Null box_name     : {null_n:,}  ({pct:.2f}%)")
    print(f"  Unique box names  : {ships['box_name'].nunique():,}")

    # Among null-box rows, how many have usable dimensions?
    dim_cols = ["box_length", "box_width", "box_height"]
    null_box = ships[ships["box_name"].isnull()]
    has_dims = (
        null_box[dim_cols].notna().all(axis=1)
        & (null_box[dim_cols] > 0).all(axis=1)
    ).sum()
    print(f"  Of {null_n:,} null-box rows, {has_dims:,} have valid box dimensions (usable fallback)")

    print(f"\n  Top 25 box names by count:")
    print(ships["box_name"].value_counts().head(25).to_string())


def check_categories(lines: pd.DataFrame) -> None:
    _hr("CATEGORY DISTRIBUTION — order lines")
    print(f"  Unique categories : {lines['category'].nunique():,}")
    null_n = int(lines["category"].isnull().sum())
    print(f"  Null category     : {null_n:,}  ({null_n / len(lines) * 100:.2f}%)")

    print(f"\n  Top 30 categories by row count:")
    print(lines["category"].value_counts().head(30).to_string())

    print(f"\n  Item volume completeness (width × length × height):")
    dim_cols = ["item_width_inches", "item_length_inches", "item_height_inches"]
    for col in dim_cols:
        n = int(lines[col].isnull().sum())
        print(f"    {col:<30}: {n:,} null  ({n / len(lines) * 100:.2f}%)")

    print(f"\n  theoretical_item_weight_lbs null: "
          f"{lines['theoretical_item_weight_lbs'].isnull().sum():,}"
          f"  ({lines['theoretical_item_weight_lbs'].isnull().mean()*100:.2f}%)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit real warehouse shipment + order-line data before retraining."
    )
    parser.add_argument("--shipments", required=True,
                        help="Path to order_shipments_anonymized.xlsx")
    parser.add_argument("--lines", required=True,
                        help="Path to order_lines_in_shipment_anonymized.xlsx")
    args = parser.parse_args()

    ships, lines = _load(args.shipments, args.lines)

    _hr("COLUMN NAMES")
    print(f"  shipments  : {list(ships.columns)}")
    print(f"  order lines: {list(lines.columns)}")

    check_nulls(ships, "order_shipments")
    check_nulls(lines, "order_lines")
    check_weight_consistency(ships)
    check_duplicates(ships, lines)
    check_actual_weight(ships)
    check_outliers(ships)
    check_ship_methods(ships)
    check_box_name(ships)
    check_categories(lines)

    _hr("AUDIT COMPLETE")


if __name__ == "__main__":
    main()

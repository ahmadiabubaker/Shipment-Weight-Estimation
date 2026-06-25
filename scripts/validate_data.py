"""Validate a real shipment CSV against the schema the model pipeline expects.

Run this against Medusa's historical data as soon as it arrives, before
trusting it for retraining:

    python scripts/validate_data.py path/to/shipments.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shipment_weight.data_gen import CARTON_TYPES  # noqa: E402

EXPECTED_COLUMNS = [
    "theoretical_weight_oz",
    "actual_weight_oz",
    "carton_type",
    "item_count",
    "total_item_volume_in3",
    "item_categories",
    "category_mode",
    "ship_method",
    "packing_material",
    "num_missing_catalog_weights",
    "category_avg_weight_error_oz",
]
KNOWN_CARTON_TYPES = {name for name, _, _ in CARTON_TYPES}


class Check:
    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail


def check_columns_present(df: pd.DataFrame) -> Check:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return Check("expected columns present", False, f"missing: {missing}")
    return Check("expected columns present", True)


def check_row_count(df: pd.DataFrame) -> Check:
    n = len(df)
    if n == 0:
        return Check("row count", False, "0 rows")
    return Check("row count", True, f"{n} rows")


def check_null_rates(df: pd.DataFrame) -> Check:
    present_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    rates = df[present_cols].isna().mean() * 100
    lines = [f"{col}: {rate:.1f}%" for col, rate in rates.items() if rate > 0]
    detail = "; ".join(lines) if lines else "no nulls in expected columns"
    return Check("null rates per column", True, detail)


def check_weight_ranges(df: pd.DataFrame) -> Check:
    problems = []
    for col in ("theoretical_weight_oz", "actual_weight_oz"):
        if col not in df.columns:
            continue
        non_positive = (df[col].dropna() <= 0).sum()
        if non_positive:
            problems.append(f"{col} has {non_positive} non-positive value(s)")
    if problems:
        return Check("weight ranges positive", False, "; ".join(problems))
    return Check("weight ranges positive", True)


def check_carton_types(df: pd.DataFrame) -> Check:
    if "carton_type" not in df.columns:
        return Check("carton_type matches known types", False, "column missing")
    unknown = set(df["carton_type"].dropna().unique()) - KNOWN_CARTON_TYPES
    if unknown:
        return Check(
            "carton_type matches known types",
            False,
            f"unrecognized carton types: {sorted(unknown)} (known: {sorted(KNOWN_CARTON_TYPES)})",
        )
    return Check("carton_type matches known types", True)


def check_item_count(df: pd.DataFrame) -> Check:
    if "item_count" not in df.columns:
        return Check("item_count is a positive integer", False, "column missing")
    col = df["item_count"].dropna()
    non_integer = (col != col.astype(int)).sum()
    non_positive = (col <= 0).sum()
    if non_integer or non_positive:
        return Check(
            "item_count is a positive integer",
            False,
            f"{non_integer} non-integer value(s), {non_positive} non-positive value(s)",
        )
    return Check("item_count is a positive integer", True)


def run_checks(df: pd.DataFrame) -> list[Check]:
    return [
        check_columns_present(df),
        check_row_count(df),
        check_null_rates(df),
        check_weight_ranges(df),
        check_carton_types(df),
        check_item_count(df),
    ]


def print_summary(checks: list[Check]) -> bool:
    all_passed = True
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        all_passed = all_passed and check.passed
        line = f"[{status}] {check.name}"
        if check.detail:
            line += f" - {check.detail}"
        print(line)
    print()
    print("Overall: " + ("PASS" if all_passed else "FAIL"))
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to a shipments CSV to validate")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    checks = run_checks(df)
    passed = print_summary(checks)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

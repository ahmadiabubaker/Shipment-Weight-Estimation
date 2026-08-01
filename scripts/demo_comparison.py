"""Client-facing demo: model predictions vs. the naive theoretical-weight
calculation, on real June 2026 test shipments.

Every prediction shown is produced by shipment_weight.predict.predict_shipment_weight()
-- the library's public entry point -- called with real box dimensions and raw
item lines reconstructed per shipment, exactly like a real integration would
call it. This is the actual shipped inference path, not a training-script
shortcut. Loads the PRODUCTION model (models/model.joblib, base features --
NOT models/model_extended.joblib, which is experimental).

Shipment selection is illustrative, not random, and not a new evaluation
methodology: the full June 2026 test set is reconstructed and scored the
same way scripts/train_real_data.py already does (same cleaning, same
shipment_weight.ingest/features pipeline, same trained pipeline.predict()
call), then ranked by |predicted - actual| to pick:
  - ~7 "best"    -- smallest error, the model at its best
  - ~7 "typical" -- spread evenly across the middle band (p30-p70) of the
                     absolute-error distribution, so the group shows a
                     realistic RANGE of everyday errors. Note this is not
                     "the rows closest to the 0.709 MAE" -- that criterion
                     is degenerate on this data (only 8 of 9,337 shipments
                     land within 0.0005 lbs of the MAE, so the 7 closest all
                     render as an identical +/-0.709 and read as fabricated).
                     See _select_typical().
  - ~6 "worst"   -- largest error, shown honestly, guaranteed to include at
                     least one shipment from a known high-variance box type.
                     "13cube" / "small flat rate" (both real values in this
                     data; see features_extended.py's fill_ratio note) were
                     phased out over Jan-May and have ZERO June 2026 rows
                     (verified directly against the raw data) -- so the
                     guarantee also matches "30x20x12", the box type
                     train_real_data.py's own bias-by-box-name evaluation
                     already shows has by far the highest error variance in
                     this dataset (MAE ~29 lbs / std ~44 lbs vs. 9-21 lbs for
                     other common box types), and which is genuinely present
                     in June.

The full-test-set numbers quoted in the summary (1.328 lbs baseline MAE,
0.709 lbs model MAE, 81.2% within 1lb) are the documented MODEL_CARD.md
numbers, not recomputed here -- this script does not calculate its own MAE
over the 20 shown rows and present it as the headline number; the 20 rows
are examples, the documented full-test-set numbers are the evidence.

Usage:
    python scripts/demo_comparison.py \\
        --shipments order_shipments_anonymized.xlsx \\
        --lines     order_lines_in_shipment_anonymized.xlsx \\
        [--model    models/model.joblib]
        [--out      outputs/demo_comparison_june2026.csv]

Takes a couple of minutes -- it reconstructs and scores the full June 2026
test set (~9,300 shipments) to select representative examples from, using
the same load/clean/feature-build pipeline scripts/train_real_data.py uses.
"""
from __future__ import annotations

import argparse
import os
import sys

# This calls into train_real_data.py's load/clean/build_features/time_split,
# whose diagnostic prints use non-ASCII characters (e.g. "->" as "→")
# and crash with UnicodeEncodeError on Windows' default (non-UTF-8) console
# encoding -- not hypothetical, this repo hit it. Reconfigure defensively
# before any of that output happens, since this script is meant to run live
# in front of a client.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shipment_weight.features import ALL_FEATURES, add_derived_features  # noqa: E402
from shipment_weight.ingest import LBS_TO_OZ, apply_category_map  # noqa: E402
from shipment_weight.predict import Box, ItemLine, Shipment, predict_shipment_weight  # noqa: E402

from train_real_data import build_features, clean, load_data, time_split  # noqa: E402

# Documented, already-reported full-test-set numbers (MODEL_CARD.md).
# NOT recomputed here -- see module docstring.
DOCUMENTED_BASELINE_MAE_LBS = 1.328
DOCUMENTED_MODEL_MAE_LBS = 0.709
DOCUMENTED_WITHIN_1LB_PCT = 81.2
DOCUMENTED_TEST_ROWS = 9337

N_BEST = 7
N_TYPICAL = 7
N_WORST = 6
# "Typical" = spread across the middle band of the absolute-error
# distribution, NOT clustered at the MAE -- see _select_typical() for why.
TYPICAL_BAND_LOW = 0.30
TYPICAL_BAND_HIGH = 0.70
# "flat rate"/"cube" box types (task's suggested examples, and real values
# in this data) were phased out over Jan-May 2026 and have zero June rows
# (verified against the raw data) -- "30x20x12" is included so the
# high-variance-box guarantee is actually satisfiable for the June test set:
# it's the box type train_real_data.py's own bias-by-box-name evaluation
# already shows has by far the highest error variance in this dataset.
HIGH_VARIANCE_BOX_KEYWORDS = ("flat rate", "cube", "30x20x12")

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "demo_comparison_june2026.csv")


# ── Step 1: reconstruct + score the June test set (same pipeline as training) ─

def score_june_test_set(model_path: str, shipments_path: str, lines_path: str):
    """Rebuild the June 2026 test set exactly as scripts/train_real_data.py
    does (load -> clean -> build_features -> time_split -> add_derived_features),
    then score it in bulk with the already-trained production pipeline.
    Returns (scored_test_df, deduplicated_lines_df, model_bundle).
    """
    bundle = joblib.load(model_path)

    ships_raw, lines_raw = load_data(shipments_path, lines_path)
    ships, _ = clean(ships_raw)
    df = build_features(ships, lines_raw)
    _, test = time_split(df)
    test = add_derived_features(test).reset_index(drop=True)

    # category_avg_weight_error_oz from the SAME map the shipped model
    # bundle carries (what predict_shipment_weight() will use per-shipment
    # below) -- not a freshly recomputed train-fold map -- so bulk scoring
    # here and the per-shipment library calls later agree by construction.
    test["category_avg_weight_error_oz"] = apply_category_map(
        test["category_mode"],
        bundle.get("category_error_map", {}),
        bundle.get("category_error_global_mean", 0.0),
    )

    feature_list = bundle.get("feature_list", ALL_FEATURES)
    test["predicted_weight_oz"] = bundle["pipeline"].predict(test[feature_list])
    test["model_error_lbs"] = (test["predicted_weight_oz"] - test["actual_weight_oz"]) / LBS_TO_OZ
    test["abs_model_error_lbs"] = test["model_error_lbs"].abs()

    # Same dedup rule as shipment_weight.ingest.aggregate_lines (2,153
    # duplicate (shipment_number, item_id) pairs in the raw data; keep
    # first) -- applied once here so the raw item lines used to rebuild
    # each selected shipment for predict_shipment_weight() match what the
    # bulk scoring above already saw.
    lines_dedup = lines_raw.drop_duplicates(subset=["shipment_number", "item_id"], keep="first")

    return test, lines_dedup, bundle


# ── Step 2: pick ~20 illustrative shipments ──────────────────────────────────

def select_demo_shipments(test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Worst is picked first so the high-variance-box guarantee gets first
    claim on the error ranking; best and typical are then picked from what's
    left, so no shipment appears in more than one group."""
    remaining = test

    worst_sorted = remaining.sort_values("abs_model_error_lbs", ascending=False)
    keyword_mask = worst_sorted["carton_type"].str.contains(
        "|".join(HIGH_VARIANCE_BOX_KEYWORDS), case=False, na=False
    )
    if keyword_mask.any():
        top_by_error = worst_sorted[~keyword_mask].head(N_WORST - 1)
        high_variance_pick = worst_sorted[keyword_mask].head(1)
        worst = pd.concat([top_by_error, high_variance_pick]).sort_values(
            "abs_model_error_lbs", ascending=False
        )
    else:
        worst = worst_sorted.head(N_WORST)
    remaining = remaining[~remaining["shipment_number"].isin(worst["shipment_number"])]

    best = remaining.sort_values("abs_model_error_lbs", ascending=True).head(N_BEST)
    remaining = remaining[~remaining["shipment_number"].isin(best["shipment_number"])]

    typical = _select_typical(remaining)

    return {"best": best, "typical": typical, "worst": worst}


def _select_typical(pool: pd.DataFrame) -> pd.DataFrame:
    """Pick N_TYPICAL shipments spread evenly across the MIDDLE BAND of the
    absolute-error distribution (p30..p70), one nearest-neighbour per target
    quantile, sampled without replacement.

    Deliberately NOT "the rows whose error is closest to the overall MAE".
    That was the original implementation and it is degenerate: only 8 of the
    9,337 test shipments sit within 0.0005 lbs of the 0.709 MAE, so taking
    the 7 closest guaranteed 7 rows that all render as exactly +/-0.709 at 3
    decimal places -- real predictions, but output that looks fabricated and
    invites exactly the "these numbers are made up" reaction you do not want
    in a client meeting. Spreading across a band shows a realistic range of
    everyday errors instead of 7 copies of the same number.
    """
    pool = pool.copy()
    targets = pool["abs_model_error_lbs"].quantile(
        np.linspace(TYPICAL_BAND_LOW, TYPICAL_BAND_HIGH, N_TYPICAL)
    ).values

    picked_idx = []
    for target in targets:
        candidates = pool.drop(index=picked_idx)
        if candidates.empty:
            break
        nearest = (candidates["abs_model_error_lbs"] - target).abs().idxmin()
        picked_idx.append(nearest)

    return pool.loc[picked_idx]


# ── Step 3: rebuild raw Shipment objects, predict via the public library ────

def _build_shipment(row: pd.Series, lines_for_shipment: pd.DataFrame) -> Shipment:
    items = []
    for _, line in lines_for_shipment.iterrows():
        unit_weight = line["theoretical_item_weight_lbs"]
        items.append(
            ItemLine(
                category=line["category"] if pd.notna(line["category"]) else "unknown",
                quantity=int(line["item_quantity"]) if pd.notna(line["item_quantity"]) else 1,
                unit_weight_lbs=None if pd.isna(unit_weight) else float(unit_weight),
                length_in=float(line["item_length_inches"]) if pd.notna(line["item_length_inches"]) else 0.0,
                width_in=float(line["item_width_inches"]) if pd.notna(line["item_width_inches"]) else 0.0,
                height_in=float(line["item_height_inches"]) if pd.notna(line["item_height_inches"]) else 0.0,
                sku=str(line["item_id"]) if pd.notna(line["item_id"]) else None,
            )
        )
    if not items:
        # No matching order lines for this shipment_number (rare -- the
        # same case build_features fills item_count=1 for).
        items = [ItemLine(category="unknown", quantity=1, unit_weight_lbs=None, length_in=0.0, width_in=0.0, height_in=0.0)]

    box = Box(
        box_name=str(row["carton_type"]),  # the filled name, same identity the model itself was scored on
        length_in=float(row["box_length"]),
        width_in=float(row["box_width"]),
        height_in=float(row["box_height"]),
        tare_weight_lbs=float(row["theoretical_empty_box_weight_lbs"]),
    )
    return Shipment(
        items=items,
        box=box,
        ship_method=str(row["ship_method"]),
        shipment_id=str(row["shipment_number"]),
    )


def build_demo_table(groups: dict[str, pd.DataFrame], lines_dedup: pd.DataFrame, model_path: str) -> pd.DataFrame:
    lines_by_shipment = {sn: g for sn, g in lines_dedup.groupby("shipment_number")}
    empty_lines = lines_dedup.iloc[0:0]

    demo_rows = []
    for group_name in ("best", "typical", "worst"):
        group_df = groups[group_name].sort_values("abs_model_error_lbs", ascending=(group_name != "worst"))
        for _, row in group_df.iterrows():
            shipment_lines = lines_by_shipment.get(row["shipment_number"], empty_lines)
            shipment = _build_shipment(row, shipment_lines)

            # The actual shipped inference path -- raw box dims + item
            # lines in, a prediction out, via the public library function.
            result = predict_shipment_weight(shipment, model_path=model_path)

            theoretical_lbs = row["theoretical_weight_oz"] / LBS_TO_OZ
            actual_lbs = row["actual_weight_oz"] / LBS_TO_OZ
            predicted_lbs = result.predicted_weight_lbs

            theoretical_error_lbs = theoretical_lbs - actual_lbs
            model_error_lbs = predicted_lbs - actual_lbs
            improvement_lbs = abs(theoretical_error_lbs) - abs(model_error_lbs)

            demo_rows.append(
                {
                    "demo_group": group_name,
                    "shipment_number": row["shipment_number"],
                    "box_name": row["carton_type"],
                    "ship_method": row["ship_method"],
                    "theoretical_weight_lbs": round(theoretical_lbs, 3),
                    "predicted_weight_lbs": round(predicted_lbs, 3),
                    "actual_weight_lbs": round(actual_lbs, 3),
                    "theoretical_error_lbs": round(theoretical_error_lbs, 3),
                    "model_error_lbs": round(model_error_lbs, 3),
                    "improvement_lbs": round(improvement_lbs, 3),
                }
            )

    return pd.DataFrame(demo_rows)


# ── Step 4: print + save ─────────────────────────────────────────────────────

TABLE_COLUMNS = [
    # (column, header, alignment, is_signed_float)
    ("demo_group", "group", "<", False),
    ("shipment_number", "shipment_number", "<", False),
    ("box_name", "box_name", "<", False),
    ("ship_method", "ship_method", "<", False),
    ("theoretical_weight_lbs", "theoretical_lbs", ">", False),
    ("predicted_weight_lbs", "predicted_lbs", ">", False),
    ("actual_weight_lbs", "actual_lbs", ">", False),
    ("theoretical_error_lbs", "theo_err_lbs", ">", True),
    ("model_error_lbs", "model_err_lbs", ">", True),
    ("improvement_lbs", "improvement_lbs", ">", True),
]

GROUP_LABELS = {
    "best": "BEST CASES  (smallest prediction error)",
    "typical": "TYPICAL CASES  (spread across the middle of the error distribution)",
    "worst": "WORST CASES  (largest prediction error -- shown honestly)",
}


def _fmt_cell(value, is_signed_float: bool) -> str:
    if isinstance(value, float):
        return f"{value:+.3f}" if is_signed_float else f"{value:.3f}"
    return str(value)


def print_table(df: pd.DataFrame) -> None:
    widths = {
        col: max(len(header), df[col].map(lambda v, c=col, s=signed: len(_fmt_cell(v, s))).max())
        for col, header, _, signed in TABLE_COLUMNS
    }

    def render_row(values: dict) -> str:
        cells = []
        for col, header, align, signed in TABLE_COLUMNS:
            text = _fmt_cell(values[col], signed)
            cells.append(text.rjust(widths[col]) if align == ">" else text.ljust(widths[col]))
        return "  ".join(cells)

    header_line = render_row({col: header for col, header, _, _ in TABLE_COLUMNS})
    print(header_line)
    print("-" * len(header_line))

    last_group = None
    for _, row in df.iterrows():
        if row["demo_group"] != last_group:
            if last_group is not None:
                print()
            print(GROUP_LABELS[row["demo_group"]])
            last_group = row["demo_group"]
        print(render_row(row.to_dict()))


def print_summary(df: pd.DataFrame) -> None:
    avg_improvement = df["improvement_lbs"].mean()
    print()
    print(f"Average improvement across these {len(df)} shipments: {avg_improvement:+.3f} lbs "
          f"closer to actual than the theoretical calculation.")
    print(f"These {len(df)} shipments are illustrative, not the evidence base -- the full June 2026 "
          f"test set ({DOCUMENTED_TEST_ROWS:,} shipments) shows a baseline MAE of "
          f"{DOCUMENTED_BASELINE_MAE_LBS} lbs vs. a model MAE of {DOCUMENTED_MODEL_MAE_LBS} lbs, "
          f"with {DOCUMENTED_WITHIN_1LB_PCT}% of predictions within 1 lb of actual.")
    print("See MODEL_CARD.md for the full evaluation.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shipments", required=True, help="Path to order_shipments_anonymized.xlsx")
    parser.add_argument("--lines", required=True, help="Path to order_lines_in_shipment_anonymized.xlsx")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the production model bundle (default: models/model.joblib)")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH, help="CSV output path (default: outputs/demo_comparison_june2026.csv)")
    args = parser.parse_args()

    test, lines_dedup, _bundle = score_june_test_set(args.model, args.shipments, args.lines)
    groups = select_demo_shipments(test)
    demo_df = build_demo_table(groups, lines_dedup, args.model)

    print_table(demo_df)
    print_summary(demo_df)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    demo_df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

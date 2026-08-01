"""Extended feature set built on top of order_lines_in_shipment data.

This module is additive only — it imports from ``shipment_weight.features``
but never modifies it, so the base pipeline (used by the API and the
current production model) is untouched and the two feature sets can be
diffed/ablated against each other.

New features:
  - distinct_sku_count        (already aggregated upstream, just newly modeled)
  - weight_density_lbs_per_in3
  - top_category_by_weight    (categorical; mode by summed item weight, not count)
  - has_multiple_categories
  - fill_ratio_winsorized     (alternative to the existing hard-capped fill_ratio)
  - is_overfilled
  - box_name_target_enc_oz    (leakage-safe expanding-window target encoding)
  - ship_method_target_enc_oz (leakage-safe expanding-window target encoding)

``distinct_sku_count`` and ``num_categories``/``item_count`` were considered
for this module too (they map to the brief's "distinct_sku_count",
"n_categories", "total_units") but ``item_count`` and ``num_categories``
already exist in ``shipment_weight.features`` under those names computed
from the same order-lines join, so only ``distinct_sku_count`` is newly
added to the modeled feature list here — the other two are reused as-is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shipment_weight.features import ALL_FEATURES as BASE_ALL_FEATURES
from shipment_weight.features import CATEGORICAL_FEATURES as BASE_CATEGORICAL_FEATURES
from shipment_weight.features import NUMERIC_FEATURES as BASE_NUMERIC_FEATURES

EXTENDED_NUMERIC_FEATURES = [
    "distinct_sku_count",
    "weight_density_lbs_per_in3",
    "has_multiple_categories",
    "fill_ratio_winsorized",
    "is_overfilled",
    "box_name_target_enc_oz",
    "ship_method_target_enc_oz",
]
EXTENDED_CATEGORICAL_FEATURES = [
    "top_category_by_weight",
]

NUMERIC_FEATURES_EXTENDED = BASE_NUMERIC_FEATURES + EXTENDED_NUMERIC_FEATURES
CATEGORICAL_FEATURES_EXTENDED = BASE_CATEGORICAL_FEATURES + EXTENDED_CATEGORICAL_FEATURES
ALL_FEATURES_EXTENDED = BASE_ALL_FEATURES + EXTENDED_NUMERIC_FEATURES + EXTENDED_CATEGORICAL_FEATURES


# ── Per-shipment aggregates from order lines not already produced upstream ──

def top_category_by_weight(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per shipment_number: the category with the largest summed
    theoretical item weight (item_quantity * theoretical_item_weight_lbs).

    Deliberately different from the existing ``category_mode`` (most
    frequent category by *line count*) — this is weight-weighted, so a
    shipment with one heavy item and five light accessories is attributed
    to the heavy item's category.
    """
    lines = lines.drop_duplicates(subset=["shipment_number", "item_id"], keep="first").copy()
    lines["line_weight_lbs"] = (
        lines["item_quantity"].fillna(1) * lines["theoretical_item_weight_lbs"].fillna(0)
    )
    cat_weight = (
        lines.groupby(["shipment_number", "category"])["line_weight_lbs"]
        .sum()
        .reset_index()
    )
    idx = cat_weight.groupby("shipment_number")["line_weight_lbs"].idxmax()
    return (
        cat_weight.loc[idx, ["shipment_number", "category"]]
        .rename(columns={"category": "top_category_by_weight"})
        .reset_index(drop=True)
    )


def add_extended_features(df: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Attach the new shipment-level features to a dataframe that has
    already been through ``train_real_data.build_features`` (needs
    ``distinct_sku_count``, ``total_item_volume_in3``, ``box_volume_in3``,
    ``theoretical_cargo_weight_lbs``, ``num_categories``).

    Everything here is derived from catalog/dimensional data only — never
    from ``actual_weight_lbs`` — so none of it can leak the target. Values
    that depend on train-fold statistics (fallback fill values, winsorize
    threshold) are intentionally left as raw/NaN here and finalized
    per-fold in ``finalize_train_fold_stats`` after the time split.
    """
    df = df.copy()

    top_cat = top_category_by_weight(lines)
    df = df.merge(top_cat, on="shipment_number", how="left")
    df["top_category_by_weight"] = df["top_category_by_weight"].fillna("unknown")

    df["has_multiple_categories"] = (df["num_categories"] > 1).astype(int)

    # weight_density: theoretical_cargo_weight_lbs / total_item_volume_in3.
    # 41 shipments have total_item_volume_in3 == 0 (audit finding) -> would
    # divide by zero; leave as NaN here, filled with the train-fold median
    # in finalize_train_fold_stats.
    vol = df["total_item_volume_in3"].replace(0, np.nan)
    df["weight_density_lbs_per_in3"] = df["theoretical_cargo_weight_lbs"] / vol

    # fill_ratio_winsorized: same ratio as the existing `fill_ratio` but
    # winsorized at the train-fold 99th percentile instead of hard-capped
    # at 1.5. Investigation: 86% of fill_ratio > 5 rows are "small flat
    # rate" (328/426) or "18f" (30/426) boxes, where nominal box dims
    # dramatically understate real packed capacity (poly-mailer overstuff,
    # not sensor/entry noise) -- see scratchpad investigation. Left raw
    # here; clipped to the train-fold p99 threshold downstream.
    df["fill_ratio_raw"] = df["total_item_volume_in3"] / df["box_volume_in3"]
    df["is_overfilled"] = (df["total_item_volume_in3"] > df["box_volume_in3"]).astype(int)

    return df


def finalize_train_fold_stats(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fallback/winsorize statistics from the TRAIN fold only and
    apply them to both train and test -- same pattern as the existing
    ``encode_category_error`` (train-fold-only, no leakage from test)."""
    train = train.copy()
    test = test.copy()

    density_median = train["weight_density_lbs_per_in3"].median()
    train["weight_density_lbs_per_in3"] = train["weight_density_lbs_per_in3"].fillna(density_median)
    test["weight_density_lbs_per_in3"] = test["weight_density_lbs_per_in3"].fillna(density_median)

    p99 = train["fill_ratio_raw"].quantile(0.99)
    train["fill_ratio_winsorized"] = train["fill_ratio_raw"].clip(upper=p99)
    test["fill_ratio_winsorized"] = test["fill_ratio_raw"].clip(upper=p99)

    return train, test


# ── Leakage-safe expanding-window target encoding ───────────────────────────

def expanding_target_encode(
    df: pd.DataFrame,
    group_col: str,
    residual_col: str,
    date_col: str = "order_date",
) -> pd.Series:
    """Encode ``group_col`` with the mean of ``residual_col`` computed ONLY
    from rows whose ``date_col`` is strictly earlier than the row's own
    date (chronological expanding window; same-day rows never see each
    other, and no row ever sees a later date).

    Cold start (a group's first-ever occurrence, or the very first date in
    the whole dataset) falls back to the global (all-groups) expanding
    mean computed the same way -- strictly-prior days only, never a
    full-dataset constant. If even that is unavailable (the first date in
    the entire dataset), falls back to 0.0 (equivalent to "assume the
    theoretical weight is unbiased" with zero evidence otherwise).

    Must be called on the full chronological dataframe (train+test
    together, pre-split) so later rows can draw on earlier rows regardless
    of which side of the train/test split they land on -- this is what
    makes it correct for real deployment, where yesterday's actual weights
    are genuinely known before today's shipment ships.
    """
    d = df[[group_col, residual_col, date_col]].reset_index(drop=True).copy()
    d["_orig_order"] = np.arange(len(d))

    daily = (
        d.groupby([group_col, date_col])[residual_col]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values([group_col, date_col])
    )
    daily["cum_sum_excl"] = daily.groupby(group_col)["sum"].cumsum() - daily["sum"]
    daily["cum_count_excl"] = daily.groupby(group_col)["count"].cumsum() - daily["count"]
    daily["group_encoding"] = daily["cum_sum_excl"] / daily["cum_count_excl"]

    global_daily = (
        d.groupby(date_col)[residual_col]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(date_col)
    )
    global_daily["cum_sum_excl"] = global_daily["sum"].cumsum() - global_daily["sum"]
    global_daily["cum_count_excl"] = global_daily["count"].cumsum() - global_daily["count"]
    global_daily["global_encoding"] = global_daily["cum_sum_excl"] / global_daily["cum_count_excl"]

    daily = daily.merge(global_daily[[date_col, "global_encoding"]], on=date_col, how="left")
    daily["final_encoding"] = daily["group_encoding"].fillna(daily["global_encoding"])

    merged = d.merge(
        daily[[group_col, date_col, "final_encoding"]],
        on=[group_col, date_col],
        how="left",
    ).sort_values("_orig_order")

    return merged["final_encoding"].fillna(0.0).reset_index(drop=True)


def train_fold_only_encode(
    train: pd.DataFrame, test: pd.DataFrame, group_col: str, residual_col: str
) -> tuple[pd.Series, pd.Series]:
    """Frozen alternative to expanding_target_encode: computes ONE mean
    residual per group from the TRAIN fold only (identical pattern to the
    existing ``encode_category_error``) and applies that fixed value to
    every test row, regardless of the test row's own date.

    Diagnostic/comparison variant only -- exists to separate how much of
    the expanding encoder's gain is genuine forward-looking signal
    (available even with a static, e.g. monthly, refresh) from the extra
    lift that comes from rolling the encoding forward using early-test-
    period actuals to predict later-test-period rows.
    """
    group_map = train.groupby(group_col)[residual_col].mean()
    global_mean = float(train[residual_col].mean())
    train_enc = train[group_col].map(group_map).fillna(global_mean)
    test_enc = test[group_col].map(group_map).fillna(global_mean)
    return train_enc, test_enc


def add_target_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add box_name_target_enc_oz and ship_method_target_enc_oz to the full
    chronological dataframe (call BEFORE time_split). Requires
    actual_weight_oz, theoretical_weight_oz, box_name/carton_type,
    ship_method, order_date.
    """
    df = df.copy()
    residual_oz = df["actual_weight_oz"] - df["theoretical_weight_oz"]
    work = df.assign(_residual_oz=residual_oz)

    df["box_name_target_enc_oz"] = expanding_target_encode(
        work.assign(_group=work["carton_type"]), "_group", "_residual_oz", "order_date"
    ).values
    df["ship_method_target_enc_oz"] = expanding_target_encode(
        work.assign(_group=work["ship_method"]), "_group", "_residual_oz", "order_date"
    ).values
    return df

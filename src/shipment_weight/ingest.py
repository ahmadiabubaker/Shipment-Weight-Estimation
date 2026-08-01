"""Raw-data-to-feature-table ingestion, shared by training and inference.

Moved out of ``scripts/train_real_data.py`` so the exact same
order-lines-aggregation and shipment-feature-construction logic used to
build the training table is also what ``shipment_weight.predict`` uses to
build a feature row for a live prediction. There is no second, simplified
feature-building path for library callers -- this module is it, for both.

``scripts/train_real_data.py`` re-exports ``build_features``, ``LBS_TO_OZ``,
etc. from here so its own callers (``train_real_data_extended.py``,
``investigate_gbt.py``, ``demo_real_data.py``) are unaffected by the move.
"""
from __future__ import annotations

import pandas as pd

LBS_TO_OZ = 16.0


def _safe_mode(s: pd.Series) -> str:
    m = s.dropna().mode()
    return str(m.iloc[0]) if len(m) > 0 else "unknown"


def aggregate_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse order-line rows to one row per shipment_number.

    Duplicate (shipment_number, item_id) pairs are treated as double-entry
    errors; the first occurrence is kept so quantities are not double-counted
    (matches the real-data audit finding in scripts/audit_real_data.py).
    """
    lines = lines.copy()
    lines = lines.drop_duplicates(subset=["shipment_number", "item_id"], keep="first")
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
    """Join shipment-level rows with order-line aggregates and compute the
    columns ``add_derived_features`` (features.py) needs: carton_type,
    box_volume_in3, theoretical_weight_oz, packing_material,
    category_avg_weight_error_oz (left NaN -- filled by the caller from
    either a train-fold encoding or a saved model-bundle category map).

    Works identically whether ``ships``/``lines`` is a full historical
    export (training) or a single freshly-built shipment/line pair
    (a live prediction) -- same columns in, same columns out.
    """
    agg = aggregate_lines(lines)

    df = ships.merge(agg, on="shipment_number", how="left")
    df["item_count"] = df["item_count"].fillna(1).clip(lower=1)
    df["total_item_volume_in3"] = df["total_item_volume_in3"].fillna(0)
    df["num_missing_catalog_weights"] = df["num_missing_catalog_weights"].fillna(0)
    df["item_categories"] = df["item_categories"].fillna("unknown")
    df["category_mode"] = df["category_mode"].fillna("unknown")

    # Unit conversions lbs -> oz
    df["theoretical_weight_oz"] = df["total_theoretical_shipment_weight_lbs"] * LBS_TO_OZ
    df["actual_weight_oz"] = df["actual_weight_lbs"] * LBS_TO_OZ if "actual_weight_lbs" in df.columns else None

    # Fill null box_name with "{length}x{width}x{height}" before any carton features
    null_box = df["box_name"].isnull()
    if null_box.any():
        df.loc[null_box, "box_name"] = (
            df.loc[null_box, "box_length"].fillna(0).round(0).astype(int).astype(str) + "x" +
            df.loc[null_box, "box_width"].fillna(0).round(0).astype(int).astype(str) + "x" +
            df.loc[null_box, "box_height"].fillna(0).round(0).astype(int).astype(str)
        )

    # Carton identity and box volume -- real box dimensions, not the
    # synthetic 4-entry CARTON_CAPACITY table in features.py. This is what
    # makes fill_ratio/void_volume_in3 (computed downstream in
    # add_derived_features) reflect the actual carton, for both training
    # rows and a live prediction.
    df["carton_type"] = df["box_name"].fillna("UNKNOWN_BOX")
    df["box_volume_in3"] = df["box_length"] * df["box_width"] * df["box_height"]

    # packing_material is not captured in real data;
    # OneHotEncoder(handle_unknown='ignore') produces an all-zero encoding.
    df["packing_material"] = "unknown"

    # category_avg_weight_error_oz is a target-derived encoding. At training
    # time it's filled per-fold from train-only data (encode_category_error
    # in train_real_data.py). At inference time it's filled from the map
    # saved in the model bundle (apply_category_map, below). Left NaN here
    # either way so both callers fill it the same way, from the same map.
    df["category_avg_weight_error_oz"] = float("nan")

    return df


def apply_category_map(category: pd.Series, category_map: dict, global_mean: float) -> pd.Series:
    """Apply a saved (or freshly computed) category -> mean-residual map to
    a category_mode column, falling back to ``global_mean`` for categories
    absent from the map (unseen at train time). Single source of truth for
    this lookup, used both when a train-fold map is computed fresh
    (train_real_data.encode_category_error) and when a map persisted in a
    model bundle is applied at serving time (shipment_weight.predict).
    """
    return category.map(category_map).fillna(global_mean)

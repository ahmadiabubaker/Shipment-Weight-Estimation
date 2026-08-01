"""Feature engineering and preprocessing for the shipment weight model.

Shared by both the training/evaluation pipeline and the FastAPI service so the
two never drift apart.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shipment_weight.data_gen import CARTON_TYPES

TARGET = "actual_weight_oz"

# Synthetic-data-only. Real carton_type values are warehouse box names
# (e.g. "14x10x8") and essentially never match these 4 keys, so this dict
# is only ever consulted by the Phase 1 synthetic pipeline (data_gen.py /
# train.py) and tests that use synthetic carton names. Real training
# (scripts/train_real_data.py) and the library (shipment_weight.predict)
# always supply a real box_volume_in3, which add_derived_features prefers
# as the fallback below -- do not extend this dict for real carton types.
CARTON_CAPACITY = {name: capacity for name, capacity, _ in CARTON_TYPES}

NUMERIC_FEATURES = [
    "theoretical_weight_oz",
    "item_count",
    "total_item_volume_in3",
    "void_volume_in3",
    "category_avg_weight_error_oz",
    "weight_per_item_oz",
    "fill_ratio",
    "num_categories",
]
CATEGORICAL_FEATURES = [
    "carton_type",
    "ship_method",
    "packing_material",
    "category_mode",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features computable from the raw shipment columns. Idempotent.

    When a ``box_volume_in3`` column is present (real warehouse data where
    box_name doesn't match the synthetic CARTON_CAPACITY keys), it is used as
    a fallback capacity so fill_ratio is still meaningful.
    """
    df = df.copy()
    df["weight_per_item_oz"] = df["theoretical_weight_oz"] / df["item_count"].clip(lower=1)
    capacity = df["carton_type"].map(CARTON_CAPACITY)
    if "box_volume_in3" in df.columns:
        capacity = capacity.fillna(df["box_volume_in3"])
    df["carton_capacity_in3"] = capacity
    df["fill_ratio"] = (df["total_item_volume_in3"] / df["carton_capacity_in3"]).clip(upper=1.5)
    if "box_volume_in3" in df.columns:
        df["void_volume_in3"] = (df["box_volume_in3"] - df["total_item_volume_in3"]).clip(lower=0)
    else:
        df["void_volume_in3"] = 0.0
    df["num_categories"] = df["item_categories"].fillna("").apply(
        lambda s: len([c for c in s.split(",") if c])
    )
    df["has_unknown_category"] = df["item_categories"].fillna("").str.contains("unknown").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that imputes missing values and encodes
    categoricals, degrading gracefully (handle_unknown='ignore') for carton
    types / categories never seen during training."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

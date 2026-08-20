"""Public library interface for shipment weight prediction.

This is the primary way to get a prediction out of a trained model --
FastAPI (api/main.py), if kept, is a thin optional wrapper around this, not
a second implementation.

Callers pass real box dimensions/name and raw item lines (``Shipment``,
``Box``, ``ItemLine``) -- never pre-computed features. Internally,
``build_feature_frame`` calls the exact same feature-engineering functions
used by training (``shipment_weight.ingest.build_features`` and
``shipment_weight.features.add_derived_features``), so a single item line
and a single trained-on-historical-data row are computed identically. This
is what makes the earlier serving-time bug (``fill_ratio`` always NaN
because carton_type never matched the synthetic CARTON_CAPACITY table,
``void_volume_in3`` hardcoded to 0.0 because the request had no box volume,
``category_avg_weight_error_oz`` a guess because the train-fold map was
never saved anywhere) structurally impossible: there is only one feature
pipeline, and it always has the inputs it needs because those inputs
(box dimensions, item lines) are now part of the public contract instead of
being pre-aggregated away before they reach the library.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from shipment_weight.features import ALL_FEATURES, add_derived_features
from shipment_weight.ingest import LBS_TO_OZ, apply_category_map
from shipment_weight.ingest import build_features as _build_features
from shipment_weight.rounding import round_to_billing_tier

CONFIDENCE_Z = 1.645  # ~90% interval for a Gaussian residual assumption


# ── Public input/output types ────────────────────────────────────────────────

@dataclass
class ItemLine:
    """One line item in a shipment. ``unit_weight_lbs=None`` represents a
    missing catalog weight (matches real data's null theoretical_item_weight_lbs,
    treated as 0 contribution to theoretical weight -- same as data_gen.py /
    train_real_data.py)."""

    category: str
    quantity: int
    unit_weight_lbs: float | None
    length_in: float
    width_in: float
    height_in: float
    sku: str | None = None


@dataclass
class Box:
    """The physical carton. ``box_name`` is a real warehouse box identifier
    (e.g. "14x10x8"), not one of the 4 synthetic CARTON_CAPACITY keys --
    capacity/void are always computed from length_in x width_in x height_in,
    never looked up against synthetic data."""

    box_name: str
    length_in: float
    width_in: float
    height_in: float
    tare_weight_lbs: float = 0.0


@dataclass
class Shipment:
    items: list[ItemLine]
    box: Box
    ship_method: str
    shipment_id: str | None = None
    order_date: str | None = None  # informational only; not used by the base feature set


@dataclass
class WeightPrediction:
    shipment_id: str | None
    predicted_weight_oz: float
    # Precise, unrounded prediction -- use this for internal accuracy
    # tracking, evaluation, and anything statistical. Unchanged by the
    # addition of predicted_weight_lbs_for_label below.
    predicted_weight_lbs: float
    # Rounded per shipment_weight.rounding.round_to_billing_tier: ALWAYS
    # rounds UP to the next whole pound for predictions >= 1 lb (never down
    # or to nearest -- carriers round their own measured weight the same
    # way before billing, verified against real data), left unrounded
    # below 1 lb. This is the presentation-layer value meant for a physical
    # shipping label, not for accuracy tracking.
    predicted_weight_lbs_for_label: float
    theoretical_weight_oz: float
    theoretical_weight_lbs: float
    adjustment_oz: float
    # Deliberately computed from the precise (unrounded) prediction, not
    # predicted_weight_lbs_for_label -- rounding is a presentation step for
    # the label, not a statistical one, so the interval should reflect the
    # model's actual residual uncertainty around its real point estimate.
    confidence_interval_oz: tuple[float, float]
    confidence_level: float
    model_version: str
    model_type: str
    # The features most relevant to the original serving-skew bug, exposed
    # so callers (and tests) can confirm they were computed from real
    # inputs rather than silently zeroed/imputed.
    features: dict = field(default_factory=dict)


# ── Feature construction (reuses shipment_weight.ingest + features) ─────────

def _to_raw_frames(shipments: Sequence[Shipment]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the same shipment-level / order-line-level column shapes that
    scripts/train_real_data.py reads from the Medusa Excel export, from
    Shipment objects instead. Internal join keys are always freshly
    generated (never derived from caller-supplied IDs) so caller-supplied
    ``shipment_id``/``sku`` values can never collide and corrupt the join.
    """
    ship_rows = []
    line_rows = []
    for i, shipment in enumerate(shipments):
        row_key = f"_row_{i}"
        cargo_lbs = 0.0
        for j, item in enumerate(shipment.items):
            line_rows.append(
                {
                    "shipment_number": row_key,
                    "item_id": item.sku or f"_line_{i}_{j}",
                    "item_quantity": item.quantity,
                    "item_width_inches": item.width_in,
                    "item_length_inches": item.length_in,
                    "item_height_inches": item.height_in,
                    "theoretical_item_weight_lbs": item.unit_weight_lbs,
                    "category": item.category,
                }
            )
            if item.unit_weight_lbs is not None:
                cargo_lbs += item.unit_weight_lbs * item.quantity

        tare_lbs = shipment.box.tare_weight_lbs or 0.0
        ship_rows.append(
            {
                "shipment_number": row_key,
                "ship_method": shipment.ship_method,
                "box_name": shipment.box.box_name,
                "box_length": shipment.box.length_in,
                "box_width": shipment.box.width_in,
                "box_height": shipment.box.height_in,
                "theoretical_cargo_weight_lbs": cargo_lbs,
                "theoretical_empty_box_weight_lbs": tare_lbs,
                "total_theoretical_shipment_weight_lbs": cargo_lbs + tare_lbs,
            }
        )
    return pd.DataFrame(ship_rows), pd.DataFrame(line_rows)


def build_feature_frame(
    shipments: Sequence[Shipment],
    category_error_map: dict | None = None,
    category_error_global_mean: float = 0.0,
) -> pd.DataFrame:
    """Build a model-ready feature dataframe (one row per shipment) from raw
    Shipment objects, using the SAME functions training uses:
    ``shipment_weight.ingest.build_features`` (order-line aggregation, box
    volume, theoretical weight) then ``shipment_weight.features.add_derived_features``
    (fill_ratio, void_volume_in3, weight_per_item_oz, num_categories) --
    no separate/simplified feature path for callers of this library.

    ``category_error_map``/``category_error_global_mean`` should come from
    the trained model's bundle (``ShipmentWeightPredictor`` does this
    automatically); passed explicitly here so this function is testable
    and usable standalone.
    """
    ships, lines = _to_raw_frames(shipments)
    df = _build_features(ships, lines)
    df["category_avg_weight_error_oz"] = apply_category_map(
        df["category_mode"], category_error_map or {}, category_error_global_mean
    )
    return add_derived_features(df)


# ── Model loading ─────────────────────────────────────────────────────────

def _default_model_path() -> Path:
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env)
    try:
        packaged = resources.files("shipment_weight") / "models" / "model.joblib"
        if packaged.is_file():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    # Dev-mode fallback: repo_root/models/model.joblib, same default api/main.py uses.
    return Path(__file__).resolve().parents[2] / "models" / "model.joblib"


class ShipmentWeightPredictor:
    """Loads a trained model bundle and serves predictions from raw
    Shipment input. This is the library's main entry point."""

    def __init__(self, bundle: dict) -> None:
        self._bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.model_type = bundle.get("model_type", "unknown")
        self.model_version = bundle.get("model_version", "unknown")
        self.residual_std = float(bundle.get("residual_std", 0.0))
        self.feature_list = bundle.get("feature_list", ALL_FEATURES)
        self.category_error_map = bundle.get("category_error_map", {})
        self.category_error_global_mean = float(bundle.get("category_error_global_mean", 0.0))

    @classmethod
    def load(cls, model_path: str | os.PathLike | None = None) -> "ShipmentWeightPredictor":
        path = Path(model_path) if model_path else _default_model_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"No model artifact at {path}. Train one first "
                f"(scripts/train_real_data.py) or pass model_path=/set MODEL_PATH."
            )
        return cls(joblib.load(path))

    def build_feature_frame(self, shipments: Sequence[Shipment]) -> pd.DataFrame:
        return build_feature_frame(shipments, self.category_error_map, self.category_error_global_mean)

    def predict_batch(self, shipments: Sequence[Shipment]) -> list[WeightPrediction]:
        if not shipments:
            return []
        df = self.build_feature_frame(shipments)
        X = df[self.feature_list]
        preds = self.pipeline.predict(X)
        half_width = CONFIDENCE_Z * self.residual_std

        results = []
        for i, shipment in enumerate(shipments):
            pred_oz = float(preds[i])
            pred_lbs_precise = pred_oz / LBS_TO_OZ
            label_lbs, _ = round_to_billing_tier(pred_lbs_precise)
            theo_oz = float(df["theoretical_weight_oz"].iloc[i])
            results.append(
                WeightPrediction(
                    shipment_id=shipment.shipment_id,
                    predicted_weight_oz=round(pred_oz, 2),
                    predicted_weight_lbs=round(pred_lbs_precise, 3),
                    predicted_weight_lbs_for_label=float(label_lbs),
                    theoretical_weight_oz=round(theo_oz, 2),
                    theoretical_weight_lbs=round(theo_oz / LBS_TO_OZ, 3),
                    adjustment_oz=round(pred_oz - theo_oz, 2),
                    confidence_interval_oz=(round(pred_oz - half_width, 2), round(pred_oz + half_width, 2)),
                    confidence_level=0.90,
                    model_version=self.model_version,
                    model_type=self.model_type,
                    features={
                        "fill_ratio": float(df["fill_ratio"].iloc[i]),
                        "void_volume_in3": float(df["void_volume_in3"].iloc[i]),
                        "category_avg_weight_error_oz": float(df["category_avg_weight_error_oz"].iloc[i]),
                    },
                )
            )
        return results

    def predict(self, shipment: Shipment) -> WeightPrediction:
        return self.predict_batch([shipment])[0]


@lru_cache(maxsize=8)
def _cached_predictor(model_path: str) -> ShipmentWeightPredictor:
    return ShipmentWeightPredictor.load(model_path)


def predict_shipment_weight(shipment: Shipment, model_path: str | os.PathLike | None = None) -> WeightPrediction:
    """Convenience one-shot entry point: predictor loading is cached per
    resolved model path, so repeated calls don't re-read the artifact from disk."""
    path = str(Path(model_path)) if model_path else str(_default_model_path())
    return _cached_predictor(path).predict(shipment)

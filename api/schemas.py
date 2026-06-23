"""Pydantic request/response models for the prediction endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ShipmentRequest(BaseModel):
    theoretical_weight_oz: float = Field(..., gt=0)
    item_count: int = Field(..., gt=0)
    total_item_volume_in3: float = Field(..., ge=0)
    item_categories: str = Field(..., description="Comma-separated category names, e.g. 'electronics,apparel'")
    category_mode: str = Field(..., description="Most frequent item category in the shipment")
    carton_type: str
    ship_method: str
    packing_material: str
    num_missing_catalog_weights: int = Field(0, ge=0)
    category_avg_weight_error_oz: float = Field(0.0, description="Historical avg weight error for these categories")


class ConfidenceInterval(BaseModel):
    interval_oz: tuple[float, float]
    level: float


class PredictionResponse(BaseModel):
    predicted_weight_oz: float
    theoretical_weight_oz: float
    adjustment_oz: float
    confidence: ConfidenceInterval
    model_version: str

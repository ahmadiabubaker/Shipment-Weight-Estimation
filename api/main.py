"""Minimal FastAPI service for shipment weight prediction.

Loads a trained model bundle (pipeline + residual std) at startup. No DB,
Redis, or auth yet -- those are documented as future-state in README.md and
will be added once deployment scope is confirmed with the client.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shipment_weight.features import ALL_FEATURES, add_derived_features  # noqa: E402

from api.schemas import ConfidenceInterval, PredictionResponse, ShipmentRequest  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib"))
CONFIDENCE_Z = 1.645  # ~90% interval for a Gaussian residual assumption

model_bundle: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        model_bundle.update(joblib.load(MODEL_PATH))
    yield


app = FastAPI(title="Shipment Weight Estimation API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": bool(model_bundle)}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: ShipmentRequest):
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([request.model_dump()])
    row = add_derived_features(row)
    X = row[ALL_FEATURES]

    pipeline = model_bundle["pipeline"]
    predicted_weight = float(pipeline.predict(X)[0])

    residual_std = model_bundle["residual_std"]
    half_width = CONFIDENCE_Z * residual_std

    return PredictionResponse(
        predicted_weight_oz=round(predicted_weight, 2),
        theoretical_weight_oz=request.theoretical_weight_oz,
        adjustment_oz=round(predicted_weight - request.theoretical_weight_oz, 2),
        confidence=ConfidenceInterval(
            interval_oz=(round(predicted_weight - half_width, 2), round(predicted_weight + half_width, 2)),
            level=0.90,
        ),
        model_version=model_bundle["model_version"],
    )

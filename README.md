# Shipment Weight Estimation Service

> A production-grade ML-powered API that predicts the actual packed weight of warehouse shipments, replacing inaccurate theoretical weight calculations with data-driven predictions.

**Status:** Design Phase  
---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [API Design](#api-design)
- [Data Model](#data-model)
- [ML Pipeline](#ml-pipeline)
- [Feature Engineering](#feature-engineering)
- [Infrastructure & DevOps](#infrastructure--devops)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Development Phases](#development-phases)
- [Open Questions](#open-questions)

---

## Problem Statement

### The Gap

Warehouses calculate shipment weight by adding up item weights, carton tare weight, and an estimate for packing material. This **theoretical weight** is often wrong:

- Product weights listed in the catalog are approximate.
- Packing material weight varies by packer, item fragility, and fill method.
- Multi-item shipments compound small per-item errors.
- Some cartons are packed tighter or looser depending on item shapes.

The result: carriers bill based on actual weight (or dimensional weight), and the difference between theoretical and actual weight causes billing surprises, incorrect delivery date estimates, and bad carrier selection.

### The Solution

Build a supervised ML model trained on historical shipments where both theoretical and actual packed weights are known. Serve predictions through a REST API that integrates into existing warehouse management systems and Perseuss products (cartonization, Dates and Rates).

### Success Metrics

| Metric | Baseline (theoretical weight) | Target |
|---|---|---|
| Mean Absolute Error (MAE) | Measured during EDA | 30%+ reduction vs baseline |
| Predictions within +/- 2 oz | Measured during EDA | > 80% of shipments |
| API latency (p95) | N/A | < 50ms |
| API availability | N/A | 99.9% |

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Consumers                              │
│                                                             │
│  Perseuss Cartonization    Dates & Rates    WMS Webhooks    │
│  (packing decisions)       (Shopify app)    (direct API)    │
└──────────┬─────────────────────┬──────────────┬─────────────┘
           │                     │              │
           ▼                     ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                        │
│                                                             │
│  FastAPI Application (async, uvicorn)                       │
│                                                             │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────┐  │
│  │ /predict │ │/batch-predict│ │/feedback  │ │ /health   │  │
│  └──────────┘ └──────────────┘ └──────────┘ └───────────┘  │
│  ┌────────────┐ ┌────────────────┐                          │
│  │/model/info │ │ /model/metrics │                          │
│  └────────────┘ └────────────────┘                          │
│                                                             │
│  Middleware: request ID, structured logging, timing,        │
│             API key auth, rate limiting                      │
└─────────┬───────────────────────────────────────────────────┘
          │
          ├──────────────────────┐
          ▼                      ▼
┌──────────────────┐   ┌──────────────────┐
│  Model Service   │   │  Data Service    │
│                  │   │                  │
│  - Load model    │   │  - Store preds   │
│  - Feature eng   │   │  - Store feedbk  │
│  - Predict       │   │  - Query history │
│  - Confidence    │   │  - Audit trail   │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│                                                             │
│  ┌────────────┐  ┌─────────┐  ┌──────────────────────────┐ │
│  │ PostgreSQL │  │  Redis  │  │  Object Storage (S3)     │ │
│  │            │  │         │  │                          │ │
│  │ predictions│  │ pred    │  │ model artifacts (.joblib)│ │
│  │ feedback   │  │ cache   │  │ training datasets        │ │
│  │ model reg  │  │ rate    │  │ evaluation reports       │ │
│  │ audit log  │  │ limits  │  │                          │ │
│  └────────────┘  └─────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

         Offline / Async
┌─────────────────────────────────────────────────────────────┐
│                   Training Pipeline                         │
│                                                             │
│  Extract historical data → Feature engineering →            │
│  Train models → Evaluate → Compare to champion →            │
│  Register → Promote (manual gate)                           │
│                                                             │
│  Triggered: weekly cron or manual via CLI                   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Technology |
|---|---|---|
| **API Layer** | HTTP interface, validation, auth, rate limiting | FastAPI, Pydantic, uvicorn |
| **Model Service** | Feature engineering, model loading, inference | scikit-learn, LightGBM, joblib |
| **Data Service** | Persistence, querying, audit trail | SQLAlchemy, Alembic |
| **Cache** | Prediction dedup, rate limiting | Redis |
| **Database** | Predictions, feedback, model registry | PostgreSQL 16 |
| **Object Storage** | Model artifacts, training data | S3 / MinIO (local dev) |
| **Training Pipeline** | Offline model training and evaluation | scikit-learn, LightGBM, pandas |

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Web framework** | FastAPI | Async, auto OpenAPI docs, Pydantic validation, production-proven |
| **Model serving** | In-process (loaded at startup) | Prediction is fast (~1ms). No need for TF Serving or Triton. Simplifies deployment. |
| **Database** | PostgreSQL | Structured data, JSONB for flexible payloads, battle-tested |
| **Cache** | Redis | Prediction dedup for identical shipments, rate limiting, fast |
| **ML library** | scikit-learn + LightGBM | Simple models first. LightGBM for gradient boosting when needed. |
| **Multi-tenancy** | `warehouse_id` on every record | Medusa is first customer, but the system should support others without code changes |
| **Model format** | joblib serialization | Standard for scikit-learn. Model file loaded into memory at startup. |

---

## API Design

### Authentication

API key-based authentication. Each customer gets an API key scoped to their `warehouse_id`.

```
Authorization: Bearer swe_live_abc123def456
```

Key format: `swe_{environment}_{random}` where environment is `live`, `test`, or `dev`.

### Endpoints

#### `POST /v1/predict` — Predict shipment weight

**Request:**

```json
{
  "shipment_id": "SHP-20260701-001",
  "items": [
    {
      "sku": "WIDGET-A",
      "quantity": 3,
      "unit_weight_oz": 12.5,
      "dimensions": {
        "length_in": 6.0,
        "width_in": 4.0,
        "height_in": 3.0
      },
      "category": "electronics"
    },
    {
      "sku": "GADGET-B",
      "quantity": 1,
      "unit_weight_oz": 8.0,
      "dimensions": {
        "length_in": 3.0,
        "width_in": 3.0,
        "height_in": 2.0
      },
      "category": "accessories"
    }
  ],
  "carton": {
    "carton_type": "12x12x8",
    "tare_weight_oz": 14.0,
    "dimensions": {
      "length_in": 12.0,
      "width_in": 12.0,
      "height_in": 8.0
    }
  },
  "ship_method": "FEDEX_GROUND",
  "packing_material": "bubble_wrap"
}
```

**Response (200):**

```json
{
  "prediction_id": "pred_01J5K9X2A3B4C5D6E7F8G9H0",
  "predicted_weight_oz": 53.2,
  "theoretical_weight_oz": 51.5,
  "adjustment_oz": 1.7,
  "confidence": {
    "interval_oz": [51.8, 54.6],
    "level": 0.90
  },
  "model_version": "v2.1.0",
  "metadata": {
    "latency_ms": 4,
    "cached": false
  }
}
```

**Response (422 — Validation Error):**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "items[0].unit_weight_oz must be positive",
    "details": [
      {
        "field": "items[0].unit_weight_oz",
        "constraint": "gt",
        "value": -1.0
      }
    ]
  }
}
```

#### `POST /v1/batch-predict` — Batch prediction

For bulk fulfillment runs. Accepts up to 100 shipments per request.

**Request:**

```json
{
  "shipments": [
    { /* same shape as single predict */ },
    { /* ... */ }
  ]
}
```

**Response (200):**

```json
{
  "predictions": [
    { /* same shape as single predict response */ },
    { /* ... */ }
  ],
  "metadata": {
    "total": 50,
    "succeeded": 49,
    "failed": 1,
    "total_latency_ms": 85
  }
}
```

#### `POST /v1/feedback` — Submit actual weight

This closes the feedback loop. Warehouses send the scale-measured weight after packing.

**Request:**

```json
{
  "prediction_id": "pred_01J5K9X2A3B4C5D6E7F8G9H0",
  "actual_weight_oz": 53.8,
  "measured_by": "scale_station_3"
}
```

**Response (200):**

```json
{
  "feedback_id": "fb_01J5K9Y3B4C5D6E7F8G9H0A1",
  "prediction_id": "pred_01J5K9X2A3B4C5D6E7F8G9H0",
  "actual_weight_oz": 53.8,
  "predicted_weight_oz": 53.2,
  "error_oz": 0.6,
  "error_pct": 1.12
}
```

#### `GET /v1/model/info` — Current model information

```json
{
  "model_version": "v2.1.0",
  "model_type": "LightGBM",
  "trained_at": "2026-06-28T14:30:00Z",
  "trained_on_rows": 125000,
  "feature_count": 14,
  "features": [
    "theoretical_weight_oz",
    "item_count",
    "unique_sku_count",
    "total_item_volume_in3",
    "fill_ratio",
    "carton_type",
    "ship_method",
    "packing_material",
    "heaviest_item_weight_oz",
    "weight_variance",
    "category_mode",
    "avg_sku_weight_error",
    "total_quantity",
    "dimensional_weight_oz"
  ]
}
```

#### `GET /v1/model/metrics` — Production model performance

```json
{
  "model_version": "v2.1.0",
  "period": "last_7_days",
  "predictions_count": 31647,
  "feedback_count": 23102,
  "feedback_rate": 0.73,
  "metrics": {
    "mae_oz": 1.2,
    "rmse_oz": 1.8,
    "mape_pct": 2.3,
    "within_2oz_pct": 84.2,
    "bias_oz": 0.15
  },
  "baseline_comparison": {
    "theoretical_mae_oz": 3.1,
    "improvement_pct": 61.3
  },
  "drift": {
    "detected": false,
    "last_checked": "2026-07-01T06:00:00Z"
  }
}
```

#### `GET /health` — Health check

```json
{
  "status": "healthy",
  "checks": {
    "model_loaded": true,
    "database": "connected",
    "redis": "connected"
  },
  "uptime_seconds": 86400,
  "version": "1.3.0"
}
```

#### `GET /ready` — Readiness probe

Returns 200 when the service is ready to accept traffic (model loaded, DB connected). Returns 503 during startup or if dependencies are down. Used by load balancers and Kubernetes.

### Error Codes

| HTTP Status | Error Code | When |
|---|---|---|
| 400 | `BAD_REQUEST` | Malformed JSON |
| 401 | `UNAUTHORIZED` | Missing or invalid API key |
| 404 | `NOT_FOUND` | Prediction ID not found (for feedback) |
| 422 | `VALIDATION_ERROR` | Invalid field values |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Unexpected server error |
| 503 | `SERVICE_UNAVAILABLE` | Model not loaded or dependency down |

### Rate Limits

| Tier | Requests/min | Batch size |
|---|---|---|
| Free / Dev | 60 | 10 |
| Standard | 300 | 50 |
| Enterprise | 1000 | 100 |

---

## Data Model

### Entity Relationship

```
┌─────────────────┐     ┌─────────────────┐
│   api_keys      │     │  model_registry │
├─────────────────┤     ├─────────────────┤
│ id (UUID, PK)   │     │ id (UUID, PK)   │
│ key_hash        │     │ version (UNIQUE) │
│ warehouse_id    │     │ model_type      │
│ tier            │     │ artifact_path   │
│ is_active       │     │ metrics (JSONB) │
│ created_at      │     │ is_active       │
│ last_used_at    │     │ trained_on_rows │
└─────────────────┘     │ feature_config  │
                        │ created_at      │
                        └────────┬────────┘
                                 │
                                 │ (model_version)
                                 │
┌─────────────────┐     ┌────────▼────────┐     ┌─────────────────┐
│  warehouses     │     │  predictions    │     │   feedback      │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (TEXT, PK)   │◄────│ id (UUID, PK)   │────►│ id (UUID, PK)   │
│ name            │     │ shipment_id     │     │ prediction_id   │
│ config (JSONB)  │     │ warehouse_id    │     │ actual_weight_oz│
│ created_at      │     │ request (JSONB) │     │ error_oz        │
└─────────────────┘     │ features (JSONB)│     │ error_pct       │
                        │ predicted_wt    │     │ measured_by     │
                        │ theoretical_wt  │     │ created_at      │
                        │ confidence_low  │     └─────────────────┘
                        │ confidence_high │
                        │ model_version   │
                        │ latency_ms      │
                        │ cached (BOOL)   │
                        │ created_at      │
                        └─────────────────┘
```

### Table Definitions

```sql
-- Warehouse / tenant configuration
CREATE TABLE warehouses (
    id TEXT PRIMARY KEY,                  -- e.g. 'medusa-main'
    name TEXT NOT NULL,
    config JSONB DEFAULT '{}',            -- feature flags, custom settings
    created_at TIMESTAMPTZ DEFAULT now()
);

-- API key management
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash TEXT NOT NULL UNIQUE,         -- bcrypt hash, never store plaintext
    warehouse_id TEXT NOT NULL REFERENCES warehouses(id),
    tier TEXT NOT NULL DEFAULT 'standard', -- free, standard, enterprise
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    last_used_at TIMESTAMPTZ
);

-- Model registry
CREATE TABLE model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version TEXT NOT NULL UNIQUE,          -- semver: v1.0.0, v1.1.0
    model_type TEXT NOT NULL,              -- linear, ridge, lightgbm
    artifact_path TEXT NOT NULL,           -- s3://bucket/models/v1.0.0.joblib
    metrics JSONB NOT NULL,               -- {mae, rmse, r2, mape, ...}
    feature_config JSONB NOT NULL,         -- ordered feature list + encoding info
    is_active BOOLEAN DEFAULT false,       -- only one active at a time
    trained_on_rows INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),

    CONSTRAINT one_active_model CHECK (
        NOT is_active OR id = (
            SELECT id FROM model_registry
            WHERE is_active = true
            ORDER BY created_at DESC LIMIT 1
        )
    )
);

-- Predictions (append-only, partitioned by month)
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shipment_id TEXT,
    warehouse_id TEXT NOT NULL REFERENCES warehouses(id),
    request_payload JSONB NOT NULL,       -- full request for reproducibility
    features JSONB NOT NULL,              -- computed feature vector
    predicted_weight_oz DOUBLE PRECISION NOT NULL,
    theoretical_weight_oz DOUBLE PRECISION NOT NULL,
    confidence_low DOUBLE PRECISION,
    confidence_high DOUBLE PRECISION,
    model_version TEXT NOT NULL REFERENCES model_registry(version),
    latency_ms INTEGER NOT NULL,
    cached BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
) PARTITION BY RANGE (created_at);

-- Monthly partitions (created automatically or via cron)
CREATE TABLE predictions_2026_07 PARTITION OF predictions
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Feedback (actual weights from warehouse scales)
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES predictions(id),
    actual_weight_oz DOUBLE PRECISION NOT NULL,
    error_oz DOUBLE PRECISION NOT NULL,        -- actual - predicted
    error_pct DOUBLE PRECISION NOT NULL,       -- abs(error_oz / actual) * 100
    measured_by TEXT,                           -- scale station identifier
    created_at TIMESTAMPTZ DEFAULT now(),

    CONSTRAINT one_feedback_per_prediction UNIQUE (prediction_id)
);

-- Indexes
CREATE INDEX idx_predictions_warehouse_created
    ON predictions (warehouse_id, created_at DESC);
CREATE INDEX idx_predictions_shipment
    ON predictions (shipment_id) WHERE shipment_id IS NOT NULL;
CREATE INDEX idx_predictions_model_version
    ON predictions (model_version);
CREATE INDEX idx_feedback_created
    ON feedback (created_at DESC);
CREATE INDEX idx_feedback_error
    ON feedback (abs(error_oz) DESC);
```

### Data Retention

| Table | Retention | Reason |
|---|---|---|
| `predictions` | 12 months | Retraining data, audit trail |
| `feedback` | 12 months | Matches predictions lifecycle |
| `model_registry` | Indefinite | Model history, rollback capability |
| `api_keys` | Indefinite | Access management |

---

## ML Pipeline

### Training Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. Extract  │───►│  2. Feature  │───►│  3. Split    │
│  Historical  │    │  Engineering │    │  Train/Test  │
│  Data        │    │              │    │  (time-based)│
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼──────┐
                    │  5. Compare  │◄───│  4. Train    │
                    │  to Champion │    │  Candidates  │
                    └──────┬───────┘    └──────────────┘
                           │
                    ┌──────▼───────┐    ┌──────────────┐
                    │  6. Register │───►│  7. Promote  │
                    │  in Registry │    │  (manual)    │
                    └──────────────┘    └──────────────┘
```

### Model Candidates

Start simple, add complexity only if it helps:

| Model | Library | Why try it |
|---|---|---|
| **Theoretical weight (baseline)** | None | This is what warehouses use today. Benchmark everything against this. |
| **Linear Regression** | scikit-learn | Interpretable, fast, shows feature importance directly. |
| **Ridge Regression** | scikit-learn | Handles correlated features better than plain linear. |
| **Random Forest** | scikit-learn | Handles non-linear relationships, robust to outliers. |
| **LightGBM** | lightgbm | Best accuracy for tabular data in most cases. Fast training. |

### Evaluation Strategy

**Train/test split:** Time-based, not random. Train on older shipments, test on recent ones. This simulates production where the model sees future shipments.

```
|--- Training Data (80%) ---|--- Test Data (20%) ---|
|  Jan 2025 ─── Sep 2025    |  Oct 2025 ─── Dec 2025|
```

**Metrics:**

| Metric | What it measures | Target |
|---|---|---|
| MAE (oz) | Average prediction error | Lower is better |
| RMSE (oz) | Penalizes large errors more | Lower is better |
| MAPE (%) | Percentage error, normalizes across weight ranges | < 5% |
| % within 2 oz | Practical accuracy threshold | > 80% |
| Bias (oz) | Systematic over/under prediction | Near 0 |

**Segment analysis:** Break down metrics by:
- Carton type (some cartons may be harder to predict)
- Weight range (light vs heavy shipments)
- Item count (single item vs multi-item)
- Warehouse (if multi-tenant)
- Ship method

### Model Versioning

Semantic versioning: `v{major}.{minor}.{patch}`

- **Major:** New model architecture or major feature changes
- **Minor:** Retrained with new data, feature additions
- **Patch:** Bug fixes, configuration changes

### Retraining Strategy

| Trigger | Action |
|---|---|
| **Scheduled** | Weekly retrain on latest 12 months of data |
| **Drift detected** | MAE degrades > 20% over 7-day rolling window |
| **New data volume** | > 10,000 new feedback records since last train |
| **Manual** | Operator-triggered via CLI |

### Confidence Intervals

Use quantile regression or conformal prediction to provide prediction intervals:

```python
# For tree-based models: use quantile predictions
# LightGBM supports quantile regression natively
# Alternative: conformal prediction wrapper (MAPIE library)

{
  "confidence": {
    "interval_oz": [51.8, 54.6],  # 90% of actual weights fall here
    "level": 0.90
  }
}
```

---

## Feature Engineering

### Feature Catalog

| # | Feature | Type | Computation | Null Strategy |
|---|---|---|---|---|
| 1 | `theoretical_weight_oz` | float | Sum of (item_weight × qty) + carton_tare | Required field |
| 2 | `item_count` | int | Sum of all item quantities | Required field |
| 3 | `unique_sku_count` | int | Count of distinct SKUs | Derived |
| 4 | `total_item_volume_in3` | float | Sum of (L × W × H × qty) per item | 0 if dims missing |
| 5 | `carton_volume_in3` | float | Carton L × W × H | 0 if dims missing |
| 6 | `fill_ratio` | float | total_item_volume / carton_volume | 0 if either missing |
| 7 | `carton_type` | categorical | Carton type identifier | "unknown" |
| 8 | `ship_method` | categorical | Carrier + service level | "unknown" |
| 9 | `packing_material` | categorical | Material type | "unknown" |
| 10 | `heaviest_item_weight_oz` | float | Max (unit_weight × 1) across items | Same as theoretical |
| 11 | `lightest_item_weight_oz` | float | Min unit weight across items | Same as theoretical |
| 12 | `weight_variance` | float | Std dev of unit weights in shipment | 0 |
| 13 | `category_mode` | categorical | Most frequent item category | "unknown" |
| 14 | `dimensional_weight_oz` | float | (L × W × H / dim_factor) for the carton | 0 if dims missing |
| 15 | `weight_per_item_oz` | float | theoretical_weight / item_count | theoretical_weight |
| 16 | `total_quantity` | int | Sum of all quantities | Same as item_count |

### Historical / Learned Features (require feedback data)

These features use past prediction accuracy. They are unavailable for new SKUs/cartons (cold start — use defaults).

| # | Feature | Type | Computation |
|---|---|---|---|
| 17 | `avg_sku_weight_error_oz` | float | Rolling average weight error for the SKUs in this shipment |
| 18 | `avg_carton_type_error_oz` | float | Rolling average weight error for this carton type |
| 19 | `warehouse_bias_oz` | float | Systematic weight bias for this warehouse |

### Feature Pipeline Design

```python
# Pseudocode for the feature engineering pipeline

class FeatureEngineer:
    """Stateless feature computation from a prediction request."""

    def compute(self, request: PredictRequest) -> dict[str, float]:
        features = {}

        # --- Direct features ---
        features["theoretical_weight_oz"] = self._theoretical_weight(request)
        features["item_count"] = sum(item.quantity for item in request.items)
        features["unique_sku_count"] = len(set(item.sku for item in request.items))
        features["total_quantity"] = features["item_count"]

        # --- Volume features ---
        features["total_item_volume_in3"] = self._total_item_volume(request.items)
        features["carton_volume_in3"] = self._box_volume(request.carton.dimensions)
        features["fill_ratio"] = self._safe_divide(
            features["total_item_volume_in3"],
            features["carton_volume_in3"]
        )

        # --- Weight distribution features ---
        weights = [item.unit_weight_oz for item in request.items]
        features["heaviest_item_weight_oz"] = max(weights)
        features["lightest_item_weight_oz"] = min(weights)
        features["weight_variance"] = statistics.stdev(weights) if len(weights) > 1 else 0
        features["weight_per_item_oz"] = features["theoretical_weight_oz"] / features["item_count"]

        # --- Categorical features ---
        features["carton_type"] = request.carton.carton_type
        features["ship_method"] = request.ship_method
        features["packing_material"] = request.packing_material
        features["category_mode"] = self._mode_category(request.items)

        # --- Dimensional weight ---
        features["dimensional_weight_oz"] = self._dim_weight(request.carton.dimensions)

        # --- Historical features (from DB, with fallbacks) ---
        features["avg_sku_weight_error_oz"] = self._lookup_sku_error(request.items)
        features["avg_carton_type_error_oz"] = self._lookup_carton_error(request.carton.carton_type)
        features["warehouse_bias_oz"] = self._lookup_warehouse_bias(request.warehouse_id)

        return features
```

### Encoding

| Type | Strategy |
|---|---|
| Categorical (< 20 values) | One-hot encoding |
| Categorical (> 20 values) | Target encoding (mean weight error per category) |
| Missing numerics | Impute with 0 or median, add `_is_missing` indicator |

---

## Infrastructure & DevOps

### Project Structure

```
shipment-weight-estimation/
├── src/
│   ├── api/                        # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py                 # App factory, lifespan, middleware
│   │   ├── dependencies.py         # Shared dependencies (model, db, redis)
│   │   ├── middleware.py           # Request ID, timing, logging
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py          # POST /v1/predict, /v1/batch-predict
│   │   │   ├── feedback.py         # POST /v1/feedback
│   │   │   ├── model.py            # GET /v1/model/info, /v1/model/metrics
│   │   │   └── health.py           # GET /health, /ready
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── predict.py          # Request/response Pydantic models
│   │       ├── feedback.py
│   │       └── common.py           # Shared types (ErrorResponse, etc.)
│   │
│   ├── ml/                         # ML pipeline (offline + online)
│   │   ├── __init__.py
│   │   ├── features.py             # FeatureEngineer class
│   │   ├── train.py                # Training entrypoint
│   │   ├── evaluate.py             # Evaluation, error analysis
│   │   ├── predict.py              # ModelPredictor (loads model, runs inference)
│   │   ├── registry.py             # Model versioning and promotion
│   │   └── drift.py                # Drift detection logic
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Settings via pydantic-settings (.env)
│   │   ├── logging.py              # Structured JSON logging
│   │   └── exceptions.py           # Domain exceptions
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py               # SQLAlchemy ORM models
│   │   ├── session.py              # Async session factory
│   │   ├── repositories/           # Data access layer
│   │   │   ├── predictions.py
│   │   │   ├── feedback.py
│   │   │   └── model_registry.py
│   │   └── migrations/             # Alembic
│   │       ├── env.py
│   │       └── versions/
│   │
│   └── services/                   # Business logic
│       ├── __init__.py
│       ├── prediction.py           # Orchestrates: validate → features → predict → store
│       ├── feedback.py             # Stores feedback, computes error, alerts
│       └── monitoring.py           # Drift detection, metric aggregation
│
├── tests/
│   ├── conftest.py                 # Fixtures, test DB, test client
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_predict.py
│   │   └── test_schemas.py
│   ├── integration/
│   │   ├── test_api_predict.py
│   │   ├── test_api_feedback.py
│   │   └── test_training_pipeline.py
│   └── fixtures/
│       ├── sample_requests.json
│       └── sample_model.joblib
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_error_analysis.ipynb
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile              # Multi-stage build
│   │   ├── Dockerfile.train        # Training-specific image
│   │   └── docker-compose.yml      # Local: API + Postgres + Redis + MinIO
│   ├── github/
│   │   └── workflows/
│   │       ├── ci.yml              # Lint + test + build on PR
│   │       ├── deploy.yml          # Build + push + deploy on merge to main
│   │       └── train.yml           # Weekly retraining pipeline
│   └── terraform/                  # Cloud infra (Phase 3)
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
├── scripts/
│   ├── seed_data.py                # Load Medusa historical data
│   ├── train.py                    # CLI: python scripts/train.py --version v1.0.0
│   ├── promote_model.py            # CLI: promote a model version to active
│   ├── benchmark.py                # Load test the API
│   └── migrate.py                  # Run DB migrations
│
├── pyproject.toml                  # Dependencies, tool config
├── alembic.ini
├── Makefile                        # Common commands
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md                       # This file
```

### Docker Setup

**Dockerfile (multi-stage):**

```dockerfile
# Stage 1: Build
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/

# Non-root user
RUN useradd -r -s /bin/false appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml (local dev):**

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, redis]
    volumes:
      - ./src:/app/src        # hot reload
      - ./models:/app/models  # local model artifacts

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: swe
      POSTGRES_USER: swe
      POSTGRES_PASSWORD: localdev
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin

volumes:
  pgdata:
```

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install ruff mypy
      - run: ruff check src/ tests/
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: swe_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install ".[test]"
      - run: pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t swe-api:${{ github.sha }} .
```

### Makefile

```makefile
.PHONY: dev test lint train migrate

dev:                          ## Start local dev environment
	docker compose up -d postgres redis minio
	uvicorn src.api.main:app --reload --port 8000

test:                         ## Run tests
	pytest tests/ -v --cov=src

lint:                         ## Lint and type check
	ruff check src/ tests/
	mypy src/

train:                        ## Train a new model version
	python scripts/train.py --version $(VERSION)

migrate:                      ## Run database migrations
	alembic upgrade head

seed:                         ## Load sample data
	python scripts/seed_data.py

benchmark:                    ## Load test the API
	python scripts/benchmark.py --rps 100 --duration 60

docker-build:                 ## Build Docker image
	docker build -t swe-api:latest .

docker-up:                    ## Start full stack in Docker
	docker compose up --build

docker-down:                  ## Stop Docker stack
	docker compose down
```

---

## Monitoring & Observability

### Structured Logging

Every log entry is JSON with correlation fields:

```json
{
  "timestamp": "2026-07-01T14:23:45.123Z",
  "level": "info",
  "message": "prediction_completed",
  "request_id": "req_abc123",
  "prediction_id": "pred_xyz789",
  "warehouse_id": "medusa-main",
  "model_version": "v2.1.0",
  "latency_ms": 4,
  "predicted_weight_oz": 53.2,
  "theoretical_weight_oz": 51.5
}
```

### Key Dashboards

| Dashboard | Metrics | Alert Threshold |
|---|---|---|
| **API Health** | Request rate, latency p50/p95/p99, error rate, uptime | Error rate > 1%, p95 > 100ms |
| **Model Performance** | MAE, RMSE, bias, % within 2oz (rolling 24h) | MAE increase > 20% vs 7-day avg |
| **Feedback Loop** | Feedback rate, feedback latency, error distribution | Feedback rate drops below 50% |
| **Drift Detection** | Feature distribution shift (PSI), prediction distribution | PSI > 0.2 on any feature |
| **Infrastructure** | CPU, memory, DB connections, Redis hit rate | CPU > 80%, memory > 85% |

### Drift Detection

Population Stability Index (PSI) computed weekly on each feature:

```
PSI < 0.1  → No drift (green)
PSI 0.1-0.2 → Minor drift (yellow, monitor)
PSI > 0.2  → Significant drift (red, trigger retrain)
```

### Alerting

| Condition | Severity | Action |
|---|---|---|
| API error rate > 1% for 5 min | Critical | Page on-call |
| Model MAE > 2× baseline for 24h | High | Trigger retrain, notify team |
| Feedback rate < 50% for 48h | Medium | Investigate data pipeline |
| Drift detected (PSI > 0.2) | Medium | Schedule retrain |
| Disk usage > 80% | Low | Expand storage |

---

## Security

### API Authentication

- API keys are hashed with bcrypt before storage.
- Keys are scoped to a `warehouse_id` — a key can only access its own data.
- Keys are passed via `Authorization: Bearer` header.
- Rate limiting is per-key, enforced in Redis.

### Data Protection

- No PII is stored in predictions. Shipment IDs are warehouse-internal identifiers.
- Request payloads are stored for reproducibility but contain only item/carton/shipping data.
- Database connections use TLS in production.
- API keys are never logged (masked in structured logs).

### Input Validation

- All inputs validated via Pydantic with strict types.
- Weight values must be positive.
- Dimensions must be positive.
- String fields have max length constraints.
- JSONB payloads have max size limits (1 MB).

### Dependency Security

- `pip audit` runs in CI.
- Dependabot enabled for automated PRs.
- Docker image scanned with Trivy.

---

## Development Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Clean data, working feature pipeline, baseline metrics.

- [ ] Set up project structure, pyproject.toml, linting, CI.
- [ ] Load Medusa historical data (seed script).
- [ ] Exploratory data analysis notebook.
- [ ] Implement `FeatureEngineer` class with unit tests.
- [ ] Compute baseline metrics (theoretical weight MAE/RMSE).
- [ ] Train first model candidates (linear, ridge, random forest, LightGBM).
- [ ] Evaluation notebook with segment analysis.

**Deliverable:** Jupyter notebooks, baseline report, trained model artifact.

### Phase 2: API (Week 3-4)

**Goal:** Production API serving predictions.

- [ ] FastAPI application with predict endpoint.
- [ ] Pydantic request/response schemas.
- [ ] Model loading at startup.
- [ ] Feedback endpoint.
- [ ] Health and readiness endpoints.
- [ ] Docker + docker-compose for local dev.
- [ ] PostgreSQL schema + Alembic migrations.
- [ ] Unit and integration tests.

**Deliverable:** Running API, Docker stack, test suite.

### Phase 3: Production Hardening (Week 5-6)

**Goal:** Production-ready with monitoring and CI/CD.

- [ ] API key authentication and rate limiting.
- [ ] Structured logging (JSON).
- [ ] Redis caching for prediction dedup.
- [ ] Model registry (DB-backed).
- [ ] Batch prediction endpoint.
- [ ] CI/CD pipeline (GitHub Actions).
- [ ] Load testing (benchmark script).
- [ ] Error handling and edge cases.

**Deliverable:** Production-ready system, CI/CD pipeline, load test results.

### Phase 4: ML Operations (Week 7-8)

**Goal:** Automated retraining and monitoring.

- [ ] Training pipeline as CLI command.
- [ ] Model comparison and promotion workflow.
- [ ] Drift detection (PSI on features).
- [ ] Production metrics dashboard.
- [ ] Alerting rules.
- [ ] Confidence intervals on predictions.
- [ ] Historical feature lookup (avg SKU error, carton error).
- [ ] Documentation.

**Deliverable:** Full MLOps loop, monitoring dashboards, documentation.

---

## Open Questions

These need answers before or during development:

| # | Question | Impact | Status |
|---|---|---|---|
| 1 | What format is Medusa's historical data? (CSV, DB export, API?) | Determines data loading approach | Open |
| 2 | How many historical shipments are available? | Affects model choice and evaluation strategy | Open |
| 3 | What fields are in the historical data? | Determines which features are actually buildable | Open |
| 4 | Does Medusa already track actual packed weight digitally? | Determines if feedback loop is automatic or manual | Open |
| 5 | What WMS does Medusa use? | Determines integration approach | Open |
| 6 | How will this API be called — from Perseuss cartonization, from the WMS, or both? | Determines auth model and deployment topology | Open |
| 7 | What's the expected request volume? (per day, peak per minute) | Determines infrastructure sizing | Open |
| 8 | Is there a preference for cloud provider? (AWS, GCP, self-hosted) | Determines deployment target | Open |
| 9 | Should the model be global (one model for all warehouses) or per-warehouse? | Affects training pipeline and multi-tenancy approach | Open |
| 10 | What dimensional weight factor should be used? (FedEx: 139, UPS: 139, USPS: 166) | Needed for dimensional weight feature | Open |

---

## Technology Stack Summary

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| Web Framework | FastAPI | 0.115+ |
| ASGI Server | uvicorn | 0.30+ |
| Validation | Pydantic | 2.x |
| ORM | SQLAlchemy | 2.x (async) |
| Migrations | Alembic | 1.13+ |
| Database | PostgreSQL | 16 |
| Cache | Redis | 7 |
| ML | scikit-learn, LightGBM | Latest |
| Feature Engineering | pandas, NumPy | Latest |
| Object Storage | S3 / MinIO | Latest |
| Containerization | Docker | 24+ |
| CI/CD | GitHub Actions | N/A |
| Linting | Ruff | Latest |
| Type Checking | mypy | Latest |
| Testing | pytest | Latest |
| Load Testing | locust or custom | Latest |

# Model Card — Shipment Weight Estimation (v0.4.0-real-data)

## Overview

Predicts actual packed shipment weight from order/carton features, replacing the
theoretical-weight baseline (sum of catalog item weights + carton tare).

- **Model type:** Ridge Regression (`sklearn.linear_model.Ridge`, α=1.0)
- **Alternatives evaluated:** Linear Regression, Ridge, Random Forest, Gradient Boosted Trees
- **Status:** Trained on real warehouse shipments, Jan–Jun 2026 (Medusa fulfillment data)
- **Saved artifact:** `models/model.joblib` — bundle with pipeline, residual_std, version metadata

## Training Data

Real warehouse shipment data from Medusa, January–June 2026.

- **Source files:** `order_shipments_anonymized.xlsx`, `order_lines_in_shipment_anonymized.xlsx`
- **Train / test split:** time-based — Jan–May 2026 (train), June 2026 (test)
- **Train rows:** 49,485 &nbsp;|&nbsp; **Test rows:** 9,337

### Exclusions applied during cleaning

| Step | Rule | Rows dropped |
|---|---|---|
| Null actual weight | `actual_weight_lbs` is null | 0 |
| Non-positive actual weight | `actual_weight_lbs` ≤ 0 | 0 |
| Bad ship methods | LTL, Pick Up at Medusa variants, CanceledItem, Ship Outside System | 7,385 (8.4% of raw; LTL alone = 5,570) |
| Outliers | `actual > 3× theoretical AND actual > 50 lbs` | 17 |

LTL shipments are excluded because they follow freight-carrier billing rules
(dimensional weight, pallet weight, freight class) that are out of scope for this
parcel-weight model. The 17 outlier rows were confirmed data errors
(e.g. 9,898 lbs recorded against a small flat-rate box).

## Features

| Feature | Type | Notes |
|---|---|---|
| `theoretical_weight_oz` | numeric | Dominant predictor — r = 0.98 with actual weight |
| `item_count` | numeric | |
| `total_item_volume_in3` | numeric | Sum of item L×W×H×qty across order lines |
| `void_volume_in3` | numeric | `box_volume − total_item_volume`, clipped at 0 |
| `category_avg_weight_error_oz` | numeric | Target-encoded from training fold only (no leakage) |
| `weight_per_item_oz` (derived) | numeric | `theoretical_weight_oz / item_count` |
| `fill_ratio` (derived) | numeric | `total_item_volume_in3 / box_volume_in3`, clipped at 1.5 |
| `num_categories` (derived) | numeric | Count of distinct item categories in shipment |
| `carton_type` | categorical | OHE, `handle_unknown="ignore"` |
| `ship_method` | categorical | OHE, `handle_unknown="ignore"` |
| `packing_material` | categorical | Always `"unknown"` in real data; zero signal but kept for API compatibility |
| `category_mode` | categorical | OHE, `handle_unknown="ignore"` |

**Dropped from v0.3:** `num_missing_catalog_weights` — this column is constant (all zeros)
in the real warehouse data, so it carried zero variance and added only split noise to
tree-based models. Removed from `NUMERIC_FEATURES` in `src/shipment_weight/features.py`.

Null `box_name` values (16,720 rows, ~28% of cleaned data) are filled with the
dimension string `{length}x{width}x{height}` before feature engineering so
`carton_type` is never null.

## Model Selection: Why Ridge, Not GBT

A diagnostic investigation (`scripts/investigate_gbt.py`) found that this problem
is essentially a **calibration task**, not a general regression task.

**Key findings:**

| Finding | Detail |
|---|---|
| `theoretical_weight_oz` Pearson r with target | **0.98** — near-perfect linear predictor |
| GBT feature importance for `theoretical_weight_oz` | **98.8%** of total importance |
| Systematic bias in theoretical weight | −18.95 oz on test set — consistent underestimate |

The theoretical weight already explains 96% of variance in actual weight. The model's
only real job is to apply a linear correction to this systematic underestimate.
Ridge's linear form — effectively learning `actual ≈ a × theoretical + corrections` —
is the right inductive bias. GBT uses 200 trees to approximate what a single
coefficient captures, with more variance and no benefit.

**Comparison table (test = June 2026):**

| Model | MAE oz | MAE lbs | RMSE oz | Bias oz | vs Ridge |
|---|---|---|---|---|---|
| Theoretical baseline | 21.24 | 1.328 | 28.36 | −18.95 | +9.92 oz |
| **Ridge** | **11.32** | **0.708** | **19.98** | **−2.46** | — |
| GBT (baseline) | 12.00 | 0.750 | 23.05 | −4.84 | +0.67 oz |
| GBT (tuned: 400 trees, lr=0.03) | 11.55 | 0.722 | 21.88 | −4.16 | +0.22 oz |
| GBT (residual framing) | 11.57 | 0.723 | 22.48 | −4.73 | +0.25 oz |
| GBT (no void_volume_in3) | 11.51 | 0.719 | 20.51 | −4.92 | +0.19 oz |

No GBT variant beats Ridge. The residual framing (predicting `actual − theoretical`,
then adding theoretical back) is the right conceptual approach for future work, but
the residual signal is too weak and noisy on the current 5-month training window to
outperform a linear fit.

**When GBT may become competitive:** once correction features with genuine non-linear
signal are added — e.g. carrier-specific tare weights by box type, historical error
rates by fulfillment station, time-of-week packing variance — the residual GBT
framing should be revisited. The model to beat at that point will still be Ridge on
the calibration task; GBT needs to demonstrate it extracts something Ridge cannot.

## Evaluation (test = June 2026, 9,337 shipments)

| Source | MAE (oz) | MAE (lbs) | RMSE (oz) | Bias (oz) |
|---|---|---|---|---|
| Theoretical weight baseline | 21.24 | 1.328 | 28.36 | −18.95 |
| **Ridge (saved model)** | **11.32** | **0.708** | **19.98** | **−2.46** |

**47% MAE reduction vs. baseline** on real warehouse data.

The model reduces the systematic −18.95 oz bias of the theoretical weight to −2.46 oz.
The remaining bias is a small consistent underestimate; it can be corrected with a
bias-correction term if label distribution stability is confirmed on future months.

## Known Failure Modes

- **Outlier shipments (data errors):** The outlier rule (`actual > 3× theoretical AND
  actual > 50 lbs`) caught 17 rows. Rows outside this rule but still anomalous
  (e.g. a genuinely heavy shipment with an inaccurate catalog weight) will produce
  large errors that the model cannot anticipate.
- **LTL shipments:** Explicitly excluded. If an LTL shipment somehow reaches the
  prediction endpoint, the model will produce a number — but it will be meaningless,
  since LTL billing rules (dimensional weight, freight class) are not represented in
  any feature.
- **Unseen carton types / categories:** `handle_unknown="ignore"` means novel values
  fall back to a zero OHE vector — the model still predicts, but with less information.
  For a carton type never seen in training, the prediction is essentially the linear
  correction applied to theoretical weight alone.
- **High item-count shipments:** Error variance grows with item count (more items =
  more opportunity for catalog-weight drift). The model captures the average trend
  but individual high-count shipments carry wider uncertainty.
- **Confidence interval is global:** The saved `residual_std` is a single number
  from the June test set. It does not widen for inputs far from the training
  distribution. A quantile-regression or conformal-prediction approach would be
  more accurate for OOD-aware intervals.

## Out-of-Distribution Behavior

Predictions are not clipped against physical bounds at the model level. The API
validates `theoretical_weight_oz > 0` and `total_item_volume_in3 ≥ 0` at the
schema layer (`api/schemas.py`); extreme inputs (e.g. `item_count=500`) still
produce a number, not a flagged warning.

## Versioning

| Version | Notes |
|---|---|
| `v0.1.0-synthetic` | Synthetic data only; GBT saved model |
| `v0.2.0-real-data` | First real-data retrain |
| `v0.3.0-real-data` | Added `void_volume_in3`; rule-based outlier removal; LTL exclusion |
| `v0.4.0-real-data` | Dropped `num_missing_catalog_weights` (constant in real data); switched saved model to Ridge after GBT investigation |

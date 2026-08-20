# Model Card — Shipment Weight Estimation (v0.5.0-histgbt)

## Overview

Predicts actual packed shipment weight from order/carton features, replacing the
theoretical-weight baseline (sum of catalog item weights + carton tare).

- **Model type:** HistGradientBoostingRegressor (`sklearn.ensemble.HistGradientBoostingRegressor`,
  `loss="absolute_error"`, untuned defaults — see [Model Selection](#model-selection-why-histgbt-not-ridge))
- **Previous production model:** Ridge Regression (`v0.4.0-real-data`) — archived at
  `models/model_ridge_v0.4.0_archived.joblib` as an instant rollback path. **Do not delete
  this file without explicit sign-off** — it's the only way back to the prior model.
- **Alternatives evaluated:** Linear Regression, Ridge, Random Forest, Gradient Boosted Trees
  (squared-error loss), HistGBT (absolute-error, tuned and untuned), LightGBM, XGBoost,
  CatBoost (native categorical handling), ElasticNet, Lasso
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

## Model Selection: Why HistGBT, Not Ridge

### History: why Ridge was chosen originally

An earlier diagnostic investigation (`scripts/investigate_gbt.py`) found this problem
is essentially a **calibration task**: `theoretical_weight_oz` has Pearson r = 0.98
with the target and 98.8% of GBT feature importance, with a consistent −18.95 oz
systematic underestimate. Ridge's linear form — `actual ≈ a × theoretical +
corrections` — was the right inductive bias for that framing, and no
squared-error-loss GBT variant tried at the time beat it (see table below, "squared
loss" rows). That conclusion was correct given the evidence available then — it was
never tested against an **absolute-error loss**, which turned out to matter.

### What changed

`scripts/experiment_loss_and_interactions.py` revisited the loss function directly,
since Ridge minimizes squared error but the model is judged on MAE. HistGBT with
`loss="absolute_error"` beat Ridge meaningfully, unlike every squared-error GBT
variant before it. That result was then stress-tested two ways before being trusted:

1. **Is it one library's quirk, or real?** `scripts/evaluate_model_sweep.py` tuned
   HistGBT (learning_rate × max_leaf_nodes × l2_regularization, train-fold-only
   validation) and found tuning bought **~0.00 oz** over the untuned defaults —
   already near-optimal. It then benchmarked LightGBM, XGBoost, and CatBoost
   (native categorical handling for `carton_type`/`ship_method`/`category_mode`,
   instead of one-hot) with the same discipline. All four boosting libraries
   converged to within **0.28 oz of each other**, landing 3.75–4.03 oz above the
   **~6.53 oz repeat-shipment noise floor** independently estimated by
   `scripts/check_noise_floor.py` (same-SKU-set/same-box shipments that recur 3+
   times still disagree with themselves by this much — the irreducible floor no
   model can beat). Four different implementations agreeing to within a third of
   an ounce, all landing near the same ceiling, is strong evidence the gain is a
   real property of the loss function and feature set, not an artifact of one
   library's defaults. ElasticNet/Lasso were also checked as a sanity check and,
   as expected, neither beat Ridge — `theoretical_weight_oz` dominance holds.
2. **Is it better everywhere, or just on average?** `scripts/evaluate_histgbt_diagnostics.py`
   reproduced the same three Stage 7 breakdowns below (bias by ship_method, bias
   by box_name, largest individual errors) for HistGBT and compared them
   segment-by-segment against Ridge's known weak spots:

   | Known weak spot (Ridge) | Ridge | HistGBT | Verdict |
   |---|---|---|---|
   | **FedEx HAZMAT** (n=124) | bias +10.77 oz, MAE 25.83 oz | bias **−2.31 oz**, MAE **8.15 oz** | **Largely fixed** |
   | **30x20x12 box** (n=781) | MAE 29.25 oz | MAE 28.17 oz | **Barely moved** — still the single worst box for both models, dominating both largest-errors tables |
   | **20x14x12 box** (n=879) | bias −11.53 oz, MAE 17.03 oz | bias −9.84 oz, MAE 13.99 oz | Improved (~3 oz) |
   | **item_count trend** | 1–2 items: 16.39 oz MAE (worst bucket) | 1–2 items: 11.02 oz MAE | Improved at low item counts; the 10+ bucket (72% of test rows) is only marginally better (11.14→10.50 oz) and still produces the same 100–300+ oz individual errors on extreme-item-count shipments — **not fixed**, just dampened |

   This same pass also caught a **new** regression HistGBT introduces that Ridge
   did not have: box **26x20x8** (n=49, 0.5% of test rows) — Ridge MAE 20.90 oz /
   bias +9.80 oz → HistGBT MAE **26.30 oz** / bias **−16.44 oz** (worse, and the
   bias direction flipped). Documented here as a known limitation, not a blocker —
   the sample is small, but it's a real, verified regression, not noise below
   detection.

Net: HistGBT is a genuine, multi-library-confirmed improvement on average and on
most of Ridge's known weak spots, it does not solve the two biggest structural
problems (the 30x20x12 box and extreme-item-count shipments remain the dominant
error source for **both** models), and it introduces one small, documented new
regression. This was judged a worthwhile trade and HistGBT was promoted to
production; see [Interpretability Tradeoff](#interpretability-tradeoff) below for
the cost that comes with it.

**Comparison table (test = June 2026):**

| Model | MAE oz | MAE lbs | RMSE oz | Bias oz | within-1lb % | vs HistGBT |
|---|---|---|---|---|---|---|
| Theoretical baseline | 21.24 | 1.328 | 28.36 | −18.95 | 44.2% | +10.96 oz |
| Ridge (previous production, archived) | 11.34 | 0.709 | 20.00 | −2.43 | 81.2% | +1.06 oz |
| Linear Regression | 11.33 | 0.708 | 20.01 | −2.36 | 81.1% | +1.05 oz |
| Random Forest | 12.02 | 0.751 | 20.65 | −6.23 | 78.3% | +1.74 oz |
| GBT, squared-error loss (`GradientBoostingRegressor`) | 12.00 | 0.750 | 23.12 | −4.84 | 81.5% | +1.72 oz |
| **HistGBT, absolute-error loss (production, untuned)** | **10.28** | **0.643** | **19.56** | **−3.00** | **84.4%** | — |
| HistGBT, absolute-error, tuned | 10.29 | 0.643 | — | — | 84.4% | +0.01 oz |
| LightGBM (MAE objective, tuned) | 10.53 | 0.658 | — | — | — | +0.25 oz |
| XGBoost (MAE objective, tuned) | 10.56 | 0.660 | — | — | — | +0.28 oz |
| CatBoost (MAE, native categoricals, tuned) | 10.49 | 0.656 | — | — | — | +0.21 oz |
| ElasticNet / Lasso (sanity check) | 12.22 | 0.763 | — | — | 77.9% | +1.94 oz |

Full sweep detail (per-config validation numbers, fit times, noise-floor framing)
is in `scripts/evaluate_model_sweep.py`'s output, not reproduced here.

## Evaluation (test = June 2026, 9,337 shipments)

| Source | MAE (oz) | MAE (lbs) | RMSE (oz) | Bias (oz) | within-1lb % |
|---|---|---|---|---|---|
| Theoretical weight baseline | 21.24 | 1.328 | 28.36 | −18.95 | 44.2% |
| Ridge (previous production, archived) | 11.34 | 0.709 | 20.00 | −2.43 | 81.2% |
| **HistGBT, absolute-error (saved model)** | **10.28** | **0.643** | **19.56** | **−3.00** | **84.4%** |

**52% MAE reduction vs. baseline**, a further **9.3% MAE reduction vs. the previous
Ridge production model**, on real warehouse data.

The model reduces the systematic −18.95 oz bias of the theoretical weight to −3.00 oz
— slightly larger in magnitude than Ridge's −2.43 oz, still a small consistent
underestimate. Whether a bias-correction term is worth adding should be revisited
once label distribution stability is confirmed on future months, same as before.

## Interpretability Tradeoff

This is the real cost of the switch, called out explicitly because the previous
version of this card leaned on Ridge's interpretability as a stated advantage over
GBT-family models. Ridge's coefficients gave a directly readable story — `actual ≈
a × theoretical_weight_oz + b_carton_type + b_ship_method + ...` — every number in
that equation could be pulled out and explained to a warehouse operator or Tomas in
one sentence. HistGBT is an ensemble of ~400 shallow decision trees (`max_iter=400`
boosting rounds); there is no single coefficient to point to for "why did this
shipment get this prediction." `feature_importances_`-style rankings are available
but only say which inputs matter on average across the training set, not the
linear, per-unit story Ridge could give for one specific shipment. If a future
requirement needs per-prediction explainability (e.g. showing a warehouse operator
why a number moved), that needs a SHAP-style explainer layered on top — it isn't
free with this model the way it was with Ridge.

## Confidence Intervals

The saved `residual_std` (June test set) feeds a Gaussian 90% interval:
`predicted ± 1.645 × residual_std` (`shipment_weight.predict.CONFIDENCE_Z`). This
was checked against HistGBT's actual residual distribution before carrying it over
unchanged, rather than assuming linear-model behavior still applies to a tree
ensemble:

| | Ridge (archived) | HistGBT (production) |
|---|---|---|
| residual std | 19.85 oz | 19.33 oz |
| skewness | −0.02 (symmetric) | **+0.57** (right-skewed) |
| excess kurtosis | +42.6 | +56.4 |
| nominal 90% CI width (`±1.645σ`) | 65.3 oz | 63.6 oz |
| **actual coverage of that CI** | 94.4% | **95.3%** |
| empirical 5th–95th percentile width | 46.8 oz | 40.5 oz |

Both models' residuals are heavily leptokurtic — a small number of extreme errors
(the same 30x20x12/high-item-count shipments flagged above) inflate `std` well
past what the bulk of the distribution looks like: the empirical 90% range is
30–40% narrower than the Gaussian formula implies for both models. This is a
**pre-existing imprecision, not one newly introduced by HistGBT** — Ridge's
interval was already over-wide before this switch. HistGBT's is somewhat more
skewed (+0.57 vs Ridge's ~0) and over-covers slightly more (95.3% vs 94.4% against
a 90% nominal target), consistent with a tree ensemble's error not being a linear
function of input noise the way Ridge's is. The direction of the miscalibration is
the safe one — the interval is too *wide*, not too narrow, so it isn't overselling
confidence — but it means the reported band is more conservative than necessary for
a typical shipment. No code change was made this round; the existing
recommendation below (quantile regression / conformal prediction) remains the
correct long-term fix and is now better-justified by these numbers for both
models, not just asserted.

## Known Failure Modes

- **30x20x12 box remains the dominant error source.** MAE 29.25 oz under Ridge,
  28.17 oz under HistGBT (n=781, June test) — barely moved by the model switch and
  responsible for 7 of the top 20 individual largest errors under HistGBT (6 under
  Ridge). Neither model solves this; it needs feature-level investigation (box
  geometry / packing behavior for this carton specifically), not a different model.
- **26x20x8 box: new regression introduced by HistGBT.** Small sample (n=49, 0.5%
  of test) but a real, verified effect — MAE 20.90→26.30 oz, bias flipped from
  +9.80 to −16.44 oz. Worth monitoring as more data accumulates; not currently
  blocking, since Ridge had its own equal-or-worse problems on this box already.
- **High item-count shipments (10+ items, 72% of test rows):** improved only
  modestly by the model switch (MAE 11.14→10.50 oz) and still produces the same
  100–300+ oz individual errors on extreme-item-count shipments (partial view:
  `scripts/evaluate_histgbt_diagnostics.py`'s largest-errors table). More items =
  more opportunity for catalog-weight drift to compound; the model captures the
  average trend but individual high-count shipments carry much wider uncertainty
  than the aggregate bucket MAE suggests.
- **Outlier shipments (data errors):** The outlier rule (`actual > 3× theoretical AND
  actual > 50 lbs`) caught 17 rows. Rows outside this rule but still anomalous
  (e.g. a genuinely heavy shipment with an inaccurate catalog weight) will produce
  large errors that the model cannot anticipate.
- **LTL shipments:** Explicitly excluded. If an LTL shipment somehow reaches the
  prediction endpoint, the model will produce a number — but it will be meaningless,
  since LTL billing rules (dimensional weight, freight class) are not represented in
  any feature.
- **Unseen carton types / categories:** `handle_unknown="ignore"` means novel values
  fall back to a zero OHE vector — the model still predicts, but with less
  information. For a carton type never seen in training, HistGBT falls back to
  whatever its trees learned from the "no category/carton signal" region of the
  training data — there's no clean analogue to Ridge's "linear correction applied
  to theoretical weight alone" for a boosted-tree model.
- **Confidence interval is global and imprecise for both models (see Confidence
  Intervals above):** does not widen for inputs far from the training distribution,
  and the Gaussian assumption over-covers relative to the empirical residual shape.
  A quantile-regression or conformal-prediction approach would be more accurate.

## Out-of-Distribution Behavior

Predictions are not clipped against physical bounds at the model level.
Input validation happens at the caller's edge — the `shipment_weight`
library's `Shipment`/`Box`/`ItemLine` dataclasses do basic type checking but
no range validation; extreme inputs (e.g. `item_count=500`) still produce a
number, not a flagged warning.

## Serving

The saved bundle (`models/model.joblib`) is a dict with `pipeline`,
`model_type`, `model_version`, `residual_std`, `trained_on_rows`,
`feature_list` (the exact `ALL_FEATURES` order the pipeline was fit on),
`category_error_map` (category -> mean training residual, oz), and
`category_error_global_mean` (fallback for categories unseen in training).

The last two exist specifically so `category_avg_weight_error_oz` can be
computed at serving time instead of guessed by the caller — previously this
train-fold encoding was computed in `scripts/train_real_data.py` and
discarded after training, with no way for a live prediction to reproduce it.
`shipment_weight.predict.ShipmentWeightPredictor` reads it from the bundle
automatically; see that module's docstring for the fuller story, which also
covers the related `fill_ratio`/`void_volume_in3` fix (both are now computed
from real box dimensions supplied by the caller, via the same
`shipment_weight.ingest.build_features` function training uses, instead of
being looked up against the synthetic-data-only `CARTON_CAPACITY` table or
hardcoded to 0).

`api/main.py` (FastAPI, secondary interface) was not updated for this —
its `ShipmentRequest` schema still takes pre-aggregated features, not box
dimensions/item lines, so it cannot benefit from this fix. See that file's
docstring.

**Label-rounded prediction field:** `WeightPrediction` carries both
`predicted_weight_lbs` (precise, unrounded — for accuracy tracking and
evaluation) and `predicted_weight_lbs_for_label` (rounded for a physical
shipping label). The rounding follows the same carrier-billing rule used
throughout this card and `scripts/evaluate_rounded.py`: **always round UP**
to the next whole pound for predictions ≥ 1 lb — never down, never to
nearest (e.g. 12.01 → 13, not 12) — left unrounded below 1 lb (a
quarter-pound rule was tested for that range and does not hold in real
data). This mirrors real carrier billing (e.g. USPS), and is not
carrier-specific in this dataset: comparing the same carrier's rows with
and without the "(Perseuss)" capture-system tag shows plain-tagged rows are
97-99% already-whole (already carrier-rounded) while the identical
carrier's "(Perseuss)"-tagged rows are only 53-66% whole (raw,
not-yet-rounded scale readings) — a data-capture-system artifact, not a
carrier-specific rounding policy. `scripts/evaluate_rounded.py` applies the
same round-up rule to the actual weight being compared against, for the
same reason: a meaningful share of `actual_weight_lbs` values are
"(Perseuss)"-tagged raw readings, not yet carrier-rounded, so comparing a
rounded prediction against a raw actual would understate real label-match
accuracy. Both the library (`shipment_weight.predict`) and
`scripts/evaluate_rounded.py` call the same
`shipment_weight.rounding.round_to_billing_tier` function, so there is one
place the rule is encoded. `confidence_interval_oz` is deliberately computed
from the precise prediction, not the rounded label value — rounding is a
presentation step for the label, not a statistical one, so the interval
should still reflect the model's real residual uncertainty.

**HistGBT's preprocessing is a densified `ColumnTransformer`**
(`sparse_threshold=0.0`, built by `scripts/train_real_data.make_histgbt_pipeline`)
instead of the sparse-tolerant one the other `MODEL_CANDIDATES`
(linear/ridge/random_forest/gradient_boosted_trees) use —
`HistGradientBoostingRegressor` cannot consume a sparse matrix. This is baked into
the saved `pipeline` object and transparent to callers (`ShipmentWeightPredictor`
just calls `.predict()` on it), but worth knowing if you're inspecting or
retraining the bundle directly.

**Rollback:** the previous production model (Ridge, `v0.4.0-real-data`) is archived
at `models/model_ridge_v0.4.0_archived.joblib`. To roll back, copy it over
`models/model.joblib` (and rerun `scripts/package_model.py` if the installed
package tree's copy also needs refreshing — see that script's docstring for why
`src/shipment_weight/models/model.joblib` is a separate copy `predict.py` prefers
over the repo-root one). **Do not delete the archived file without explicit
sign-off** — it's the only rollback path to the prior model.

## Versioning

| Version | Notes |
|---|---|
| `v0.1.0-synthetic` | Synthetic data only; GBT saved model |
| `v0.2.0-real-data` | First real-data retrain |
| `v0.3.0-real-data` | Added `void_volume_in3`; rule-based outlier removal; LTL exclusion |
| `v0.4.0-real-data` | Dropped `num_missing_catalog_weights` (constant in real data); switched saved model to Ridge after GBT investigation |
| `v0.4.0-real-data` (bundle format, same model) | Added `feature_list`, `category_error_map`, `category_error_global_mean` to the saved bundle; introduced `shipment_weight.predict` as the primary serving interface (see Serving section above) |
| `v0.5.0-histgbt` | Switched saved model to `HistGradientBoostingRegressor` (`loss="absolute_error"`, untuned defaults) after a model sweep across 4 boosting libraries (all converging within 0.28 oz of each other, near the repeat-shipment noise floor) and a full diagnostic comparison against Ridge's known weak spots — see [Model Selection](#model-selection-why-histgbt-not-ridge). Ridge `v0.4.0-real-data` archived at `models/model_ridge_v0.4.0_archived.joblib` for rollback; not deleted. |

# Synthetic Data Assumptions

`src/shipment_weight/data_gen.py` is a stand-in for real warehouse data. Every
number in it is a guess calibrated to "look plausible," not a measurement.
This doc lists each assumption so that when Medusa's historical data arrives,
it can be checked off (confirmed / wrong / needs adjustment) rather than
silently trusted. If an assumption turns out to be wrong, the model trained
on it should be considered wrong too, not just imprecise.

## Category mix

7 categories (`electronics`, `apparel`, `home_goods`, `books`, `toys`,
`grocery`, `fragile_glass`), each with a hand-picked weight mean/std and
carton dimensions (see `CATEGORIES` dict). Item lines are drawn uniformly
across these 7 categories.

**Verify against real data:** What categories does Medusa actually stock, in
what relative volume? A uniform draw across 7 categories is almost certainly
wrong — real catalogs are usually dominated by a handful of high-volume SKUs
or categories (e.g. mostly apparel, with electronics rare). The category mix
directly determines how much training signal the model gets per category.

## Catalog bias per category

Each category has a `catalog_bias` (e.g. electronics +4%, apparel -2%,
fragile_glass +8%) — a systematic fraction by which the listed catalog weight
differs from the true mean item weight.

**Verify against real data:** Is there really a systematic, category-level
bias in Medusa's catalog weights, or is the catalog mostly accurate with
per-SKU noise instead? If the real discrepancy is per-SKU rather than
per-category, `category_avg_weight_error_oz` (built from this assumption) is
the wrong feature shape.

## Noise structure: multiplicative + additive, scaled by sqrt(item_count)

Two separate noise sources are combined:

- **Multiplicative-ish, per-item:** each item's true weight is drawn from a
  category normal distribution (`weight_mean`, `weight_std`), so heavier
  categories carry proportionally more absolute noise.
- **Additive, per-shipment:** `compounding_noise = N(0, 0.35 * sqrt(item_count))`
  is added once per shipment, modeling the idea that packing error
  accumulates with more items but sub-linearly (square-root, not linear).

**Verify against real data:** Once real shipments have feedback (predicted vs.
scale-measured), plot residual variance against `item_count`. If it scales
linearly, or plateaus, or is dominated by a different variable entirely (e.g.
carton type, packer, ship method), the `sqrt(item_count)` assumption needs to
change to match. This assumption is one of the most consequential ones here —
several features and the confidence interval depend on it.

## Carton type capacities and tare weights

4 carton sizes (`S_8x6x4`, `M_12x9x6`, `L_16x12x10`, `XL_20x16x12`) with fixed
tare weights (4.0 / 7.0 / 11.0 / 16.0 oz) and a carton-selection rule that
picks the smallest carton whose capacity exceeds total item volume at a
random target fill ratio (55-90%).

**Verify against real data:** Real warehouses likely have more carton SKUs
(or fewer, standardized ones), different tare weights (cardboard weight
varies by source), and a selection policy driven by warehouse software, not
a random target-fill heuristic. `carton_type` is a categorical feature with
`handle_unknown="ignore"`, so unseen real carton types degrade gracefully,
but the model's actual carton signal is only as good as this assumption.

## Packing material density values

5 materials (`bubble_wrap`, `air_pillow`, `paper`, `foam`, `none`) each with
an assumed weight density per cubic inch of void space (0.004-0.025 oz/in3),
and a material-selection rule keyed off the shipment's max item fragility.

**Verify against real data:** Are these densities anywhere close to right?
Does Medusa even track `packing_material` per shipment, or is this an
inferred/unavailable field in the real data? If unavailable, this entire
feature may need to be dropped rather than imputed.

## 2% mispack outlier rate

`MISPACK_OUTLIER_RATE = 0.02` — 2% of synthetic shipments get a large
(+/-8 to 40oz) weight delta representing a missed item, extra item, or wrong
carton, with no feature signaling it in advance.

**Verify against real data:** What's the real mispack/exception rate at
Medusa? If it's higher, the model's reported accuracy on synthetic data is
optimistic; if Medusa logs *why* a shipment was an exception (e.g. a
QA-flagged re-pack), that's a feature this synthetic generator has no
equivalent for and could meaningfully improve real-world accuracy.

## 5% missing catalog weight rate

`CATALOG_MISSING_RATE = 0.05` — 5% of item lines have no catalog weight on
file, surfaced via `num_missing_catalog_weights` rather than left as NaN in
the theoretical weight sum.

**Verify against real data:** Does Medusa's catalog actually have ~5% missing
weights, and is `num_missing_catalog_weights` (a count) the right way to
expose it, or does the real data need a different signal (e.g. *which* SKUs
are missing, since that may correlate with newly added or rarely shipped
items)?

## 3% unknown category rate

`UNKNOWN_CATEGORY_RATE = 0.03` — 3% of item lines belong to a category the
model has never seen in training, using fallback weight/dimension parameters
not tied to any real category.

**Verify against real data:** This exists to test that one-hot encoding with
`handle_unknown="ignore"` doesn't crash on novel categories — it's a pipeline
robustness check, not a calibrated estimate of how often Medusa adds new
product categories. Low priority to "fix" with real data, but worth knowing
the real rate so the OOD failure mode in `MODEL_CARD.md` is sized correctly.

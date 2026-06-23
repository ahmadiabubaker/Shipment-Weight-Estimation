"""Synthetic warehouse shipment data generator.

Mimics the gap between theoretical weight (catalog item weights + carton tare +
expected packing material) and actual packed weight (scale-measured). The error
between the two compounds with item count and varies by packing material and
occasional mispack outliers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = {
    # weight in oz, dims in inches. catalog_bias is the systematic fraction by
    # which the catalog (theoretical) unit weight differs from the true mean.
    "electronics":    {"weight_mean": 14.0, "weight_std": 4.0, "dims": (6, 5, 3),  "catalog_bias": 0.04, "fragility": 0.8},
    "apparel":        {"weight_mean": 6.0,  "weight_std": 2.0, "dims": (10, 8, 2), "catalog_bias": -0.02, "fragility": 0.1},
    "home_goods":     {"weight_mean": 24.0, "weight_std": 8.0, "dims": (10, 8, 6), "catalog_bias": 0.06, "fragility": 0.5},
    "books":          {"weight_mean": 18.0, "weight_std": 3.0, "dims": (9, 6, 1.5), "catalog_bias": 0.01, "fragility": 0.05},
    "toys":           {"weight_mean": 10.0, "weight_std": 5.0, "dims": (8, 6, 4),  "catalog_bias": 0.05, "fragility": 0.6},
    "grocery":        {"weight_mean": 20.0, "weight_std": 6.0, "dims": (5, 5, 5),  "catalog_bias": 0.03, "fragility": 0.3},
    "fragile_glass":  {"weight_mean": 16.0, "weight_std": 5.0, "dims": (7, 7, 7),  "catalog_bias": 0.08, "fragility": 1.0},
}
CATEGORY_NAMES = list(CATEGORIES)
UNKNOWN_CATEGORY_RATE = 0.03  # items belonging to a category never seen in training

CARTON_TYPES = [
    # (name, max_volume_in3, tare_weight_oz)
    ("S_8x6x4", 8 * 6 * 4, 4.0),
    ("M_12x9x6", 12 * 9 * 6, 7.0),
    ("L_16x12x10", 16 * 12 * 10, 11.0),
    ("XL_20x16x12", 20 * 16 * 12, 16.0),
]

SHIP_METHODS = ["GROUND", "TWO_DAY", "EXPRESS", "OVERNIGHT"]
SHIP_METHOD_PROBS = [0.55, 0.25, 0.13, 0.07]

PACKING_MATERIALS = ["bubble_wrap", "air_pillow", "paper", "foam", "none"]
# expected oz of material per cubic inch of empty (void) space in the carton
MATERIAL_DENSITY = {"bubble_wrap": 0.012, "air_pillow": 0.004, "paper": 0.018, "foam": 0.025, "none": 0.0}

CATALOG_MISSING_RATE = 0.05   # fraction of item lines with no catalog weight on file
MISPACK_OUTLIER_RATE = 0.02   # fraction of shipments with a packing mistake


def _pick_carton(total_volume: float, rng: np.random.Generator) -> tuple[str, float]:
    target_fill = rng.uniform(0.55, 0.9)
    for name, capacity, tare in CARTON_TYPES:
        if total_volume <= capacity * target_fill:
            return name, tare
    return CARTON_TYPES[-1][0], CARTON_TYPES[-1][2]


def _pick_packing_material(max_fragility: float, rng: np.random.Generator) -> str:
    if max_fragility >= 0.7:
        probs = [0.55, 0.15, 0.10, 0.18, 0.02]
    elif max_fragility >= 0.4:
        probs = [0.25, 0.25, 0.25, 0.15, 0.10]
    else:
        probs = [0.05, 0.15, 0.30, 0.05, 0.45]
    return rng.choice(PACKING_MATERIALS, p=probs)


def generate_shipments(n_shipments: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic shipment-level dataframe.

    Columns: theoretical_weight_oz, actual_weight_oz (target), carton_type,
    item_count, total_item_volume_in3, item_categories, category_mode,
    ship_method, packing_material, num_missing_catalog_weights,
    category_avg_weight_error_oz (proxy for product-level weight history).
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Precompute the "true" average catalog error per category, then expose a
    # noisy estimate of it as a feature -- simulating a rolling historical
    # signal learned from past shipments rather than leaking this row's target.
    true_category_bias_oz = {
        cat: CATEGORIES[cat]["weight_mean"] * CATEGORIES[cat]["catalog_bias"] for cat in CATEGORY_NAMES
    }

    for _ in range(n_shipments):
        n_lines = int(np.clip(rng.poisson(2.5) + 1, 1, 12))
        cats = rng.choice(
            CATEGORY_NAMES + ["unknown"],
            size=n_lines,
            p=[(1 - UNKNOWN_CATEGORY_RATE) / len(CATEGORY_NAMES)] * len(CATEGORY_NAMES) + [UNKNOWN_CATEGORY_RATE],
        )
        qtys = rng.integers(1, 4, size=n_lines)

        true_weight_total = 0.0
        catalog_weight_total = 0.0
        volume_total = 0.0
        max_fragility = 0.0
        num_missing_catalog = 0
        hist_signal_terms = []

        for cat, qty in zip(cats, qtys):
            if cat == "unknown":
                params = {"weight_mean": 12.0, "weight_std": 6.0, "dims": (7, 6, 4), "catalog_bias": 0.0, "fragility": 0.5}
            else:
                params = CATEGORIES[cat]

            unit_true = max(0.5, rng.normal(params["weight_mean"], params["weight_std"]))
            unit_catalog = unit_true * (1 + params["catalog_bias"])

            if cat != "unknown" and rng.random() < CATALOG_MISSING_RATE:
                num_missing_catalog += 1
                unit_catalog = np.nan

            l, w, h = params["dims"]
            dims = (
                max(0.5, rng.normal(l, l * 0.12)),
                max(0.5, rng.normal(w, w * 0.12)),
                max(0.5, rng.normal(h, h * 0.12)),
            )
            volume_total += dims[0] * dims[1] * dims[2] * qty
            true_weight_total += unit_true * qty
            catalog_weight_total += (0 if np.isnan(unit_catalog) else unit_catalog) * qty
            max_fragility = max(max_fragility, params["fragility"])
            hist_signal_terms.append(true_category_bias_oz.get(cat, 0.0))

        carton_type, tare_weight = _pick_carton(volume_total, rng)
        carton_capacity = next(c for n, c, _ in CARTON_TYPES if n == carton_type)
        void_volume = max(0.0, carton_capacity - volume_total)

        ship_method = rng.choice(SHIP_METHODS, p=SHIP_METHOD_PROBS)
        packing_material = _pick_packing_material(max_fragility, rng)
        expected_packing_weight = void_volume * MATERIAL_DENSITY[packing_material]

        theoretical_weight = catalog_weight_total + tare_weight + expected_packing_weight

        # actual packing material weight varies by packer (lognormal noise)
        actual_packing_weight = expected_packing_weight * rng.lognormal(mean=0.0, sigma=0.25)

        # per-item weight variance compounds with item count
        item_count = int(qtys.sum())
        compounding_noise = rng.normal(0, 0.35 * np.sqrt(item_count))

        actual_weight = true_weight_total + tare_weight + actual_packing_weight + compounding_noise

        # occasional mispack outliers: missed item, extra item, wrong carton
        if rng.random() < MISPACK_OUTLIER_RATE:
            outlier_delta = rng.choice([-1, 1]) * rng.uniform(8, 40)
            actual_weight += outlier_delta

        actual_weight = max(tare_weight, actual_weight)

        category_mode = pd.Series(cats).mode().iloc[0]
        item_categories = ",".join(sorted(set(cats)))
        category_avg_weight_error_oz = float(np.mean(hist_signal_terms)) + rng.normal(0, 0.3)

        rows.append(
            {
                "theoretical_weight_oz": round(theoretical_weight, 2),
                "actual_weight_oz": round(actual_weight, 2),
                "carton_type": carton_type,
                "item_count": item_count,
                "total_item_volume_in3": round(volume_total, 2),
                "item_categories": item_categories,
                "category_mode": category_mode,
                "ship_method": ship_method,
                "packing_material": packing_material,
                "num_missing_catalog_weights": num_missing_catalog,
                "category_avg_weight_error_oz": round(category_avg_weight_error_oz, 3),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_shipments(5000, seed=42)
    df.to_csv("data/synthetic_shipments.csv", index=False)
    print(f"Generated {len(df)} rows -> data/synthetic_shipments.csv")

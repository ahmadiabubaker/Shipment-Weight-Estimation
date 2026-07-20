"""Investigate why GBT underperforms Ridge and explore fixes.

Sections:
  A — Top-15 GBT feature importances (what is the model actually learning?)
  B — Pearson correlation of every numeric feature with actual_weight_oz on train set
  C — Six-way comparison table:
        theoretical_baseline | ridge | gbt_baseline | gbt_tuned
        | gbt_residual | gbt_no_void

Usage:
    python scripts/investigate_gbt.py \\
        --shipments order_shipments_anonymized.xlsx \\
        --lines     order_lines_in_shipment_anonymized.xlsx
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shipment_weight.evaluate import regression_metrics
from shipment_weight.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    add_derived_features,
    build_preprocessor,
)
from shipment_weight.train import make_pipeline

from train_real_data import (
    LBS_TO_OZ,
    _hr,
    build_features,
    clean,
    encode_category_error,
    load_data,
    time_split,
)

# ── Feature set without void_volume_in3 (ablation) ───────────────────────────
NUMERIC_NO_VOID = [f for f in NUMERIC_FEATURES if f != "void_volume_in3"]
ALL_FEATURES_NO_VOID = NUMERIC_NO_VOID + CATEGORICAL_FEATURES


def _build_pipeline_no_void(estimator) -> Pipeline:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("numeric", num_pipe, NUMERIC_NO_VOID),
        ("categorical", cat_pipe, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", pre), ("model", estimator)])


# ── Section A ─────────────────────────────────────────────────────────────────

def section_a(pipe: Pipeline, top_n: int = 15) -> None:
    _hr("A — GBT FEATURE IMPORTANCES  (baseline model, top 15)")
    preprocessor = pipe.named_steps["preprocess"]
    try:
        raw_names = preprocessor.get_feature_names_out()
        feat_names = [n.split("__", 1)[-1] for n in raw_names]
    except AttributeError:
        feat_names = list(NUMERIC_FEATURES)
        ohe = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        for feat, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
            feat_names += [f"{feat}_{c}" for c in cats]

    importances = pipe.named_steps["model"].feature_importances_
    df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df["cumulative_%"] = (df["importance"].cumsum() * 100).round(1)
    df["importance"] = df["importance"].map(lambda x: f"{x:.4f}")
    print(df.to_string(index=False))


# ── Section B ─────────────────────────────────────────────────────────────────

def section_b(train: pd.DataFrame) -> None:
    _hr("B — PEARSON CORRELATION WITH actual_weight_oz  (train set)")
    cols = NUMERIC_FEATURES + [TARGET]
    # use only columns present (guard against edge cases)
    cols = [c for c in cols if c in train.columns]
    corr = train[cols].corr()[TARGET].drop(TARGET)
    df = (
        pd.DataFrame({"feature": corr.index, "corr": corr.values, "abs_corr": corr.abs().values})
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    df["corr"] = df["corr"].map(lambda x: f"{x:+.4f}")
    df["abs_corr"] = df["abs_corr"].map(lambda x: f"{x:.4f}")
    print(df.to_string(index=False))


# ── Section C ─────────────────────────────────────────────────────────────────

def section_c(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    theoretical_train_oz: pd.Series, theoretical_test_oz: pd.Series,
) -> tuple[dict, Pipeline]:
    _hr("C — FITTING MODEL VARIANTS  (train = Jan–May, test = June)")

    results: dict[str, dict] = {}

    # Theoretical baseline
    results["theoretical_baseline"] = regression_metrics(y_test, theoretical_test_oz.values)

    # Ridge (benchmark)
    print("  ridge          ...", end=" ", flush=True)
    ridge_pipe = make_pipeline(Ridge(alpha=1.0))
    ridge_pipe.fit(X_train, y_train)
    results["ridge"] = regression_metrics(y_test, ridge_pipe.predict(X_test))
    print("done")

    # GBT baseline (matches current MODEL_CANDIDATES entry)
    print("  gbt_baseline   ...", end=" ", flush=True)
    gbt_base_pipe = make_pipeline(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        )
    )
    gbt_base_pipe.fit(X_train, y_train)
    results["gbt_baseline"] = regression_metrics(y_test, gbt_base_pipe.predict(X_test))
    print("done")

    # GBT tuned: more trees, shallower learning, subsampling for regularization
    print("  gbt_tuned      ...", end=" ", flush=True)
    gbt_tuned_pipe = make_pipeline(
        GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, random_state=42,
        )
    )
    gbt_tuned_pipe.fit(X_train, y_train)
    results["gbt_tuned"] = regression_metrics(y_test, gbt_tuned_pipe.predict(X_test))
    print("done")

    # GBT residual: predict (actual - theoretical), add theoretical back
    print("  gbt_residual   ...", end=" ", flush=True)
    y_resid_train = y_train.values - theoretical_train_oz.values
    gbt_resid_pipe = make_pipeline(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        )
    )
    gbt_resid_pipe.fit(X_train, y_resid_train)
    preds_resid = gbt_resid_pipe.predict(X_test) + theoretical_test_oz.values
    results["gbt_residual"] = regression_metrics(y_test, preds_resid)
    print("done")

    # GBT no-void: ablation — all features except void_volume_in3
    print("  gbt_no_void    ...", end=" ", flush=True)
    gbt_novoid_pipe = _build_pipeline_no_void(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        )
    )
    gbt_novoid_pipe.fit(X_train[ALL_FEATURES_NO_VOID], y_train)
    preds_novoid = gbt_novoid_pipe.predict(X_test[ALL_FEATURES_NO_VOID])
    results["gbt_no_void"] = regression_metrics(y_test, preds_novoid)
    print("done")

    return results, gbt_base_pipe


# ── Final table ───────────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    ridge_mae = results["ridge"]["mae_oz"]
    header = (
        f"  {'Model':<24} {'MAE oz':>8} {'MAE lbs':>8} "
        f"{'RMSE oz':>8} {'Bias oz':>9} {'≤0.5oz%':>9} {'vs ridge':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in results.items():
        delta = m["mae_oz"] - ridge_mae
        delta_str = f"{delta:+.2f}" if name != "ridge" else "  —"
        print(
            f"  {name:<24} {m['mae_oz']:>8.2f} {m['mae_oz']/LBS_TO_OZ:>8.3f} "
            f"{m['rmse_oz']:>8.2f} {m['bias_oz']:>9.2f} "
            f"{m['within_0_5oz_pct']:>8.1f}% {delta_str:>9}"
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GBT vs Ridge investigation.")
    parser.add_argument("--shipments", required=True)
    parser.add_argument("--lines", required=True)
    args = parser.parse_args()

    # Reproduce the exact same pipeline as train_real_data.py
    ships_raw, lines_raw = load_data(args.shipments, args.lines)
    ships, _ = clean(ships_raw)
    df = build_features(ships, lines_raw)
    train_df, test_df = time_split(df)
    train_df, test_df = encode_category_error(train_df, test_df)
    train_df = add_derived_features(train_df)
    test_df = add_derived_features(test_df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET]
    theoretical_train_oz = train_df["theoretical_weight_oz"]
    theoretical_test_oz = test_df["theoretical_weight_oz"]

    # B first — no model fitting needed
    section_b(train_df)

    # C — fit all variants; returns fitted baseline GBT for section A
    results, gbt_base_pipe = section_c(
        X_train, y_train, X_test, y_test,
        theoretical_train_oz, theoretical_test_oz,
    )

    # A — needs fitted GBT
    section_a(gbt_base_pipe)

    _hr("FINAL COMPARISON TABLE  (test = June 2026,  'vs ridge' = MAE delta in oz)")
    print_table(results)


if __name__ == "__main__":
    main()

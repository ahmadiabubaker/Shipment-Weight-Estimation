"""Retrain Ridge with the extended (order-lines-derived) feature set and
report before/after metrics, coefficients, and a leave-one-in ablation.

This script does NOT modify the production pipeline. It reuses
load/clean/build_features/time_split/encode_category_error unmodified
from scripts/train_real_data.py, and adds the new features from
shipment_weight.features_extended on top -- so the base model (trained the
same way, on the same cleaned/split rows) can be diffed against the
extended model within a single run.

Pipeline stages:
  1-2. Load + Clean        -- reused unmodified from train_real_data
  3.   Feats (base)        -- reused unmodified from train_real_data
  3b.  Feats (extended)    -- new: order-lines aggregates, target encodings
  4.   Split                -- reused unmodified (Jan-May train / June test)
  5.   Encode (category)   -- reused unmodified (train-fold only)
  5b.  Finalize extended    -- new: train-fold-only fallback/winsorize stats
  6.   Train + compare      -- base Ridge vs extended Ridge, same rows
  7.   Coefficients          -- Ridge coefficients for the new features
  8.   Ablation              -- leave-one-in: baseline + each new feature alone
  9.   Save                  -- models/model_extended.joblib (does NOT
                                 touch models/model.joblib)

Usage:
    python scripts/train_real_data_extended.py \\
        --shipments order_shipments_anonymized.xlsx \\
        --lines     order_lines_in_shipment_anonymized.xlsx
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shipment_weight.evaluate import regression_metrics
from shipment_weight.features import ALL_FEATURES, TARGET, add_derived_features, build_preprocessor
from shipment_weight.features_extended import (
    ALL_FEATURES_EXTENDED,
    CATEGORICAL_FEATURES_EXTENDED,
    EXTENDED_CATEGORICAL_FEATURES,
    EXTENDED_NUMERIC_FEATURES,
    NUMERIC_FEATURES_EXTENDED,
    add_extended_features,
    add_target_encodings,
    finalize_train_fold_stats,
    train_fold_only_encode,
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


def build_preprocessor_extended() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES_EXTENDED),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES_EXTENDED),
        ]
    )


def make_pipeline_extended() -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor_extended()), ("model", Ridge(alpha=1.0))])


def make_pipeline_leave_one_in(numeric_extra: list[str], categorical_extra: list[str]) -> tuple[Pipeline, list[str]]:
    from shipment_weight.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    numeric = NUMERIC_FEATURES + numeric_extra
    categorical = CATEGORICAL_FEATURES + categorical_extra
    feats = numeric + categorical
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    pre = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)]
    )
    return Pipeline(steps=[("preprocess", pre), ("model", Ridge(alpha=1.0))]), feats


def report_metrics_row(label: str, m: dict) -> str:
    return (
        f"  {label:<38} {m['mae_oz']:>8.2f} {m['mae_oz']/LBS_TO_OZ:>8.3f} "
        f"{m['rmse_oz']:>8.2f} {m['bias_oz']:>9.2f} {m['within_0_3lb_pct']:>8.1f}% "
        f"{m['within_0_5lb_pct']:>8.1f}% {m['within_1lb_pct']:>8.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain Ridge with the extended feature set.")
    parser.add_argument("--shipments", required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--out", default="models/model_extended.joblib")
    args = parser.parse_args()

    ships_raw, lines_raw = load_data(args.shipments, args.lines)
    ships, _ = clean(ships_raw)
    df = build_features(ships, lines_raw)
    # add_derived_features is idempotent and purely row-wise (no fold-dependent
    # stats), so it's safe to run once here, before the extended features (which
    # need num_categories) and before the time split.
    df = add_derived_features(df)

    _hr("STAGE 3b — EXTENDED FEATURE ENGINEERING")
    df = add_extended_features(df, lines_raw)
    df = add_target_encodings(df)
    print(f"  Extended feature dataframe shape: {df.shape}")

    train, test = time_split(df)
    train, test = encode_category_error(train, test)

    _hr("STAGE 5b — FINALIZE TRAIN-FOLD STATS (extended features)")
    train, test = finalize_train_fold_stats(train, test)
    print("  weight_density fallback median and fill_ratio p99 threshold computed from TRAIN only")

    X_train_base = train[ALL_FEATURES]
    X_test_base = test[ALL_FEATURES]
    X_train_ext = train[ALL_FEATURES_EXTENDED]
    X_test_ext = test[ALL_FEATURES_EXTENDED]
    y_train = train[TARGET]
    y_test = test[TARGET]
    theoretical_test_oz = test["theoretical_weight_oz"]

    _hr("STAGE 6 — BASE vs EXTENDED RIDGE  (test = June 2026)")
    header = (f"{'Model':<40} {'MAE oz':>8} {'MAE lbs':>8} {'RMSE oz':>8} {'Bias oz':>9} "
              f"{'<=0.3lb%':>9} {'<=0.5lb%':>9} {'<=1lb%':>9}")
    print(header)
    print("-" * len(header))

    base_metric = regression_metrics(y_test, theoretical_test_oz.values)
    print(report_metrics_row("theoretical_baseline", base_metric))

    base_pipe = make_pipeline(Ridge(alpha=1.0))
    base_pipe.fit(X_train_base, y_train)
    base_preds = base_pipe.predict(X_test_base)
    base_ridge_metric = regression_metrics(y_test, base_preds)
    print(report_metrics_row("ridge_BASE (current production features)", base_ridge_metric))

    ext_pipe = make_pipeline_extended()
    ext_pipe.fit(X_train_ext, y_train)
    ext_preds = ext_pipe.predict(X_test_ext)
    ext_ridge_metric = regression_metrics(y_test, ext_preds)
    print(report_metrics_row("ridge_EXTENDED (base + new features)", ext_ridge_metric))

    mae_delta_lbs = (ext_ridge_metric["mae_oz"] - base_ridge_metric["mae_oz"]) / LBS_TO_OZ
    print(f"\n  MAE delta (extended - base): {mae_delta_lbs:+.4f} lbs")
    print(
        f"  Within-1lb delta (extended - base): "
        f"{ext_ridge_metric['within_1lb_pct'] - base_ridge_metric['within_1lb_pct']:+.2f} pp"
    )

    _hr("STAGE 7 — RIDGE COEFFICIENTS (extended model, new features)")
    preprocessor = ext_pipe.named_steps["preprocess"]
    coefs = ext_pipe.named_steps["model"].coef_
    try:
        feat_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    except AttributeError:
        feat_names = list(NUMERIC_FEATURES_EXTENDED)
        ohe = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        for feat, cats in zip(CATEGORICAL_FEATURES_EXTENDED, ohe.categories_):
            feat_names += [f"{feat}_{c}" for c in cats]

    coef_df = pd.DataFrame({"feature": feat_names, "coef_oz": coefs})
    new_feature_prefixes = tuple(EXTENDED_NUMERIC_FEATURES) + tuple(EXTENDED_CATEGORICAL_FEATURES)
    is_new = coef_df["feature"].apply(lambda f: any(f == p or f.startswith(p + "_") for p in new_feature_prefixes))
    new_coefs = coef_df[is_new].copy()
    new_coefs["abs_coef_oz"] = new_coefs["coef_oz"].abs()
    new_coefs = new_coefs.sort_values("abs_coef_oz", ascending=False)
    print(new_coefs[["feature", "coef_oz"]].to_string(index=False))

    print("\n  All numeric feature coefficients (base + new), for context:")
    numeric_only = coef_df[coef_df["feature"].isin(NUMERIC_FEATURES_EXTENDED)].sort_values(
        "coef_oz", key=lambda s: s.abs(), ascending=False
    )
    print(numeric_only.to_string(index=False))

    _hr("STAGE 8 — ABLATION (leave-one-in: base features + ONE new feature)")
    ablation_rows = []
    for feat in EXTENDED_NUMERIC_FEATURES:
        pipe, _ = make_pipeline_leave_one_in([feat], [])
        pipe.fit(train[[*ALL_FEATURES, feat]], y_train)
        preds = pipe.predict(test[[*ALL_FEATURES, feat]])
        m = regression_metrics(y_test, preds)
        ablation_rows.append((feat, m["mae_oz"], m["mae_oz"] / LBS_TO_OZ, m["within_1lb_pct"]))
    for feat in EXTENDED_CATEGORICAL_FEATURES:
        pipe, _ = make_pipeline_leave_one_in([], [feat])
        pipe.fit(train[[*ALL_FEATURES, feat]], y_train)
        preds = pipe.predict(test[[*ALL_FEATURES, feat]])
        m = regression_metrics(y_test, preds)
        ablation_rows.append((feat, m["mae_oz"], m["mae_oz"] / LBS_TO_OZ, m["within_1lb_pct"]))

    ablation_df = pd.DataFrame(ablation_rows, columns=["new_feature", "mae_oz", "mae_lbs", "within_1lb_pct"])
    ablation_df["mae_delta_lbs_vs_base"] = ablation_df["mae_lbs"] - base_ridge_metric["mae_oz"] / LBS_TO_OZ
    ablation_df = ablation_df.sort_values("mae_delta_lbs_vs_base")
    print(f"  base_ridge MAE: {base_ridge_metric['mae_oz']/LBS_TO_OZ:.4f} lbs  (reference row, 0.0 delta)")
    print(ablation_df.to_string(index=False))

    print(f"\n  all_new_features_together (ridge_EXTENDED) MAE: {ext_ridge_metric['mae_oz']/LBS_TO_OZ:.4f} lbs "
          f"({mae_delta_lbs:+.4f} lbs vs base)")

    _hr("STAGE 8b — EXPANDING vs FROZEN (train-fold-only) TARGET ENCODING")
    print("  Isolates how much of the expanding encoder's gain is genuine forward-looking")
    print("  signal (available with a static end-of-May refresh) vs extra lift from rolling")
    print("  the encoding forward using early-June actuals to predict later-June rows.\n")
    train_resid_oz = train["actual_weight_oz"] - train["theoretical_weight_oz"]
    test_resid_oz = test["actual_weight_oz"] - test["theoretical_weight_oz"]

    frozen_rows = []
    for group_col, feat_name in [("carton_type", "box_name_target_enc_oz_FROZEN"),
                                  ("ship_method", "ship_method_target_enc_oz_FROZEN")]:
        train_frozen, test_frozen = train_fold_only_encode(
            train.assign(_resid=train_resid_oz), test.assign(_resid=test_resid_oz), group_col, "_resid"
        )
        train_aug = train[ALL_FEATURES].copy()
        train_aug[feat_name] = train_frozen.values
        test_aug = test[ALL_FEATURES].copy()
        test_aug[feat_name] = test_frozen.values

        pipe, feats = make_pipeline_leave_one_in([feat_name], [])
        # make_pipeline_leave_one_in built its ColumnTransformer against
        # NUMERIC_FEATURES + [feat_name]; fit/predict on the matching frame.
        pipe.fit(train_aug[feats], y_train)
        preds = pipe.predict(test_aug[feats])
        m = regression_metrics(y_test, preds)
        frozen_rows.append((feat_name, m["mae_oz"] / LBS_TO_OZ, m["within_1lb_pct"]))

    print(f"  {'feature':<32} {'mae_lbs':>9} {'delta_vs_base':>14}")
    base_mae_lbs = base_ridge_metric["mae_oz"] / LBS_TO_OZ
    for feat in ["box_name_target_enc_oz", "ship_method_target_enc_oz"]:
        row = ablation_df[ablation_df["new_feature"] == feat].iloc[0]
        print(f"  {feat + ' (expanding)':<32} {row['mae_lbs']:>9.4f} {row['mae_lbs']-base_mae_lbs:>+14.4f}")
    for feat_name, mae_lbs, within_1lb in frozen_rows:
        print(f"  {feat_name:<32} {mae_lbs:>9.4f} {mae_lbs-base_mae_lbs:>+14.4f}")

    _hr("STAGE 9 — SAVE EXTENDED MODEL ARTIFACT")
    residuals = y_test.values - ext_preds
    bundle = {
        "pipeline": ext_pipe,
        "model_type": "ridge_extended",
        "model_version": "v0.5.0-extended-experimental",
        "residual_std": float(residuals.std()),
        "trained_on_rows": len(X_train_ext),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"  Saved to: {args.out}  (production models/model.joblib untouched)")

    _hr("DONE")


if __name__ == "__main__":
    main()

"""Train and compare model candidates: linear -> ridge -> random forest -> GBT."""
from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from shipment_weight.data_gen import generate_shipments
from shipment_weight.features import ALL_FEATURES, TARGET, add_derived_features, build_preprocessor

MODEL_CANDIDATES = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "gradient_boosted_trees": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def load_or_generate(csv_path: str | None, n_shipments: int, seed: int) -> pd.DataFrame:
    if csv_path:
        return pd.read_csv(csv_path)
    return generate_shipments(n_shipments=n_shipments, seed=seed)


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    df = add_derived_features(df)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train_all_candidates(X_train, y_train) -> dict[str, Pipeline]:
    fitted = {}
    for name, estimator in MODEL_CANDIDATES.items():
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted


MODEL_VERSION = "v0.1.0-synthetic"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to a shipments CSV; generates synthetic data if omitted")
    parser.add_argument("--n-shipments", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/model.joblib")
    parser.add_argument("--model", default="gradient_boosted_trees", choices=list(MODEL_CANDIDATES))
    args = parser.parse_args()

    df = load_or_generate(args.csv, args.n_shipments, args.seed)
    X_train, X_test, y_train, y_test = split_data(df, seed=args.seed)

    pipe = make_pipeline(MODEL_CANDIDATES[args.model])
    pipe.fit(X_train, y_train)

    # Empirical residual std on held-out data, used to build a simple
    # Gaussian confidence band at serving time (90% interval).
    residuals = y_test.values - pipe.predict(X_test)
    residual_std = float(residuals.std())

    bundle = {
        "pipeline": pipe,
        "model_type": args.model,
        "model_version": MODEL_VERSION,
        "residual_std": residual_std,
        "trained_on_rows": len(X_train),
    }
    joblib.dump(bundle, args.out)
    print(f"Trained {args.model} on {len(X_train)} rows, residual_std={residual_std:.2f}, saved to {args.out}")


if __name__ == "__main__":
    main()

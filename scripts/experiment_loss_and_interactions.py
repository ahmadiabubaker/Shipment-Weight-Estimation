"""Two quick accuracy experiments on the existing feature set.

EXPERIMENT 3 — optimise the loss we are actually measured on.
    Ridge minimises squared error, but the reported metrics are MAE and
    within-1lb. Squared loss lets a handful of tail rows dominate the fit.
    Candidates: Huber, median (L1) regression, and HistGradientBoosting
    with loss="absolute_error" -- including the residual framing
    (predict actual-theoretical, add theoretical back). The earlier GBT
    investigation only ever used squared loss, so it does not cover this.

EXPERIMENT 4 — per-segment calibration slopes.
    The model card's own conclusion is that this is a calibration task:
    actual ~ a * theoretical + b. A single global `a` assumes every box
    type and ship method mis-estimates weight by the same proportion.
    carton_type is already one-hot encoded, so each box has its own
    INTERCEPT but shares one SLOPE. Adding theoretical x carton_type
    interaction columns gives each box its own slope too -- completing the
    per-box affine calibration while keeping the linear inductive bias
    that beat GBT.

Also included (cheap, from the same fixtures):
    - alpha tuning by chronological CV inside the train fold (alpha has
      been pinned at 1.0 and never tuned)
    - multiplicative framing: model log(actual/theoretical)

Read-only: nothing is written to models/.

Usage:
    python scripts/experiment_loss_and_interactions.py
    python scripts/experiment_loss_and_interactions.py --quantile   # add exact L1 LP (slow)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from shipment_weight.evaluate import regression_metrics
from shipment_weight.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from shipment_weight.ingest import apply_category_map
from shipment_weight.train import make_pipeline

from _experiment_prep import prepare
from train_real_data import LBS_TO_OZ, _hr, category_error_map

TOP_N_CARTONS = 25      # boxes given their own calibration slope
TOP_N_SHIP = 10


# ── Preprocessor over an arbitrary feature list ──────────────────────────────

def build_preprocessor_for(
    numeric: list[str], categorical: list[str], dense: bool = False
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        # HistGradientBoosting cannot consume a sparse matrix; sparse_threshold=0
        # forces the one-hot block to be densified. Harmless for the linear
        # models, which accept either, so only the tree models pay the cost.
        sparse_threshold=0.0 if dense else 0.3,
    )


def pipeline_for(estimator, numeric: list[str], categorical: list[str], dense: bool = False) -> Pipeline:
    return Pipeline(
        steps=[("preprocess", build_preprocessor_for(numeric, categorical, dense)), ("model", estimator)]
    )


# ── Experiment 4: interaction columns ────────────────────────────────────────

def add_interaction_columns(
    train: pd.DataFrame, test: pd.DataFrame, col: str, top_n: int, prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add ``theoretical_weight_oz * 1[col == level]`` columns for the top_n
    most frequent levels in TRAIN (levels chosen from the train fold only,
    so no test information influences the design matrix).

    Rows whose level is outside the top_n contribute to none of the new
    columns and therefore fall back to the shared global slope -- which is
    the desired behaviour for rare boxes where a bespoke slope would just
    fit noise.
    """
    levels = train[col].astype(str).value_counts().head(top_n).index.tolist()
    train = train.copy()
    test = test.copy()
    new_cols: list[str] = []
    for lvl in levels:
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(lvl))[:40]
        name = f"{prefix}_{safe}"
        train[name] = np.where(train[col].astype(str) == lvl, train["theoretical_weight_oz"], 0.0)
        test[name] = np.where(test[col].astype(str) == lvl, test["theoretical_weight_oz"], 0.0)
        new_cols.append(name)
    coverage = train[col].astype(str).isin(levels).mean() * 100
    print(f"    {col}: {len(levels)} levels given their own slope, covering {coverage:.1f}% of train rows")
    return train, test, new_cols


# ── Reporting ────────────────────────────────────────────────────────────────

class Reporter:
    def __init__(self, baseline_mae: float | None = None):
        self.rows: list[dict] = []
        self.baseline_mae = baseline_mae

    def add(self, name: str, y_true, y_pred, seconds: float = float("nan")) -> dict:
        m = regression_metrics(y_true, y_pred)
        if self.baseline_mae is None:
            self.baseline_mae = m["mae_oz"]
        row = {
            "model": name,
            "mae_oz": m["mae_oz"],
            "mae_lbs": m["mae_oz"] / LBS_TO_OZ,
            "rmse_oz": m["rmse_oz"],
            "bias_oz": m["bias_oz"],
            "within_0_3lb_pct": m["within_0_3lb_pct"],
            "within_0_5lb_pct": m["within_0_5lb_pct"],
            "within_1lb_pct": m["within_1lb_pct"],
            "delta_mae_oz": m["mae_oz"] - self.baseline_mae,
            "fit_s": seconds,
        }
        self.rows.append(row)
        self._print_row(row)
        return row

    @staticmethod
    def header() -> None:
        print(f"  {'model':<34} {'MAE oz':>8} {'MAE lbs':>8} {'RMSE oz':>8} {'bias oz':>8} "
              f"{'<=0.3lb':>8} {'<=0.5lb':>8} {'<=1lb':>7} {'dMAE oz':>8} {'fit s':>7}")
        print("  " + "-" * 110)

    @staticmethod
    def _print_row(r: dict) -> None:
        better = " *" if r["delta_mae_oz"] < -0.05 else ""
        print(f"  {r['model']:<34} {r['mae_oz']:>8.2f} {r['mae_lbs']:>8.3f} {r['rmse_oz']:>8.2f} "
              f"{r['bias_oz']:>+8.2f} {r['within_0_3lb_pct']:>7.1f}% {r['within_0_5lb_pct']:>7.1f}% "
              f"{r['within_1lb_pct']:>6.1f}% {r['delta_mae_oz']:>+8.2f} "
              f"{r['fit_s']:>7.1f}{better}")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).sort_values("mae_oz")


def fit_eval(pipe, X_tr, y_tr, X_te, scale_y: bool = False) -> tuple[np.ndarray, float]:
    """Fit, predict, and time it. With ``scale_y`` the target is standardised
    for the fit and the predictions are mapped back.

    Huber and SGD are solved by iterative optimisers whose conditioning
    depends on the scale of y, and the target here is ounces spanning
    0-16,000 -- lbfgs hits its iteration cap without converging. Both
    estimators are scale-equivariant (HuberRegressor jointly fits its own
    ``scale_``), so this yields the same model, just one the optimiser can
    actually reach.
    """
    y_arr = np.asarray(y_tr, dtype=float)
    t0 = time.time()
    if scale_y:
        mu, sd = y_arr.mean(), y_arr.std()
        pipe.fit(X_tr, (y_arr - mu) / sd)
        preds = pipe.predict(X_te) * sd + mu
    else:
        pipe.fit(X_tr, y_arr)
        preds = pipe.predict(X_te)
    return preds, time.time() - t0


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Loss-function and interaction experiments.")
    parser.add_argument("--shipments", default="order_shipments_anonymized.xlsx")
    parser.add_argument("--lines", default="order_lines_in_shipment_anonymized.xlsx")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--quantile", action="store_true",
                        help="also fit sklearn QuantileRegressor (exact L1 via LP; slow on 49k rows)")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    train, test = prepare(args.shipments, args.lines, refresh=args.refresh)

    X_train, y_train = train[ALL_FEATURES], train[TARGET]
    X_test, y_test = test[ALL_FEATURES], test[TARGET]
    theo_train = train["theoretical_weight_oz"].values
    theo_test = test["theoretical_weight_oz"].values

    rep = Reporter()

    _hr("BASELINES")
    Reporter.header()
    rep.add("theoretical_baseline", y_test, theo_test)
    baseline_row_idx = len(rep.rows)
    ridge_preds, secs = fit_eval(make_pipeline(Ridge(alpha=1.0)), X_train, y_train, X_test)
    ridge_row = rep.add("ridge_alpha1_PRODUCTION", y_test, ridge_preds, secs)
    # Everything after this is judged against production Ridge, not the
    # theoretical baseline, so reset the reference.
    rep.baseline_mae = ridge_row["mae_oz"]
    rep.rows[0]["delta_mae_oz"] = rep.rows[0]["mae_oz"] - ridge_row["mae_oz"]
    rep.rows[baseline_row_idx]["delta_mae_oz"] = 0.0

    # ── EXPERIMENT 3 ─────────────────────────────────────────────────────────
    _hr("EXPERIMENT 3 — LOSS FUNCTIONS ALIGNED TO MAE")
    print("  Ridge optimises squared error; we report MAE. These optimise absolute")
    print("  (or robust) loss instead, on the identical feature set and rows.\n")
    Reporter.header()

    # Huber: quadratic near zero, linear in the tails, so a 100 oz outlier
    # pulls the fit no harder than a 20 oz one.
    for eps, label in [(1.35, "huber_eps1.35"), (1.10, "huber_eps1.10_more_robust")]:
        preds, secs = fit_eval(
            make_pipeline(HuberRegressor(epsilon=eps, alpha=1e-4, max_iter=2000)),
            X_train, y_train, X_test, scale_y=True,
        )
        rep.add(label, y_test, preds, secs)

    # Median regression by SGD on epsilon-insensitive loss with epsilon=0,
    # which is exactly L1.
    preds, secs = fit_eval(
        make_pipeline(
            SGDRegressor(loss="epsilon_insensitive", epsilon=0.0, penalty="l2", alpha=1e-5,
                         max_iter=3000, tol=1e-5, learning_rate="adaptive", eta0=0.01,
                         random_state=42)
        ),
        X_train, y_train, X_test, scale_y=True,
    )
    rep.add("sgd_L1_median_regression", y_test, preds, secs)

    if args.quantile:
        from sklearn.linear_model import QuantileRegressor
        print("  (fitting exact QuantileRegressor -- this can take several minutes)")
        preds, secs = fit_eval(
            make_pipeline(QuantileRegressor(quantile=0.5, alpha=1e-4, solver="highs")),
            X_train, y_train, X_test, scale_y=True,
        )
        rep.add("quantile_median_exact_LP", y_test, preds, secs)

    # HistGBT with absolute_error loss -- direct and residual framings.
    hgb_kwargs = dict(loss="absolute_error", max_iter=400, learning_rate=0.05,
                      max_depth=None, min_samples_leaf=40, l2_regularization=1.0,
                      early_stopping=True, validation_fraction=0.1, random_state=42)
    preds, secs = fit_eval(
        pipeline_for(HistGradientBoostingRegressor(**hgb_kwargs),
                     NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True),
        X_train, y_train, X_test,
    )
    rep.add("histgbt_absolute_error", y_test, preds, secs)

    # Residual framing: predict (actual - theoretical), add theoretical back.
    pipe = pipeline_for(HistGradientBoostingRegressor(**hgb_kwargs),
                        NUMERIC_FEATURES, CATEGORICAL_FEATURES, dense=True)
    t0 = time.time()
    pipe.fit(X_train, y_train.values - theo_train)
    preds = pipe.predict(X_test) + theo_test
    rep.add("histgbt_abs_err_RESIDUAL", y_test, preds, time.time() - t0)

    # Ridge on the residual framing, for completeness. Should be near-identical
    # to plain Ridge because theoretical_weight_oz is itself a feature.
    pipe = make_pipeline(Ridge(alpha=1.0))
    t0 = time.time()
    pipe.fit(X_train, y_train.values - theo_train)
    preds = pipe.predict(X_test) + theo_test
    rep.add("ridge_RESIDUAL_framing", y_test, preds, time.time() - t0)

    # Multiplicative framing: log(actual / theoretical).
    valid = (theo_train > 0) & (y_train.values > 0)
    log_ratio = np.log(y_train.values[valid] / theo_train[valid])
    pipe = make_pipeline(Ridge(alpha=1.0))
    t0 = time.time()
    pipe.fit(X_train[valid], log_ratio)
    safe_theo_test = np.where(theo_test > 0, theo_test, np.nan)
    preds = safe_theo_test * np.exp(pipe.predict(X_test))
    preds = np.where(np.isfinite(preds), preds, ridge_preds)  # fall back where theoretical is 0
    rep.add("ridge_LOG_RATIO_multiplicative", y_test, preds, time.time() - t0)

    # ── EXPERIMENT 4 ─────────────────────────────────────────────────────────
    _hr("EXPERIMENT 4 — PER-SEGMENT CALIBRATION SLOPES (interactions)")
    print("  carton_type/ship_method already give per-segment INTERCEPTS via one-hot.")
    print("  These add theoretical_weight_oz x segment columns = per-segment SLOPES.\n")

    tr_i, te_i, box_cols = add_interaction_columns(
        train, test, "carton_type", TOP_N_CARTONS, "theo_x_box"
    )
    tr_i, te_i, ship_cols = add_interaction_columns(
        tr_i, te_i, "ship_method", TOP_N_SHIP, "theo_x_ship"
    )
    print()
    Reporter.header()

    for label, extra in [
        ("ridge_+box_slopes", box_cols),
        ("ridge_+ship_slopes", ship_cols),
        ("ridge_+box_and_ship_slopes", box_cols + ship_cols),
    ]:
        numeric = NUMERIC_FEATURES + extra
        preds, secs = fit_eval(
            pipeline_for(Ridge(alpha=1.0), numeric, CATEGORICAL_FEATURES),
            tr_i[numeric + CATEGORICAL_FEATURES], y_train, te_i[numeric + CATEGORICAL_FEATURES],
        )
        rep.add(label, y_test, preds, secs)

    # Interactions add many correlated columns, so the alpha that was fine for
    # 8 numeric features is probably too weak here -- try a stronger penalty.
    numeric = NUMERIC_FEATURES + box_cols + ship_cols
    for alpha in (10.0, 100.0):
        preds, secs = fit_eval(
            pipeline_for(Ridge(alpha=alpha), numeric, CATEGORICAL_FEATURES),
            tr_i[numeric + CATEGORICAL_FEATURES], y_train, te_i[numeric + CATEGORICAL_FEATURES],
        )
        rep.add(f"ridge_+slopes_alpha{alpha:g}", y_test, preds, secs)

    # Best-of-both: interactions + Huber loss.
    preds, secs = fit_eval(
        pipeline_for(HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000), numeric, CATEGORICAL_FEATURES),
        tr_i[numeric + CATEGORICAL_FEATURES], y_train, te_i[numeric + CATEGORICAL_FEATURES],
        scale_y=True,
    )
    rep.add("huber_+box_and_ship_slopes", y_test, preds, secs)

    # ── EXPERIMENT 5 (follows from the segmentation drift finding) ───────────
    _hr("EXPERIMENT 5 — RECENCY WEIGHTING / ROLLING WINDOW")
    print("  scripts/analyze_residuals.py found the raw (actual - theoretical) residual")
    print("  trending upward month over month: Jan +12.8 oz -> Jun +19.0 oz. A model")
    print("  fit with equal weight on all of Jan-May is therefore calibrated to a")
    print("  staler average than June. These down-weight or drop old rows.\n")
    Reporter.header()

    cutoff = pd.Timestamp(test["order_date"].min())
    age_days = (cutoff - pd.to_datetime(train["order_date"])).dt.days.values.astype(float)

    for half_life in (30.0, 60.0, 90.0):
        w = 0.5 ** (age_days / half_life)
        pipe = make_pipeline(Ridge(alpha=1.0))
        t0 = time.time()
        pipe.fit(X_train, y_train, model__sample_weight=w)
        preds = pipe.predict(X_test)
        rep.add(f"ridge_recency_halflife{half_life:g}d", y_test, preds, time.time() - t0)

    for months in (1, 2, 3):
        mask = age_days <= months * 30.5
        pipe = make_pipeline(Ridge(alpha=1.0))
        t0 = time.time()
        pipe.fit(X_train[mask], y_train[mask])
        preds = pipe.predict(X_test)
        rep.add(f"ridge_last{months}mo_only_n{mask.sum()}", y_test, preds, time.time() - t0)

    # Best-of-both: the strongest loss + slopes + recency together.
    numeric_all = NUMERIC_FEATURES + box_cols + ship_cols
    w = 0.5 ** (age_days / 60.0)
    pipe = pipeline_for(Ridge(alpha=10.0), numeric_all, CATEGORICAL_FEATURES)
    t0 = time.time()
    pipe.fit(tr_i[numeric_all + CATEGORICAL_FEATURES], y_train, model__sample_weight=w)
    preds = pipe.predict(te_i[numeric_all + CATEGORICAL_FEATURES])
    rep.add("ridge_slopes+recency60d", y_test, preds, time.time() - t0)

    pipe = pipeline_for(HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000),
                        numeric_all, CATEGORICAL_FEATURES)
    y_mu, y_sd = float(y_train.mean()), float(y_train.std())
    t0 = time.time()
    pipe.fit(tr_i[numeric_all + CATEGORICAL_FEATURES], (y_train.values - y_mu) / y_sd,
             model__sample_weight=w)
    preds = pipe.predict(te_i[numeric_all + CATEGORICAL_FEATURES]) * y_sd + y_mu
    rep.add("huber_slopes+recency60d_COMBINED", y_test, preds, time.time() - t0)

    # ── Bonus: alpha tuning by chronological CV ──────────────────────────────
    _hr("BONUS — RIDGE ALPHA BY CHRONOLOGICAL CV  (alpha has been pinned at 1.0)")
    print("  4 expanding-window folds inside Jan-May. The category encoding is")
    print("  recomputed inside each fold, so the tuning itself does not leak.\n")

    tr_sorted = train.sort_values("order_date").reset_index(drop=True)
    n = len(tr_sorted)
    bounds = [int(n * f) for f in (0.4, 0.55, 0.7, 0.85, 1.0)]
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    cv_scores: dict[float, list[float]] = {a: [] for a in alphas}

    for i in range(len(bounds) - 1):
        fold_tr = tr_sorted.iloc[: bounds[i]].copy()
        fold_va = tr_sorted.iloc[bounds[i] : bounds[i + 1]].copy()
        cmap, gmean = category_error_map(fold_tr)
        fold_tr["category_avg_weight_error_oz"] = apply_category_map(fold_tr["category_mode"], cmap, gmean)
        fold_va["category_avg_weight_error_oz"] = apply_category_map(fold_va["category_mode"], cmap, gmean)
        for a in alphas:
            p = make_pipeline(Ridge(alpha=a))
            p.fit(fold_tr[ALL_FEATURES], fold_tr[TARGET])
            mae = np.abs(p.predict(fold_va[ALL_FEATURES]) - fold_va[TARGET].values).mean()
            cv_scores[a].append(mae)

    print(f"  {'alpha':>10} {'mean CV MAE oz':>16} {'per-fold MAE oz':>40}")
    best_alpha, best_cv = None, float("inf")
    for a in alphas:
        mean_mae = float(np.mean(cv_scores[a]))
        folds = "  ".join(f"{s:.2f}" for s in cv_scores[a])
        marker = ""
        if mean_mae < best_cv:
            best_cv, best_alpha = mean_mae, a
            marker = " <- best"
        print(f"  {a:>10g} {mean_mae:>16.3f} {folds:>40}{marker}")

    print(f"\n  CV-selected alpha: {best_alpha:g}  (production uses 1.0)")
    print("  Held-out June performance at the CV-selected alpha:\n")
    Reporter.header()
    preds, secs = fit_eval(make_pipeline(Ridge(alpha=best_alpha)), X_train, y_train, X_test)
    rep.add(f"ridge_alpha{best_alpha:g}_CV_selected", y_test, preds, secs)

    # ── Summary ──────────────────────────────────────────────────────────────
    _hr("SUMMARY — all candidates ranked by June MAE")
    summary = rep.frame()
    print(summary.to_string(index=False, formatters={
        "mae_oz": "{:.2f}".format, "mae_lbs": "{:.3f}".format, "rmse_oz": "{:.2f}".format,
        "bias_oz": "{:+.2f}".format, "within_0_3lb_pct": "{:.1f}".format,
        "within_0_5lb_pct": "{:.1f}".format, "within_1lb_pct": "{:.1f}".format,
        "delta_mae_oz": "{:+.2f}".format, "fit_s": "{:.1f}".format,
    }))

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "experiment_loss_and_interactions.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\n  Written to {out_csv}")

    prod = summary[summary["model"] == "ridge_alpha1_PRODUCTION"].iloc[0]
    winners = summary[summary["delta_mae_oz"] < -0.05]
    winners = winners[winners["model"] != "ridge_alpha1_PRODUCTION"]
    print(f"\n  Production Ridge MAE: {prod['mae_oz']:.2f} oz ({prod['mae_lbs']:.3f} lbs)")
    if len(winners) == 0:
        print("  No candidate beat production Ridge by more than 0.05 oz.")
    else:
        print(f"  {len(winners)} candidate(s) beat it:")
        for _, r in winners.iterrows():
            print(f"    {r['model']:<34} {r['mae_oz']:.2f} oz  ({r['delta_mae_oz']:+.2f} oz, "
                  f"{r['within_1lb_pct']:.1f}% within 1 lb)")

    _hr("DONE")


if __name__ == "__main__":
    main()

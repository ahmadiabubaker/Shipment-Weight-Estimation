"""Evaluation utilities: error metrics, baseline comparison, segment bias, large errors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    bias = float(np.mean(y_pred - y_true))
    within_2oz = float(np.mean(np.abs(y_pred - y_true) <= 2.0)) * 100
    return {"mae_oz": mae, "rmse_oz": rmse, "bias_oz": bias, "within_2oz_pct": within_2oz}


def compare_to_baseline(y_true: pd.Series, y_pred: np.ndarray, theoretical: pd.Series) -> pd.DataFrame:
    model_metrics = regression_metrics(y_true, y_pred)
    baseline_metrics = regression_metrics(y_true, theoretical)
    return pd.DataFrame([{"source": "theoretical_weight_baseline", **baseline_metrics},
                          {"source": "model", **model_metrics}])


def bias_by_segment(y_true: pd.Series, y_pred: np.ndarray, segment: pd.Series, segment_name: str) -> pd.DataFrame:
    errors = pd.DataFrame({segment_name: segment.values, "error_oz": y_pred - y_true.values})
    return (
        errors.groupby(segment_name)["error_oz"]
        .agg(count="count", mean_bias_oz="mean", mae_oz=lambda s: s.abs().mean(), std_oz="std")
        .reset_index()
        .sort_values("count", ascending=False)
    )


def bias_by_item_count_bucket(y_true: pd.Series, y_pred: np.ndarray, item_count: pd.Series) -> pd.DataFrame:
    """Bucket shipments by item_count and report bias/MAE per bucket."""
    buckets = pd.cut(item_count, bins=[0, 2, 5, 9, 100], labels=["1-2", "3-5", "6-9", "10+"])
    return bias_by_segment(y_true, y_pred, buckets, "item_count_bucket")


def largest_errors(X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, n: int = 20) -> pd.DataFrame:
    out = X.copy()
    out["actual_weight_oz"] = y_true.values
    out["predicted_weight_oz"] = y_pred
    out["error_oz"] = y_pred - y_true.values
    out["abs_error_oz"] = out["error_oz"].abs()
    return out.sort_values("abs_error_oz", ascending=False).head(n)

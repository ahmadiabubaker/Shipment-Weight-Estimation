"""Evaluation utilities: error metrics, baseline comparison, segment bias, large errors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


LBS_TO_OZ = 16.0

# Tolerance bands reported alongside MAE/RMSE. The pound-denominated bands are
# the operationally meaningful ones -- a picker cares whether the estimate is
# within half a pound, not within half an ounce.
TOLERANCE_BANDS_OZ: dict[str, float] = {
    "within_0_5oz_pct": 0.5,
    "within_2oz_pct": 2.0,
    "within_0_3lb_pct": 0.3 * LBS_TO_OZ,   # 4.8 oz
    "within_0_5lb_pct": 0.5 * LBS_TO_OZ,   # 8.0 oz
    "within_1lb_pct": 1.0 * LBS_TO_OZ,     # 16.0 oz
}


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    abs_err = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    bias = float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))
    metrics = {"mae_oz": mae, "rmse_oz": rmse, "bias_oz": bias}
    for name, threshold_oz in TOLERANCE_BANDS_OZ.items():
        metrics[name] = float(np.mean(abs_err <= threshold_oz)) * 100
    return metrics


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

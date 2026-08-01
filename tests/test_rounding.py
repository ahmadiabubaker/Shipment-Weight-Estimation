"""Unit tests for the shared carrier billing-tier rounding rule.

The rule itself was verified against real data before implementation (see
scripts/evaluate_rounded.py and shipment_weight.rounding module docstrings);
these tests only check the function applies that already-verified rule
correctly, on both scalar and array input.
"""
import numpy as np
import pytest

from shipment_weight.rounding import round_to_billing_tier


@pytest.mark.parametrize(
    "pred_lbs,expected",
    [
        (12.34, 12.0),   # rounds down to nearest whole pound
        (12.5, 13.0),    # exact half rounds up
        (1.0, 1.0),      # boundary: >= 1lb, stays whole
        (1.499999, 1.0),
    ],
)
def test_rounds_to_nearest_whole_pound_at_or_above_one_lb(pred_lbs, expected):
    rounded, sub_1lb = round_to_billing_tier(pred_lbs)
    assert float(rounded) == pytest.approx(expected)
    assert bool(sub_1lb) is False


@pytest.mark.parametrize("pred_lbs", [0.7, 0.99, 0.999999, 0.0])
def test_leaves_sub_one_lb_unrounded(pred_lbs):
    rounded, sub_1lb = round_to_billing_tier(pred_lbs)
    assert float(rounded) == pytest.approx(pred_lbs)
    assert bool(sub_1lb) is True


def test_works_on_array_input():
    pred_lbs = np.array([12.34, 0.7, 0.99, 1.0, 12.5])
    rounded, sub_1lb = round_to_billing_tier(pred_lbs)
    np.testing.assert_allclose(rounded, [12.0, 0.7, 0.99, 1.0, 13.0])
    np.testing.assert_array_equal(sub_1lb, [False, True, True, False, False])

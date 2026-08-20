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
        (12.34, 13.0),   # ALWAYS rounds up -- never down, never to nearest
        (12.01, 13.0),   # even a tiny fraction rounds all the way up
        (12.99, 13.0),
        (12.5, 13.0),
        (1.0, 1.0),      # boundary: already whole, stays put
        (1.000001, 2.0),
        (1.499999, 2.0),
    ],
)
def test_always_rounds_up_at_or_above_one_lb(pred_lbs, expected):
    rounded, sub_1lb = round_to_billing_tier(pred_lbs)
    assert float(rounded) == pytest.approx(expected)
    assert bool(sub_1lb) is False


@pytest.mark.parametrize("pred_lbs", [0.7, 0.99, 0.999999, 0.0])
def test_leaves_sub_one_lb_unrounded(pred_lbs):
    rounded, sub_1lb = round_to_billing_tier(pred_lbs)
    assert float(rounded) == pytest.approx(pred_lbs)
    assert bool(sub_1lb) is True


def test_works_on_array_input():
    pred_lbs = np.array([12.34, 0.7, 0.99, 1.0, 12.01])
    rounded, sub_1lb = round_to_billing_tier(pred_lbs)
    np.testing.assert_allclose(rounded, [13.0, 0.7, 0.99, 1.0, 13.0])
    np.testing.assert_array_equal(sub_1lb, [False, True, True, False, False])

"""Carrier billing-tier rounding.

Single source of truth for the rounding rule, shared by the serving library
(``shipment_weight.predict``, for the label-facing
``predicted_weight_lbs_for_label`` field) and ``scripts/evaluate_rounded.py``
(for evaluating predictions the same way a carrier's own scale rounds
weight before billing). Moved here so both call sites use the exact same
function instead of two copies drifting apart.

Rule (verified against real data -- see scripts/evaluate_rounded.py's module
docstring for the full audit):
  - actual_weight_lbs >= 1 lb: 80-90%+ of values are exact whole numbers in
    every weight bucket (99% in the 50-100 lb range) -- strong evidence
    carriers round to the nearest pound here. Predictions >= 1 lb are
    rounded to match.
  - actual_weight_lbs < 1 lb: only ~20% of values land on an exact
    quarter-pound (.25/.5/.75); real examples include 0.51, 0.55, 0.78,
    0.98, which aren't quarter-pound multiples. No quarter-pound rounding
    rule holds here, so predictions < 1 lb are left unrounded.
"""
from __future__ import annotations

import numpy as np


def round_to_billing_tier(pred_lbs):
    """Round predicted weight(s) in pounds to the nearest whole pound for
    values >= 1 lb (half rounds up, matching carrier billing convention);
    leave values < 1 lb unrounded.

    Works on a scalar float (returns a numpy scalar) or an array-like
    (returns an array) -- same function either way, since ``np.where``
    handles both. Returns ``(rounded, sub_1lb_mask)``; ``sub_1lb_mask`` is
    True where the input was < 1 lb and therefore left unrounded.
    """
    sub_1lb = pred_lbs < 1.0
    rounded = np.where(sub_1lb, pred_lbs, np.floor(pred_lbs + 0.5))
    return rounded, sub_1lb

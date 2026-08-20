"""Carrier billing-tier rounding.

Single source of truth for the rounding rule, shared by the serving library
(``shipment_weight.predict``, for the label-facing
``predicted_weight_lbs_for_label`` field) and ``scripts/evaluate_rounded.py``
(for evaluating predictions -- and, per Tomas, the actual weight they're
compared against -- the same way a carrier's own scale rounds weight before
billing). Moved here so both call sites use the exact same function instead
of two copies drifting apart.

Rule: for weights >= 1 lb, ALWAYS round up to the next whole pound, never
down or to nearest -- e.g. 9.1 -> 10, 9.99 -> 10; only an already-whole
value (9.0) stays put. This is real carrier convention (Tomas's example was
USPS, but it's not USPS-specific -- verified in this dataset by comparing
the same carrier's weight with and without the "(Perseuss)" system tag:
plain-tagged rows are 97-99% already-whole [i.e. already carrier-rounded],
while the identical carrier's "(Perseuss)"-tagged rows are only 53-66%
whole -- raw, not-yet-rounded scale readings from that capture system, not
a different rounding rule by carrier). Weights < 1 lb are left unrounded --
a quarter-pound rounding rule was tested there and does not hold (real
examples: 0.51, 0.55, 0.78, 0.98, none of which are quarter-pound
multiples).

Applies to BOTH sides of any label-matching comparison: our predictions
(so the label reflects what the carrier would actually charge for that
estimate) and the actual weight itself when it's a raw, not-yet-rounded
reading (so "does the label match the bill" is evaluated against what was
truly billed, not an artifact of which system captured the number).
"""
from __future__ import annotations

import numpy as np


def round_to_billing_tier(pred_lbs):
    """Round weight(s) in pounds UP to the next whole pound for values
    >= 1 lb (never down, never to nearest -- see module docstring for why);
    leave values < 1 lb unrounded.

    Works on a scalar float (returns a numpy scalar) or an array-like
    (returns an array) -- same function either way, since ``np.where``
    handles both. Returns ``(rounded, sub_1lb_mask)``; ``sub_1lb_mask`` is
    True where the input was < 1 lb and therefore left unrounded.

    Used for both predicted weights and actual/ground-truth weights --
    same function, same rule, either side of the comparison.
    """
    sub_1lb = pred_lbs < 1.0
    rounded = np.where(sub_1lb, pred_lbs, np.ceil(pred_lbs))
    return rounded, sub_1lb

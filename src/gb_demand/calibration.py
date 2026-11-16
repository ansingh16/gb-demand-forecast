"""Prediction intervals that hold their stated coverage (notebooks 07 and 08).

Validation compared conformalized quantile regression with a simpler method: the point
forecast plus quantiles of its own recent out-of-sample errors, separately for each half-hour
of the day. The simpler method had the better 90% interval score, so it is the one used.
"""

import numpy as np
import pandas as pd

from .config import INTERVALS


def conformal_shift(scores, nominal):
    """Split-conformal adjustment: the finite-sample-corrected quantile of the scores."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * nominal) / n)
    return np.quantile(scores, level, method="higher")


def conformalize(frame, history, low, high):
    """Conformalized quantile regression for one interval, with one adjustment for the day."""
    y = history["Demand_Outturn"]
    score = np.maximum(history[f"q{low}"] - y, y - history[f"q{high}"])
    shift = conformal_shift(score, high - low)
    return frame[f"q{low}"] - shift, frame[f"q{high}"] + shift


def error_quantiles(history, intervals=INTERVALS):
    """Quantiles of recent errors (outturn minus point forecast) for each half-hour slot."""
    errors = (history["Demand_Outturn"] - history["point"]).groupby(history["slot"])
    levels = sorted({q for pair in intervals for q in pair})
    return pd.DataFrame({q: errors.quantile(q) for q in levels})


def apply_intervals(frame, table):
    """Add calibrated interval ends c{q} = point + error quantile for the row's slot."""
    out = frame.copy()
    for q in table.columns:
        out[f"c{q}"] = out["point"] + out["slot"].map(table[q])
    return out


def reference(frame, history, intervals=INTERVALS):
    """Calibrate frame with errors from history (the method chosen on validation)."""
    return apply_intervals(frame, error_quantiles(history, intervals))

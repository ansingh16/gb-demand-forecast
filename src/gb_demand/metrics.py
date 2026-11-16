"""Scores for point and interval forecasts (notebooks 02, 07 and 08)."""

import numpy as np
import pandas as pd


def scores(forecast, actual):
    """MAE, RMSE, MAPE (%) and bias, all in MW except MAPE."""
    error = np.asarray(forecast, dtype=float) - np.asarray(actual, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return pd.Series({
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "MAPE": (np.abs(error) / actual).mean() * 100,
        "bias": error.mean(),
    })


def pinball(y, f, q):
    """Mean pinball (quantile) loss of forecast f for quantile q."""
    diff = np.asarray(y, dtype=float) - np.asarray(f, dtype=float)
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def coverage(y, lo, hi):
    """Share of outcomes inside [lo, hi]."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return ((y >= lo) & (y <= hi)).mean()


def interval_score(y, lo, hi, nominal):
    """Interval (Winkler) score: width plus 2/alpha times the miss. Lower is better."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    alpha = 1 - nominal
    miss = np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None)
    return ((hi - lo) + 2 / alpha * miss).mean()

"""What is known when the forecast is made (notebook 04).

The forecast for a local target day is made at 08:30 UTC on the day before. NESO publishes
its day-ahead forecast between 08:33 and 10:33 UTC, so this origin is never later than NESO's.
"""

import pandas as pd

ORIGIN_TIME = pd.Timedelta(hours=8, minutes=30)


def forecast_origin(target_date):
    """UTC moment the forecast for a local target date is made: 08:30 UTC the day before."""
    day_before = pd.Timestamp(target_date).normalize() - pd.Timedelta(days=1)
    return (day_before + ORIGIN_TIME).tz_localize("UTC")


def latest_demand_allowed(target_date):
    """Start (UTC) of the latest live demand (INDO) period the forecast may use.

    INDO is published at the end of each period. The period ending at 08:30 is published at
    08:30 exactly, with no room for a late update, so the rule stops one period earlier: the
    latest usable period starts at 07:30 and ends at 08:00.
    """
    return forecast_origin(target_date) - pd.Timedelta(hours=1)

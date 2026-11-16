"""Settlement periods and time zones (notebooks 01 and 02)."""

import pandas as pd

LONDON = "Europe/London"
DATE_FORMATS = ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d")


def period_start_utc(dates, periods):
    """UTC start of each settlement period, given local dates and period numbers.

    Adding minutes to a time-zone-aware local midnight counts real elapsed time, so days
    with 46 or 50 periods (clock changes) need no special case.
    """
    local_midnight = pd.to_datetime(pd.Series(dates)).dt.tz_localize(LONDON)
    offset = pd.to_timedelta((pd.Series(periods).to_numpy() - 1) * 30, unit="min")
    return (local_midnight + offset).dt.tz_convert("UTC")


def parse_settlement_dates(values):
    """Parse one file's dates, using the single format that fits every row."""
    for date_format in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=date_format, errors="coerce")
        if parsed.notna().all():
            return parsed
    raise ValueError(f"No known date format fits, for example {values.iloc[0]!r}")


def local_naive(utc):
    """UTC timestamps to local clock time without a time zone."""
    return utc.dt.tz_convert(LONDON).dt.tz_localize(None)


def local_day_utc(target_date):
    """UTC start and end of a local day (the end is the next local midnight)."""
    day = pd.Timestamp(target_date)
    start = day.tz_localize(LONDON).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(LONDON).tz_convert("UTC")
    return start, end

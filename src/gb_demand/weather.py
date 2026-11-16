"""Weather: Open-Meteo responses to one value per settlement period (notebook 04)."""

import pandas as pd

from .config import VARIABLES


def to_long(payload, cities, suffix=""):
    """Open-Meteo multi-location response to one long table (one row per city and hour)."""
    frames = []
    for city, item in zip(cities, payload, strict=True):
        hourly = pd.DataFrame(item["hourly"])
        hourly.columns = [c.replace(suffix, "") for c in hourly.columns]
        hourly["city"] = city
        frames.append(hourly)
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"]).dt.tz_localize("UTC")
    return out


def to_half_hours(hourly):
    """Equal-weight mean over locations, then one row per settlement period (UTC start).

    Temperature and wind are values at the hour, so each half-hour takes the value at its
    midpoint by straight-line interpolation. Radiation is the mean over the hour *before*
    the timestamp, so a half-hour starting at 12:00 or 12:30 takes the 13:00 value.
    """
    mean = hourly.groupby("time")[VARIABLES].mean()
    grid = pd.date_range(mean.index.min(), mean.index.max() - pd.Timedelta(minutes=30), freq="30min")
    out = pd.DataFrame(index=grid)

    midpoints = grid + pd.Timedelta(minutes=15)
    for column in ["temperature_2m", "wind_speed_10m"]:
        series = mean[column].reindex(mean.index.union(midpoints)).interpolate("time")
        out[column] = series.reindex(midpoints).values

    covering_hour = grid.floor("h") + pd.Timedelta(hours=1)
    out["shortwave_radiation"] = mean["shortwave_radiation"].reindex(covering_hour).values
    return out.rename_axis("start_utc").reset_index()
